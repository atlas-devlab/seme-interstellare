"""Send and run a trained digit classifier through the local SIF1 channel."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from demo import bounded_read, save_json, simulate_channel, write_atomic
from macchina import MAX_CAPSULE_BYTES, read_json
from modello_ai import load_model, predict
import trame
import trasporto

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/cifre.json'
EXAMPLES = ROOT / 'models/cifre-esempi.json'
MAX_TEST_FILE = 1024 * 1024
MAX_TEST_ROWS = 2000


def load_examples(path):
    document = read_json(bounded_read(path, MAX_CAPSULE_BYTES))
    if type(document) is not dict or type(document.get('examples')) is not list:
        raise ValueError('File di esempi non valido')
    return validate_examples(document['examples'])


def validate_examples(examples):
    if not 1 <= len(examples) <= MAX_TEST_ROWS:
        raise ValueError('Numero di esempi fuori limite')
    for row in examples:
        if (type(row) is not dict or type(row.get('label')) is not int
                or not 0 <= row['label'] <= 9 or type(row.get('pixels')) is not list):
            raise ValueError('Esempio di cifra non valido')
    return examples


def load_test_split(path):
    rows = csv.reader(bounded_read(path, MAX_TEST_FILE).decode('ascii').splitlines())
    examples = []
    for index, row in enumerate(rows):
        if len(row) != 65 or index >= MAX_TEST_ROWS:
            raise ValueError('Ogni riga deve avere 64 valori e una cifra. Massimo 2.000 righe.')
        values = [int(value.strip()) for value in row]
        examples.append(dict(index=index, pixels=values[:-1], label=values[-1]))
    return validate_examples(examples)


def predictions(model, examples):
    return [predict(model, example['pixels']) for example in examples]


def evaluate(model, examples):
    predicted = predictions(model, examples)
    correct = sum(a == e['label'] for a, e in zip(predicted, examples))
    return dict(examples=len(examples), correct=correct, accuracy=correct/len(examples),
                predictions=predicted)


def receive(wire, examples):
    return infer_recovery(trame.recover(wire), examples)


def infer_recovery(recovered, examples):
    recovered = dict(recovered)
    payload_hex = recovered.pop('payload_hex')
    if recovered['status'] != 'complete':
        return recovered, None, None
    restored = bytes.fromhex(payload_hex)
    return recovered, restored, evaluate(load_model(restored), examples)


def make_frames(payload, seed, *, serializer=trame.frame):
    transmission = trasporto.encode(payload, block_bytes=128, repair_multiplier=2, seed=seed)
    manifest, packets = transmission['manifest'], transmission['packets']
    return manifest, [serializer(manifest, packet) for packet in packets]


def campaign(repetitions=24):
    payload = bounded_read(MODEL, MAX_CAPSULE_BYTES)
    cases = []
    for loss in (0.0, 0.35, 0.60, 0.65, 0.70, 1.0):
        complete = 0
        rejected = 0
        for seed in range(repetitions):
            _, frames = make_frames(payload, seed)
            wire, _, _ = simulate_channel(frames, loss, 0.05, seed + 1980)
            recovered = trame.recover(wire)
            if recovered['status'] == 'complete':
                if bytes.fromhex(recovered['payload_hex']) != payload:
                    raise RuntimeError('Byte diversi dopo il recupero')
                complete += 1
            elif recovered['status'] != 'need_more' or recovered['payload_hex'] is not None:
                raise RuntimeError(f'Esito inatteso nella campagna: {recovered["status"]}')
            rejected += recovered['framing']['rejected_frames']
        cases.append(dict(loss_probability=loss, cases=repetitions, recovered_exactly=complete,
                          need_more=repetitions-complete, corrupted_frames_rejected=rejected))
    return dict(format='SIF-NN-campaign/1', model_sha256=hashlib.sha256(payload).hexdigest(),
                corruption_probability=0.05, trials=len(cases)*repetitions, cases=cases)


def demo(directory, loss=0.60, corruption=0.05, seed=19770815,
         repairs_only=False, test_path=None):
    payload = bounded_read(MODEL, MAX_CAPSULE_BYTES)
    examples = load_test_split(test_path) if test_path else load_examples(EXAMPLES)
    original = load_model(payload)
    manifest, frames = make_frames(payload, seed)
    source_blocks = manifest['block_count']
    if repairs_only:
        repairs = frames[source_blocks:2*source_blocks]
        random.Random(seed).shuffle(repairs)
        wire = b'LOCAL_CHANNEL_NOISE\x00\xff' + b''.join(repairs)
        lost, altered = len(frames)-len(repairs), 0
    else:
        wire, lost, altered = simulate_channel(frames, loss, corruption, seed)
    reconstruction, restored, result = receive(wire, examples)
    identical = None
    if restored is not None:
        identical = restored == payload and result['predictions'] == predictions(original, examples)
        if not identical:
            raise RuntimeError('La ricostruzione o le predizioni differiscono dall’originale')
    report = dict(
        format='SIF-NN-demo/1', completed_utc=datetime.now(timezone.utc).isoformat(),
        model_bytes=len(payload), model_sha256=hashlib.sha256(payload).hexdigest(),
        source_blocks=source_blocks, frames_sent=len(frames), frames_lost=lost,
        frames_corrupted=altered, simulated_channel=True,
        profile='repair_only' if repairs_only else 'random_loss',
        configured_loss_probability=None if repairs_only else loss,
        configured_corruption_probability=None if repairs_only else corruption,
        seed=seed, reconstruction=reconstruction, inference=result,
        evaluation_set=test_path.name if test_path else '20 selected held-out UCI examples',
        examples_sha256=hashlib.sha256(trame.canonical(examples)).hexdigest(),
        exact_model_and_predictions_match=identical,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'risultato.json').unlink(missing_ok=True)
    (directory/'cifre_ricostruito.json').unlink(missing_ok=True)
    write_atomic(directory/'ricevuto.bin', wire)
    if restored is not None:
        write_atomic(directory/'cifre_ricostruito.json', restored)
    save_json(directory/'risultato.json', report)
    return report


def cli_summary(report):
    recovery = report['reconstruction']
    print(f"Modello: {report['model_bytes']} byte in {report['source_blocks']} blocchi")
    print(f"Canale: {report['frames_sent']} pacchetti, {report['frames_lost']} persi, "
          f"{report['frames_corrupted']} corrotti")
    print(f"Ricostruzione: {recovery['status']} (rango {recovery.get('rank',0)}/{report['source_blocks']})")
    metrics = report['inference']
    if metrics is not None:
        print(f"Inferenza locale: {metrics['correct']}/{metrics['examples']} cifre corrette "
              f"({metrics['accuracy']:.2%}). Modello e predizioni identici all'originale.")
    else:
        print('Inferenza non avviata. Servono altri pacchetti validi e indipendenti.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command')
    run = commands.add_parser('demo', help='Invia il modello sul canale simulato e lo esegue')
    run.add_argument('--perdita', type=float, default=0.60)
    run.add_argument('--corruzione', type=float, default=0.05)
    run.add_argument('--seed', type=int, default=19770815)
    run.add_argument('--solo-riparazione', action='store_true')
    run.add_argument('--test-set', type=Path, help='File ufficiale UCI optdigits.tes facoltativo')
    run.add_argument('--output', type=Path, default=ROOT/'risultati'/'ai')
    receiver = commands.add_parser('ricevi', help='Esegue il modello usando i pacchetti ricevuti')
    receiver.add_argument('file', type=Path)
    receiver.add_argument('--esempi', type=Path, default=EXAMPLES)
    receiver.add_argument('--test-set', type=Path)
    trial = commands.add_parser('campagna', help='Misura i recuperi esatti con diverse perdite')
    trial.add_argument('--output', type=Path, default=ROOT/'risultati'/'campagna-ai.json')
    args = parser.parse_args()
    try:
        if args.command == 'ricevi':
            examples = load_test_split(args.test_set) if args.test_set else load_examples(args.esempi)
            recovered, _, result = receive(bounded_read(args.file, trame.MAX_WIRE_BYTES), examples)
            print(json.dumps(dict(reconstruction=recovered, inference=result), indent=2))
        elif args.command == 'demo':
            cli_summary(demo(args.output, args.perdita, args.corruzione, args.seed,
                             args.solo_riparazione, args.test_set))
        elif args.command == 'campagna':
            report = campaign()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.output, report)
            for case in report['cases']:
                print(f"Perdita {case['loss_probability']:.0%}: "
                      f"{case['recovered_exactly']}/{case['cases']} recuperi esatti")
        else:
            cli_summary(demo(ROOT/'risultati'/'ai'))
    except (ValueError, OSError) as error:
        parser.exit(2, f'Errore: {error}\n')


if __name__ == '__main__':
    main()
