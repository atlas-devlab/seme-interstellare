"""Build, impair and receive an authored interstellar-seed demonstration locally."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile

import macchina
import trame
import trasporto

ROOT = Path(__file__).resolve().parent


def save_json(path,value):
    write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def write_atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def bounded_read(path,maximum):
    with path.open('rb') as source:
        data = source.read(maximum+1)
    if len(data)>maximum:
        raise ValueError('File exceeds size limit')
    return data


def reference_fibonacci(first,second,count):
    # Independent receiver-side check used by the authored demonstration only.
    output = []
    for _ in range(count):
        output.append(first)
        first,second = second,(first+second)&macchina.WORD_MASK
    return output


def simulate_channel(frames, loss, corruption, seed):
    if not 0<=loss<=1 or not 0<=corruption<=1:
        raise ValueError('Loss and corruption must be in [0,1]')
    rng = random.Random(seed)
    received, lost, altered = [],0,0
    for item in frames:
        if rng.random()<loss:
            lost += 1
            continue
        if rng.random()<corruption:
            changed = bytearray(item)
            # Affect the body and let outer CRC detect it.
            changed[rng.randrange(8,len(changed)-4)] ^= 1<<rng.randrange(8)
            item = bytes(changed)
            altered += 1
        received.append(item)
    rng.shuffle(received)
    # Some irrelevant bytes precede the signal; framing must find the first sync.
    impaired = b'LOCAL_CHANNEL_NOISE\x00\xff'+b''.join(received)
    return impaired, lost, altered


def demonstration(directory,loss=0.35,corruption=0.05,seed=19770815,first=0,second=1,terms=16):
    payload = macchina.make_seed(first,second,terms)
    transmission = trasporto.encode(payload,block_bytes=128,repair_multiplier=2,seed=seed)
    manifest,packets = transmission['manifest'],transmission['packets']
    frames = [trame.frame(manifest,packet) for packet in packets]
    impaired, lost, altered = simulate_channel(frames, loss, corruption, seed)
    recovered = trame.recover(impaired)
    compact = {key:value for key,value in recovered.items() if key!='payload_hex'}
    execution = None
    if recovered['status']=='complete':
        restored = bytes.fromhex(recovered['payload_hex'])
        execution = macchina.receive_capsule(restored)
        if execution['output'] != reference_fibonacci(first,second,terms):
            raise RuntimeError('Independent program-output check failed')
        if restored != payload:
            raise RuntimeError('Capsule does not match original source bytes')
    report = dict(format='SIF-local-demo-1',completed_utc=datetime.now(timezone.utc).isoformat(),
        content='Authored algorithm, not an AI model or recovered Wow message.',
        seed_bytes=len(payload),seed_sha256=hashlib.sha256(payload).hexdigest(),
        source_blocks=manifest['block_count'],packets_sent=len(packets),
        packets_lost=lost,frames_altered=altered,frames_received=len(frames)-lost,
        loss_probability=loss,corruption_probability_among_survivors=corruption,
        random_seed=seed,reconstruction=compact,execution=execution,
        source_bytes_match=True if execution else None,
        independent_output_check_passed=True if execution else None,
        bootstrap_assumptions=['Known SIF1 framing','UTF-8 JSON','uint32 stack primitives'],
        network_transmission_performed=False,real_AI_transported=False,Wow_decoding=False)
    directory.mkdir(parents=True,exist_ok=True)
    # The report is the completion marker and is installed last.
    (directory/'risultato.json').unlink(missing_ok=True)
    (directory/'seme_ricostruito.json').unlink(missing_ok=True)
    write_atomic(directory/'seme.json', payload)
    write_atomic(directory/'trasmissione.bin', b''.join(frames))
    write_atomic(directory/'ricevuto.bin', impaired)
    if recovered['status']=='complete':
        write_atomic(directory/'seme_ricostruito.json', bytes.fromhex(recovered['payload_hex']))
    save_json(directory/'risultato.json',report)
    return report


def print_demo(report):
    print('SEME INTERSTELLARE - DIMOSTRAZIONE LOCALE')
    print(f"Seme: {report['seed_bytes']} byte; blocchi: {report['source_blocks']}")
    print(f"Pacchetti: {report['packets_sent']} inviati, {report['packets_lost']} persi, "
          f"{report['frames_altered']} alterati")
    result = report['reconstruction']
    print(f"Ricostruzione: {result['status']}; rango {result.get('rank',0)}/{result.get('required_rank','?')}")
    if report['execution']:
        print('Risultato del programma: '+', '.join(map(str,report['execution']['output'])))
        print('Capsula identica all\'originale e risultato verificato indipendentemente.')
    else:
        print('Nessun programma eseguito: informazioni insufficienti o verifica fallita.')
    print('Contenuto costruito per la prova; non contiene un modello AI.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command')
    demo = commands.add_parser('demo',help='Simulate a local transmitter and receiver')
    demo.add_argument('--perdita',type=float,default=0.35)
    demo.add_argument('--corruzione',type=float,default=0.05)
    demo.add_argument('--seed',type=int,default=19770815)
    demo.add_argument('--primo',type=int,default=0)
    demo.add_argument('--secondo',type=int,default=1)
    demo.add_argument('--termini',type=int,default=16)
    demo.add_argument('--output',type=Path,default=ROOT/'risultati'/'demo')
    receive = commands.add_parser('ricevi',help='Recover one capsule and interpret bounded VM instructions')
    receive.add_argument('file',type=Path)
    args = parser.parse_args()
    try:
        if args.command=='ricevi':
            result = trame.recover(bounded_read(args.file,trame.MAX_WIRE_BYTES))
            payload_hex = result.pop('payload_hex',None)
            if result['status']=='complete':
                result['execution'] = macchina.receive_capsule(bytes.fromhex(payload_hex))
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return
        if args.command is None:
            report = demonstration(ROOT/'risultati'/'demo')
        else:
            report = demonstration(args.output,args.perdita,args.corruzione,args.seed,
                                   args.primo,args.secondo,args.termini)
        print_demo(report)
    except (ValueError,OSError) as error:
        parser.exit(2,f'Errore controllato: {error}\n')


if __name__=='__main__':
    main()
