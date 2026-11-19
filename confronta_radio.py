"""Compare JSON and binary packet formats under the same simulated channel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import demo_ai
from demo import save_json
from modello_ai import load_model
import radio
import radio_compatto
from radio_phy import demodulate
import trame

SEED = 19770815
SNR_DB = 8
OFFSET_HZ = 140


def frame_id(frame):
    if frame[:4] == radio_compatto.MAGIC:
        return radio_compatto.unpack_frame(frame)[1]['id']
    records = trame.scan(frame)['records']
    if len(records) != 1:
        raise ValueError('Atteso un frame SIF1 valido')
    return records[0]['packet']['id']


def describe(frames, expected, payload, examples):
    rec, restored, inf = radio.decode_frames(frames, examples)
    exact = None if restored is None else restored == payload and inf['predictions'] == expected
    if restored is not None and not exact:
        raise RuntimeError('Modello o predizioni alterati')
    return dict(status=rec['status'], rank=rec.get('rank', 0),
                required_rank=rec.get('required_rank'), payload_sha256=rec.get('payload_sha256'),
                exact_model_and_predictions_match=exact, inference=inf)


def compare(examples, reference=None, iq_output=None):
    payload = demo_ai.MODEL.read_bytes()
    expected = demo_ai.predictions(load_model(payload), examples)
    runs = {}
    frame_lists = {}
    for mode in ('SIF1', 'SIC1'):
        frames = radio.make_frames(payload, SEED, compact=(mode == 'SIC1'))
        iq, channel = radio.channel(frames, SEED, SNR_DB, True, OFFSET_HZ)
        if mode == 'SIF1' and reference is not None:
            saved = np.load(reference, allow_pickle=False, mmap_mode='r')
            if not np.array_equal(iq, saved):
                raise RuntimeError('La riproduzione SIF1 differisce dai campioni della prova precedente')
        if mode == 'SIC1' and iq_output is not None:
            np.save(iq_output, iq, allow_pickle=False)
        decoded = demodulate(iq)
        received = decoded.pop('frames')
        by_id = {frame_id(frame): frame for frame in received}
        if len(by_id) != len(received):
            raise RuntimeError('Più frame validi dello stesso ID nella prova comparativa')
        sent_ids = [frame_id(frame) for frame in frames]
        sent_by_id = dict(zip(sent_ids, frames))
        if any(value != sent_by_id.get(key) for key, value in by_id.items()):
            raise RuntimeError('Un frame demodulato non coincide col frame trasmesso')
        frame_lists[mode] = by_id
        budget = radio.wire_bytes(frames)
        runs[mode] = dict(wire_bytes_with_radio_headers=budget, airtime_s=budget*8/radio.SYMBOL_RATE,
                          sent_ids=sent_ids, valid_ids=sorted(by_id), modem=decoded, channel=channel,
                          receiver=describe(received, expected, payload, examples))

    baseline, compact = runs['SIF1'], runs['SIC1']
    shared = ['noise_samples_sha256', 'packet_start_samples', 'initial_phase_rad', 'gap_intervals_s',
              'peak_sample_snr_db_before_filter', 'frequency_offset_hz', 'filter_taps']
    if (not baseline['channel']['noise_samples_sha256'] or baseline['sent_ids'] != compact['sent_ids']
            or any(baseline['channel'][key] != compact['channel'][key] for key in shared)):
        raise RuntimeError('Le condizioni del canale non coincidono')

    baseline_ids = baseline['valid_ids']
    compact_frames = frame_lists['SIC1']
    common = [identifier for identifier in baseline_ids if identifier in compact_frames]
    id_test = describe([compact_frames[i] for i in common], expected, payload, examples)
    id_test.update(packet_ids=common, exactly_same_valid_packet_ids=common == baseline_ids,
                   previous_valid_ids_missing_in_compact=sorted(set(baseline_ids)-set(common)),
                   extra_compact_valid_ids_discarded=sorted(set(compact_frames)-set(baseline_ids)))
    first, second = (runs[mode]['wire_bytes_with_radio_headers'] for mode in ('SIF1', 'SIC1'))
    return dict(format='SIF-radio-format-comparison/1', model_bytes=len(payload),
                model_sha256=hashlib.sha256(payload).hexdigest(),
                same_noise_samples_gaps_start_times_and_packet_ids=True,
                historical_iq_matched_sample_for_sample=True if reference is not None else None,
                bytes_saved=first-second, fraction_reduction=(first-second)/first,
                simulation=runs, same_received_ids_control=id_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-set', type=Path)
    parser.add_argument('--riferimento-iq', type=Path)
    parser.add_argument('--salva-iq', action='store_true')
    parser.add_argument('--output', type=Path, default=radio.ROOT/'risultati'/'radio-compatto')
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        examples = (demo_ai.load_test_split(args.test_set) if args.test_set else
                    demo_ai.load_examples(demo_ai.EXAMPLES))
        report = compare(examples, args.riferimento_iq,
                         args.output/'ricevuto_compatto_iq.npy' if args.salva_iq else None)
        save_json(args.output/'confronto.json', report)
        for mode, run in report['simulation'].items():
            print(f"{mode}: {run['wire_bytes_with_radio_headers']} byte, "
                  f"{run['airtime_s']:.2f} s in aria, {len(run['valid_ids'])} frame validi, "
                  f"{run['receiver']['status']} (rango {run['receiver']['rank']})")
        control = report['same_received_ids_control']
        print(f"Controllo stessi ID: {len(control['packet_ids'])} frame in comune, {control['status']}")
        print(f"Riduzione byte: {report['fraction_reduction']:.2%}")
    except (ValueError, OSError) as error:
        parser.exit(2, f'Errore: {error}\n')


if __name__ == '__main__':
    main()
