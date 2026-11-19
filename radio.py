"""Simulate a 72-second complex-baseband observation and receive a trained model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import firwin, oaconvolve

from demo import bounded_read, save_json, write_atomic
import demo_ai
from macchina import MAX_CAPSULE_BYTES
from modello_ai import load_model
import radio_compatto
from radio_medie import integrate_power, same_power_control
from radio_phy import HEADER_BITS, SAMPLE_RATE, SYMBOL_RATE, demodulate, encode_frame
import trame

ROOT = Path(__file__).resolve().parent
DURATION_S = 72
SAMPLES = DURATION_S * SAMPLE_RATE
SOURCE_GAPS = ((9.30, 13.40), (21.20, 24.00), (44.60, 50.80), (57.70, 61.30))
WOW_CLASSES = [6, 14, 26, 30, 19, 5]
BANDWIDTH_HZ = 10_000
BEAM_CENTER_S = 36.2
BEAM_SIGMA_S = 17.5


def channel(frames, seed, snr_db, gaps=True, offset_hz=140):
    """Transmit baseband bursts with a Gaussian power gain and receiver noise."""
    if snr_db is not None and (not math.isfinite(snr_db) or not -40 <= snr_db <= 40):
        raise ValueError('SNR fuori intervallo [-40, 40] dB')
    if not math.isfinite(offset_hz) or abs(offset_hz) > 1000:
        raise ValueError('Lo scarto di frequenza deve essere tra -1000 e 1000 Hz')
    rng = np.random.default_rng(seed)
    time = np.arange(SAMPLES, dtype=np.float64) / SAMPLE_RATE
    envelope = np.exp(-0.5 * ((time - BEAM_CENTER_S) / BEAM_SIGMA_S) ** 2)
    ideal = np.zeros(SAMPLES, dtype=np.complex64)
    bursts = [encode_frame(frame) for frame in frames]
    slot = (SAMPLES - SAMPLE_RATE // 2) // len(bursts)
    if max(map(len, bursts)) + 36 >= slot:
        raise ValueError('I frame non entrano nei 72 secondi alla velocità scelta')
    starts = []
    phase = rng.uniform(0, 2 * np.pi)
    for index, burst in enumerate(bursts):
        start = SAMPLE_RATE // 4 + index * slot + int(rng.integers(0, 36))
        stop = start + len(burst)
        rotation = np.exp(1j * (phase + 2 * np.pi * offset_hz * time[start:stop]))
        ideal[start:stop] = burst * np.sqrt(envelope[start:stop]) * rotation
        starts.append((start, stop))

    taps = firwin(129, BANDWIDTH_HZ / 2, fs=SAMPLE_RATE).astype(np.float32)
    ideal_filtered = oaconvolve(ideal, taps, mode='same')
    interrupted = ideal.copy()
    intervals = [(round(a * SAMPLE_RATE), round(b * SAMPLE_RATE)) for a, b in SOURCE_GAPS] if gaps else []
    for start, stop in intervals:
        interrupted[start:stop] = 0
    erased = sum(any(a <= start and stop <= b for a, b in intervals) for start, stop in starts)
    touched = sum(any(start < b and a < stop for a, b in intervals) for start, stop in starts)

    # E[|n|^2] is the complex sample noise power before the channel filter.
    noise_variance = 0.0 if snr_db is None else 10 ** (-snr_db / 10)
    sigma = np.float32(math.sqrt(noise_variance / 2))
    noise_hash = None
    if sigma:
        digest = hashlib.sha256()
        for component in (interrupted.real, interrupted.imag):
            samples = sigma * rng.standard_normal(SAMPLES, dtype=np.float32)
            digest.update(samples.astype('<f4', copy=False).tobytes())
            component += samples
        noise_hash = digest.hexdigest()
    observed = oaconvolve(interrupted, taps, mode='same').astype(np.complex64, copy=False)
    noise_filtered = noise_variance * float(np.dot(taps, taps))
    return observed, {
        'sample_rate_hz': SAMPLE_RATE, 'symbol_rate_baud': SYMBOL_RATE, 'duration_s': DURATION_S,
        'channel_filter_hz': BANDWIDTH_HZ, 'filter_taps': len(taps), 'peak_sample_snr_db_before_filter': snr_db,
        'noise_power_after_filter': noise_filtered, 'noise_samples_sha256': noise_hash,
        'frequency_offset_hz': offset_hz, 'initial_phase_rad': phase,
        'packet_start_samples': [start for start, _ in starts],
        'beam_center_s': BEAM_CENTER_S, 'beam_sigma_s': BEAM_SIGMA_S,
        'gap_intervals_s': SOURCE_GAPS if gaps else [],
        'gap_duration_s': sum(b-a for a, b in SOURCE_GAPS) if gaps else 0,
        'frames_fully_inside_gaps': erased, 'frames_intersecting_gaps': touched,
        'power_ideal': integrate_power(ideal_filtered, SAMPLE_RATE),
        'power_observed': integrate_power(observed, SAMPLE_RATE, noise_filtered),
    }


def receive(iq, examples):
    """Use sampled I/Q and the public modem contract. No transmitter metadata."""
    radio = demodulate(iq)
    reconstruction, restored, metrics = decode_frames(radio['frames'], examples)
    modem = {key: value for key, value in radio.items() if key != 'frames'}
    return modem, reconstruction, restored, metrics


def decode_frames(frames, examples):
    magics = {frame[:4] for frame in frames}
    if magics == {radio_compatto.MAGIC}:
        return demo_ai.infer_recovery(radio_compatto.recover(frames), examples)
    if len(magics) > 1:
        return dict(status='format_conflict', reason='Mixed frame formats in one capture'), None, None
    return demo_ai.receive(b''.join(frames), examples)


def make_frames(payload, seed, compact=False):
    serializer = radio_compatto.frame if compact else trame.frame
    _, frames = demo_ai.make_frames(payload, seed, serializer=serializer)
    return frames


def wire_bytes(frames):
    return sum(len(frame) + HEADER_BITS//8 for frame in frames)


def run(output, snr_db, gaps, seed, offset_hz, examples, save_iq=False, compact=False):
    original = bounded_read(demo_ai.MODEL, MAX_CAPSULE_BYTES)
    encoded = make_frames(original, seed, compact)
    iq, channel_report = channel(encoded, seed, snr_db, gaps, offset_hz)
    modem, reconstruction, restored, inference = receive(iq, examples)
    same = None
    if restored is not None:
        expected = demo_ai.predictions(load_model(original), examples)
        same = restored == original and inference['predictions'] == expected
        if not same:
            raise RuntimeError('Il modello ricostruito o le sue predizioni differiscono')
    report = dict(
        format='SIF-radio-demo/1', completed_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, simulated=True, frames_sent=len(encoded), channel=channel_report,
        frame_format='SIC1' if compact else 'SIF1', wire_bytes=wire_bytes(encoded),
        modem=modem, reconstruction=reconstruction, inference=inference,
        exact_model_and_predictions_match=same, model_bytes=len(original),
        model_sha256=hashlib.sha256(original).hexdigest(),
        published_wow_classes_for_shape_only=WOW_CLASSES,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / 'risultato.json').unlink(missing_ok=True)
    (output / 'cifre_ricostruito.json').unlink(missing_ok=True)
    (output / 'ricevuto_iq.npy').unlink(missing_ok=True)
    if restored is not None:
        write_atomic(output / 'cifre_ricostruito.json', restored)
    if save_iq:
        np.save(output / 'ricevuto_iq.npy', iq, allow_pickle=False)
    save_json(output / 'risultato.json', report)
    return report


def print_summary(report):
    rec = report['reconstruction']
    modem = report['modem']
    result = report['inference']
    print(f"Radio: {modem['crc_valid_frames']}/{report['frames_sent']} frame con CRC valido")
    print(f"Recupero: {rec['status']} (rango {rec.get('rank', 0)}/{rec.get('required_rank') or 'sconosciuto'})")
    if result is not None:
        print(f"Inferenza: {result['correct']}/{result['examples']} corrette, {result['accuracy']:.2%}")
        print('Byte del modello e predizioni identici al riferimento.')
    else:
        print('Inferenza non avviata: mancano equazioni indipendenti valide.')


def plot(path, report):
    import matplotlib.pyplot as plt

    data = report['channel']
    fig, (ax, lower) = plt.subplots(2, 1, figsize=(10, 6.3), layout='constrained')
    for key, label, color in [('power_ideal', 'Senza rumore né vuoti', '#168673'),
                              ('power_observed', 'Con rumore e vuoti (fondo sottratto)', '#255a92')]:
        bins = data[key]
        ax.plot([(v['start_s'] + v['end_s'])/2 for v in bins], [v['excess_power'] for v in bins],
                '-o', label=label, color=color)
    for i, (start, stop) in enumerate(data['gap_intervals_s']):
        ax.axvspan(start, stop, color='#ec9e46', alpha=.25, label='Vuoto simulato' if not i else None)
    ax.set(xlim=(0, DURATION_S), ylabel='Potenza media relativa', xlabel='Tempo (s)',
           title='Sei medie di potenza nei campioni radio simulati')
    ax.legend(loc='best')
    times = np.array(report['modem']['sample_offsets']) / SAMPLE_RATE
    lower.eventplot(times, lineoffsets=0, linelengths=.65, color='#168673')
    for start, stop in data['gap_intervals_s']:
        lower.axvspan(start, stop, color='#ec9e46', alpha=.25)
    lower.set(xlim=(0, DURATION_S), yticks=[], xlabel='Inizio del pacchetto (s)',
              title='Pacchetti ricevuti con CRC valido')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    tx = modes.add_parser('prova')
    tx.add_argument('--snr', type=float, default=8)
    tx.add_argument('--senza-rumore', action='store_true')
    tx.add_argument('--senza-vuoti', action='store_true')
    tx.add_argument('--offset-hz', type=float, default=140)
    tx.add_argument('--seed', type=int, default=19770815)
    tx.add_argument('--salva-iq', action='store_true')
    tx.add_argument('--grafico', action='store_true')
    tx.add_argument('--compatto', action='store_true', help='Usa frame binari SIC1')
    tx.add_argument('--test-set', type=Path)
    tx.add_argument('--output', type=Path, default=ROOT/'risultati'/'radio')
    rx = modes.add_parser('ricevi')
    rx.add_argument('file', type=Path)
    rx.add_argument('--test-set', type=Path)
    rx.add_argument('--esempi', type=Path, default=demo_ai.EXAMPLES)
    power = modes.add_parser('controllo-medie')
    power.add_argument('--output', type=Path, default=ROOT/'risultati'/'radio-medie.json')
    args = parser.parse_args()

    try:
        if args.mode == 'controllo-medie':
            report = same_power_control(SAMPLE_RATE)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.output, report)
            print(json.dumps(report, ensure_ascii=True))
            return
        examples = (demo_ai.load_test_split(args.test_set) if args.test_set else
                    demo_ai.load_examples(getattr(args, 'esempi', demo_ai.EXAMPLES)))
        if args.mode == 'ricevi':
            if args.file.stat().st_size > SAMPLES * 8 + 4096:
                raise ValueError('File I/Q oltre il limite di dimensione')
            iq = np.load(args.file, allow_pickle=False, mmap_mode='r')
            if not isinstance(iq, np.ndarray) or iq.dtype != np.dtype(np.complex64) or iq.shape != (SAMPLES,):
                raise ValueError('Attesi 72 s di campioni I/Q complex64 a 48.000 campioni/s')
            modem, reconstruction, _, inference = receive(iq, examples)
            print(json.dumps(dict(modem=modem, reconstruction=reconstruction, inference=inference)))
            return
        report = run(args.output, None if args.senza_rumore else args.snr,
                     not args.senza_vuoti, args.seed, args.offset_hz, examples, args.salva_iq, args.compatto)
        print_summary(report)
        if args.grafico:
            plot(args.output/'profilo_radio.png', report)
    except (ValueError, OSError) as error:
        parser.exit(2, f'Errore: {error}\n')


if __name__ == '__main__':
    main()
