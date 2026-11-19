"""Six synthetic power integrations at the Wow! printout cadence."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _sample_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral):
        raise ValueError('La frequenza di campionamento deve essere un intero positivo')
    if sample_rate < 1:
        raise ValueError('La frequenza di campionamento deve essere positiva')
    return int(sample_rate)


def integrate_power(iq: np.ndarray, sample_rate: int,
                    noise_power: float = 0.0) -> list[dict]:
    """Average |I+jQ|² for 10 seconds, skipping 2 seconds between windows."""
    rate = _sample_rate(sample_rate)
    if not isinstance(iq, np.ndarray) or iq.ndim != 1 or iq.dtype.kind not in 'fciu':
        raise ValueError('Servono campioni numerici I/Q in un vettore')
    if iq.size < 70 * rate:
        raise ValueError('Servono almeno 70 secondi per le sei integrazioni')
    if not isinstance(noise_power, Real) or not np.isfinite(noise_power) or noise_power < 0:
        raise ValueError('La potenza del rumore deve essere finita e non negativa')

    result = []
    for start in range(0, 72, 12):
        end = start + 10
        part = iq[start * rate : end * rate]
        power = np.square(part.real, dtype=np.float64)
        if np.iscomplexobj(part):
            power += np.square(part.imag, dtype=np.float64)
        mean = float(power.mean())
        if not np.isfinite(mean):
            raise ValueError('La finestra contiene campioni non finiti o potenza fuori scala')
        result.append(dict(start_s=start, end_s=end, count=int(part.size),
                           mean_power=mean, excess_power=mean - float(noise_power)))
    return result


def same_power_control(sample_rate: int) -> dict:
    """Compare two different BPSK sequences under the same power envelope.

    Both sequences flip the sign of the carrier at the selected symbol
    boundaries. A sign flip changes the I/Q samples and leaves |I+jQ|² unchanged.
    This is a synthetic control, separate from the frame transmission.
    """
    rate = _sample_rate(sample_rate)
    if rate > 96_000:
        raise ValueError('Il controllo a 72 secondi supporta fino a 96.000 campioni/s')
    duration = 72
    symbol_rate = 20
    rng = np.random.default_rng(19770815)
    signs = 2 * rng.integers(0, 2, size=(2, duration * symbol_rate), dtype=np.int8) - 1
    signals = np.empty((2, duration * rate), dtype=np.complex64)
    offsets = np.arange(rate)
    different_samples = 0
    squared_difference_sum = 0.0
    for second in range(duration):
        time = second + offsets / rate
        envelope = np.exp(-0.5 * ((time - 36.0) / 15.0)**2)
        carrier_phase = (second * rate + offsets) % 8 * (np.pi / 4.0)
        carrier = (envelope * np.exp(1j * carrier_phase)).astype(np.complex64)
        symbol_indices = second * symbol_rate + offsets * symbol_rate // rate
        chunk = signals[:, second * rate : (second + 1) * rate]
        chunk[:] = signs[:, symbol_indices] * carrier
        different_samples += int(np.count_nonzero(chunk[0] != chunk[1]))
        squared_difference_sum += float(np.square(np.abs(chunk[0] - chunk[1]),
                                                 dtype=np.float64).sum())

    first, second = (integrate_power(signal, rate) for signal in signals)
    first_power = [window['mean_power'] for window in first]
    second_power = [window['mean_power'] for window in second]
    differences = [abs(a - b) for a, b in zip(first_power, second_power)]
    tolerance = 1e-12
    return dict(
        duration_s=duration,
        sample_rate_hz=rate,
        demonstration='Different synthetic BPSK sequences, identical instantaneous magnitudes',
        different_symbols=int(np.count_nonzero(signs[0] != signs[1])),
        total_symbols=int(signs.shape[1]),
        different_iq_samples=different_samples,
        total_iq_samples=duration * rate,
        rms_iq_difference=float(np.sqrt(squared_difference_sum / (duration * rate))),
        first_mean_power=first_power,
        second_mean_power=second_power,
        absolute_differences=differences,
        max_abs_power_difference=max(differences),
        absolute_tolerance=tolerance,
        same_window_power=max(differences) <= tolerance and different_samples > 0,
    )
