"""Binary FSK framing and receiver for a local complex baseband simulation."""
from __future__ import annotations

import zlib

import numpy as np

import radio_compatto
import trame

SPS = 8
SYMBOL_RATE = 6000
SAMPLE_RATE = SPS * SYMBOL_RATE
TONE_HZ = SYMBOL_RATE / 2

# A 64-bit fixed pseudo-random synchronisation word, MSB first. It is followed
# by a two-byte big-endian frame length and that many bytes of the SIF1 frame.
PREAMBLE = bytes.fromhex('d39b647a95c7e20d')
SYNC_BITS = len(PREAMBLE) * 8
HEADER_BITS = SYNC_BITS + 16
MAX_SYNC_ERRORS = 6
MAX_SAMPLES = 4_000_000
MAX_SYNC_CANDIDATES = 8192
MIN_FRAME_BYTES = 13  # four magic + four length + at least one body + four CRC
MAX_FRAME_BYTES = trame.MAX_FRAME_BYTES + 12

_TONE_TIMES = np.arange(SPS) / SAMPLE_RATE
_TONE_ONE = np.exp(2j * np.pi * TONE_HZ * _TONE_TIMES).astype(np.complex64)
_TONES = np.stack((_TONE_ONE.conj(), _TONE_ONE))
_MATCHED = _TONES.conj().T.copy()
_SYNC_SIGN = np.unpackbits(np.frombuffer(PREAMBLE, dtype=np.uint8)).astype(np.int8) * 2 - 1


def encode_frame(frame: bytes) -> np.ndarray:
    """Modulate an SIF1 frame using orthogonal ±3000 Hz tones at 48 ksample/s."""
    if type(frame) is not bytes or not MIN_FRAME_BYTES <= len(frame) <= MAX_FRAME_BYTES:
        raise ValueError('Frame type or size is outside the radio format')
    stream = PREAMBLE + len(frame).to_bytes(2, 'big') + frame
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))
    symbols = _TONES[bits]
    # Either tone advances by π or −π over one symbol. The alternating sign
    # gives continuous phase between symbols, regardless of their bit values.
    symbols[1::2] *= -1
    return symbols.reshape(-1)


def _valid_outer_crc(frame: bytes) -> bool:
    if (frame[:4] not in (trame.MAGIC, radio_compatto.MAGIC)
            or int.from_bytes(frame[4:8], 'big') != len(frame) - 12):
        return False
    return zlib.crc32(frame[8:-4]) == int.from_bytes(frame[-4:], 'big')


def _detect_phase(iq: np.ndarray, phase: int) -> tuple[np.ndarray, np.ndarray]:
    count = (iq.size - phase) // SPS
    symbols = iq[phase:phase + count*SPS].reshape(-1, SPS)
    correlations = symbols @ _MATCHED
    bits = (np.abs(correlations[:, 1]) > np.abs(correlations[:, 0])).astype(np.uint8)
    match = np.correlate(bits.astype(np.int8)*2 - 1, _SYNC_SIGN, mode='valid')
    peaks = np.flatnonzero(match >= SYNC_BITS - 2*MAX_SYNC_ERRORS)
    return bits, peaks


def demodulate(iq: np.ndarray) -> dict:
    """Blindly scan eight sample phases with a known symbol clock/preamble.

    Detection is noncoherent (unknown constant RF phase is fine). This version
    assumes the receiver knows the sample rate and nominal tone frequencies.
    There is no carrier/Doppler or symbol clock tracking across a frame.
    """
    if not isinstance(iq, np.ndarray) or iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError('Expected one-dimensional complex I/Q samples')
    if iq.size > MAX_SAMPLES or not np.isfinite(iq).all():
        raise ValueError('I/Q samples exceed the cap or contain non-finite values')
    iq = iq.astype(np.complex64, copy=False)
    candidates = 0
    valid = []
    if iq.size >= (HEADER_BITS+MIN_FRAME_BYTES*8)*SPS:
        for phase in range(SPS):
            bits, peaks = _detect_phase(iq, phase)
            candidates += len(peaks)
            if candidates > MAX_SYNC_CANDIDATES:
                raise ValueError('Too many preamble candidates in the recording')
            for offset in peaks.tolist():
                payload_start = offset + HEADER_BITS
                if payload_start > len(bits):
                    continue
                length = int.from_bytes(np.packbits(bits[offset+SYNC_BITS:payload_start]).tobytes(), 'big')
                payload_end = payload_start + length*8
                if not MIN_FRAME_BYTES <= length <= MAX_FRAME_BYTES or payload_end > len(bits):
                    continue
                frame = np.packbits(bits[payload_start:payload_end]).tobytes()
                if _valid_outer_crc(frame):
                    score = int(np.count_nonzero((bits[offset:offset+SYNC_BITS] != 0) != (_SYNC_SIGN > 0)))
                    valid.append((phase+offset*SPS, score, frame))

    # A single burst may decode at more than one neighbouring sample phase.
    # Collapse those copies with the same bytes, while retaining later repeats.
    copies = {}
    unique = []
    for offset, error_count, frame in sorted(valid, key=lambda row: row[0]):
        previous = copies.get(frame)
        if previous is not None and offset - unique[previous][0] < SPS:
            if error_count < unique[previous][1]:
                unique[previous] = (offset, error_count, frame)
        else:
            copies[frame] = len(unique)
            unique.append((offset, error_count, frame))
    unique.sort(key=lambda row: row[0])
    return dict(frames=[row[2] for row in unique], sync_candidates=candidates,
                crc_valid_frames=len(unique), sample_offsets=[row[0] for row in unique])
