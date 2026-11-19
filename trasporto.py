"""Bounded local XOR repair transport; integrity is not sender authentication.

The first k repair packets form an invertible upper-triangular binary matrix.
The receiver returns no bytes until all source blocks are uniquely determined
and the original length, zero padding and SHA-256 are verified.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import zlib

MAX_PAYLOAD_BYTES = 256 * 1024
MAX_BLOCKS = 256
MAX_BLOCK_BYTES = 4096
MAX_PACKETS = 1024
_MANIFEST_KEYS = {'version', 'payload_length', 'block_bytes', 'block_count', 'payload_sha256'}
_PACKET_KEYS = {'id', 'kind', 'mask_hex', 'data_hex', 'crc32'}


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def _manifest(manifest):
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        raise ValueError('Invalid manifest keys')
    _integer(manifest['version'], 1, 1, 'version')
    length = _integer(manifest['payload_length'], 0, MAX_PAYLOAD_BYTES, 'payload_length')
    size = _integer(manifest['block_bytes'], 1, MAX_BLOCK_BYTES, 'block_bytes')
    count = _integer(manifest['block_count'], 1, MAX_BLOCKS, 'block_count')
    if count != max(1, (length + size - 1) // size):
        raise ValueError('Manifest block count and length disagree')
    digest = manifest['payload_sha256']
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('Invalid SHA-256 encoding')
    return length, size, count


def packet_crc(packet):
    """CRC of the canonical id/kind/mask/data object, excluding the CRC itself."""
    fields = {name: packet[name] for name in ('id', 'kind', 'mask_hex', 'data_hex')}
    encoded = json.dumps(fields, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return f'{zlib.crc32(encoded) & 0xffffffff:08x}'


def _xor(first, second):
    return bytes(a ^ b for a, b in zip(first, second))


def _encoded_block(blocks, mask):
    value = bytes(len(blocks[0]))
    for index, block in enumerate(blocks):
        if mask & (1 << index):
            value = _xor(value, block)
    return value


def encode(payload: bytes, block_bytes=128, repair_multiplier=2, seed=19770815) -> dict:
    if type(payload) is not bytes or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError('Payload must be bytes within the 256 KiB limit')
    size = _integer(block_bytes, 1, MAX_BLOCK_BYTES, 'block_bytes')
    multiplier = _integer(repair_multiplier, 1, 3, 'repair_multiplier')
    _integer(seed, 0, 2**64 - 1, 'seed')
    count = max(1, (len(payload) + size - 1) // size)
    if count > MAX_BLOCKS or count * (1 + multiplier) > MAX_PACKETS:
        raise ValueError('Too many blocks or packets; choose a larger block size')
    manifest = dict(version=1, payload_length=len(payload), block_bytes=size,
                    block_count=count, payload_sha256=hashlib.sha256(payload).hexdigest())
    padded = payload.ljust(count * size, b'\0')
    blocks = [padded[i*size:(i+1)*size] for i in range(count)]
    rng = random.Random(seed)
    masks = [1 << index for index in range(count)]
    for index in range(count):
        higher = rng.getrandbits(count-index-1) << (index+1)
        masks.append((1 << index) | higher)
    for _ in range((multiplier-1)*count):
        mask = 0
        while mask == 0:
            mask = rng.getrandbits(count)
        masks.append(mask)
    packets = []
    for identifier, mask in enumerate(masks):
        packet = dict(id=identifier, kind='source' if identifier < count else 'repair',
                      mask_hex=f'{mask:x}', data_hex=_encoded_block(blocks, mask).hex())
        packet['crc32'] = packet_crc(packet)
        packets.append(packet)
    return dict(manifest=manifest, packets=packets)


def _packet(packet, count, size):
    if type(packet) is not dict or set(packet) != _PACKET_KEYS:
        raise ValueError('Invalid packet keys')
    identifier = _integer(packet['id'], 0, MAX_PACKETS-1, 'packet id')
    kind = packet['kind']
    if kind not in ('source', 'repair') or not isinstance(kind, str):
        raise ValueError('Unknown packet kind')
    mask_text, data_text, crc = packet['mask_hex'], packet['data_hex'], packet['crc32']
    if (not isinstance(mask_text, str) or not 1 <= len(mask_text) <= (count+3)//4
            or re.fullmatch('[1-9a-f][0-9a-f]*', mask_text) is None):
        raise ValueError('Invalid mask encoding')
    mask = int(mask_text, 16)
    if mask >= 1 << count:
        raise ValueError('Mask outside block range')
    if kind == 'source' and (identifier >= count or mask != 1 << identifier):
        raise ValueError('Source identifier/mask mismatch')
    if kind == 'repair' and identifier < count:
        raise ValueError('Repair identifier is in source range')
    if (not isinstance(data_text, str) or len(data_text) != 2*size
            or re.fullmatch('[0-9a-f]*', data_text) is None):
        raise ValueError('Invalid data encoding or length')
    if not isinstance(crc, str) or re.fullmatch('[0-9a-f]{8}', crc) is None:
        raise ValueError('Invalid CRC encoding')
    if crc != packet_crc(packet):
        raise ValueError('CRC mismatch')
    return identifier, mask, bytes.fromhex(data_text)


def decode(manifest: dict, packets: list[dict]) -> dict:
    length, size, count = _manifest(manifest)
    if type(packets) is not list or len(packets) > MAX_PACKETS:
        raise ValueError('Packets must be a bounded list')
    basis, seen = {}, {}
    rejected = duplicate = 0
    inconsistent = False
    reasons = {}
    for packet in packets:
        try:
            identifier, mask, value = _packet(packet, count, size)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            rejected += 1
            reason = str(error)
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        identity = (mask, value, packet['kind'])
        if identifier in seen:
            if seen[identifier] == identity:
                duplicate += 1
            else:
                inconsistent = True
            continue
        seen[identifier] = identity
        for pivot in sorted(basis):
            if mask & (1 << pivot):
                old_mask, old_value = basis[pivot]
                mask ^= old_mask
                value = _xor(value, old_value)
        if mask == 0:
            if any(value):
                inconsistent = True
            continue
        pivot = (mask & -mask).bit_length()-1
        basis[pivot] = (mask, value)
    rank = len(basis)
    result = dict(status='need_more', rank=rank, required_rank=count,
                  rejected_packets=rejected, duplicate_packets=duplicate,
                  rejected_reasons=reasons, payload_hex=None, payload_sha256=None,
                  integrity_is_authentication=False)
    if inconsistent:
        result.update(status='integrity_failed', reason='Contradictory valid-CRC packets')
        return result
    if rank < count:
        result['reason'] = 'Insufficient independent equations; no free variables guessed'
        return result
    blocks = [None] * count
    for pivot in sorted(basis, reverse=True):
        mask, value = basis[pivot]
        for index in range(pivot+1, count):
            if mask & (1 << index):
                value = _xor(value, blocks[index])
        blocks[pivot] = value
    padded = b''.join(blocks)
    payload = padded[:length]
    digest = hashlib.sha256(payload).hexdigest()
    if any(padded[length:]) or digest != manifest['payload_sha256']:
        result.update(status='integrity_failed', reason='Payload SHA-256 or padding mismatch')
        return result
    result.update(status='complete', payload_hex=payload.hex(), payload_sha256=digest,
                  reason='Unique reconstruction and integrity verified')
    return result
