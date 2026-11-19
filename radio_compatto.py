"""Compact binary framing for the existing Seme Interstellare XOR packets.

SIC1 | uint32 BE body length | body | uint32 BE CRC32 of body.
The body is a 43-byte header (version, payload length, block size, block
count, SHA-256, packet ID) followed by a repair mask when needed and raw
block bytes. The receiver derives source masks from the source packet ID.
"""
from __future__ import annotations

from collections import Counter
import struct
import zlib

import trasporto


MAGIC = b'SIC1'
HEADER = struct.Struct('>BIHH32sH')
WORD = struct.Struct('>I')
OUTER_BYTES = len(MAGIC) + 2*WORD.size
MAX_FRAME_BYTES = OUTER_BYTES + HEADER.size + (trasporto.MAX_BLOCKS + 7)//8 + trasporto.MAX_BLOCK_BYTES


def _mask_bytes(count):
    return (count+7)//8


def frame(manifest: dict, packet: dict) -> bytes:
    """Serialize a validated XOR equation without changing its contents."""
    length, size, count = trasporto._manifest(manifest)
    identifier, mask, data = trasporto._packet(packet, count, size)
    header = HEADER.pack(manifest['version'], length, size, count,
                         bytes.fromhex(manifest['payload_sha256']), identifier)
    mask_field = mask.to_bytes(_mask_bytes(count), 'big') if identifier >= count else b''
    body = header + mask_field + data
    return MAGIC + WORD.pack(len(body)) + body + WORD.pack(zlib.crc32(body))


def unpack_frame(data: bytes) -> tuple[dict, dict]:
    """Verify the entire compact frame before returning a transport packet."""
    if type(data) is not bytes or not OUTER_BYTES + HEADER.size + 1 <= len(data) <= MAX_FRAME_BYTES:
        raise ValueError('invalid_compact_frame_size')
    if data[:4] != MAGIC:
        raise ValueError('invalid_compact_magic')
    length_field = WORD.unpack_from(data, 4)[0]
    if length_field != len(data)-OUTER_BYTES:
        raise ValueError('compact_body_length_mismatch')
    body = data[8:-4]
    if zlib.crc32(body) != WORD.unpack_from(data, len(data)-4)[0]:
        raise ValueError('crc_mismatch')

    version, length, size, count, digest, identifier = HEADER.unpack_from(body)
    manifest = dict(version=version, payload_length=length, block_bytes=size,
                    block_count=count, payload_sha256=digest.hex())
    trasporto._manifest(manifest)
    if identifier >= trasporto.MAX_PACKETS:
        raise ValueError('invalid_compact_packet_id')
    source = identifier < count
    body_data_offset = HEADER.size + (0 if source else _mask_bytes(count))
    if len(body) != body_data_offset+size:
        raise ValueError('compact_block_length_mismatch')
    if source:
        mask = 1 << identifier
    else:
        mask = int.from_bytes(body[HEADER.size:body_data_offset], 'big')
        if not 0 < mask < (1 << count):
            raise ValueError('invalid_compact_repair_mask')
    packet = dict(id=identifier, kind='source' if source else 'repair',
                  mask_hex=f'{mask:x}', data_hex=body[body_data_offset:].hex())
    # The outer CRC already covered the transport equation on the wire. Rebuild
    # the canonical in-memory checksum so the existing decoder can validate it.
    packet['crc32'] = trasporto.packet_crc(packet)
    return manifest, packet


def recover(frames: list[bytes]) -> dict:
    """Reassemble an already demodulated list of compact frames."""
    if type(frames) is not list:
        raise ValueError('Compact receiver expects a list of frames')
    if len(frames) > trasporto.MAX_PACKETS:
        return dict(status='scan_limit', payload_hex=None, framing=dict(scan_complete=False))
    records = []
    rejected = Counter()
    for item in frames:
        try:
            records.append(unpack_frame(item))
        except ValueError as error:
            rejected[str(error)] += 1
    stats = dict(wire_bytes=sum(len(item) for item in frames if type(item) is bytes),
                 valid_frames=len(records), rejected_frames=sum(rejected.values()),
                 rejection_reasons=dict(rejected), scan_complete=True,
                 candidate_headers_examined=len(frames))
    if not records:
        return dict(status='need_more', rank=0, required_rank=None, payload_hex=None,
                    reason='No valid compact frame received.', framing=stats)
    manifest = records[0][0]
    if any(item_manifest != manifest for item_manifest, _ in records):
        return dict(status='metadata_conflict', payload_hex=None, framing=stats,
                    reason='Different compact manifests in one transmission.')
    result = trasporto.decode(manifest, [packet for _, packet in records])
    result.update(framing=stats, manifest=manifest)
    return result
