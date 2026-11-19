"""Local binary framing with repeated metadata and resynchronization.

Wire format: SIF1 | body_length:uint32 BE | canonical UTF-8 JSON | CRC32:uint32 BE.
Each body repeats the full transport manifest and carries one repair/source packet.
The receiver already knows this framing convention. No radio transmission occurs.
"""
from __future__ import annotations

from collections import Counter
import json
import struct
import zlib

from macchina import read_json
import trasporto

MAGIC = b'SIF1'
MAX_FRAME_BYTES = 32768
MAX_WIRE_BYTES = 4*1024*1024
MAX_CANDIDATES = 8192


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')


def frame(manifest, packet):
    body = canonical(dict(manifest=manifest,packet=packet))
    if len(body)>MAX_FRAME_BYTES:
        raise ValueError('Frame too large')
    return MAGIC+struct.pack('>I',len(body))+body+struct.pack('>I',zlib.crc32(body)&0xffffffff)


def scan(wire):
    if type(wire) is not bytes or len(wire)>MAX_WIRE_BYTES:
        raise ValueError('Wire input exceeds type or byte limit')
    records, rejected = [], Counter()
    cursor = candidates = 0
    complete = True
    while cursor < len(wire):
        start = wire.find(MAGIC,cursor)
        if start < 0:
            break
        candidates += 1
        if candidates>MAX_CANDIDATES:
            complete = False
            break
        cursor = start+1
        if start+8>len(wire):
            rejected['truncated_header'] += 1
            continue
        size = struct.unpack_from('>I',wire,start+4)[0]
        if not 1<=size<=MAX_FRAME_BYTES:
            rejected['invalid_size'] += 1
            continue
        end = start+8+size+4
        if end>len(wire):
            rejected['truncated_body'] += 1
            continue
        body = wire[start+8:end-4]
        expected = struct.unpack_from('>I',wire,end-4)[0]
        if zlib.crc32(body)&0xffffffff != expected:
            rejected['crc_mismatch'] += 1
            continue
        try:
            record = read_json(body)
            if type(record) is not dict or set(record)!={'manifest','packet'}:
                raise ValueError('Unexpected record shape')
            # Validate the manifest without taking any packet content on trust.
            trasporto.decode(record['manifest'],[])
        except (ValueError,TypeError):
            rejected['invalid_record_or_manifest'] += 1
            continue
        records.append(record)
        cursor = end
        if len(records)>trasporto.MAX_PACKETS:
            complete = False
            break
    return dict(records=records,stats=dict(wire_bytes=len(wire),valid_frames=len(records),
                rejected_frames=sum(rejected.values()),rejection_reasons=dict(rejected),
                scan_complete=complete,candidate_headers_examined=candidates))


def recover(wire):
    scanned = scan(wire)
    stats = scanned['stats']
    if not stats['scan_complete']:
        return dict(status='scan_limit',payload_hex=None,framing=stats)
    records = scanned['records']
    if not records:
        return dict(status='need_more',rank=0,required_rank=None,payload_hex=None,
                    reason='No valid frame; the repeated manifest was not received.',framing=stats)
    groups = {}
    for record in records:
        identifier = canonical(record['manifest'])
        groups.setdefault(identifier,[]).append(record['packet'])
    if len(groups)!=1:
        return dict(status='metadata_conflict',payload_hex=None,framing=stats,
                    reason='Different manifests in one transmission; no source selected arbitrarily.')
    identifier,packets = next(iter(groups.items()))
    manifest = read_json(identifier)
    result = trasporto.decode(manifest,packets)
    result.update(framing=stats,manifest=manifest)
    return result
