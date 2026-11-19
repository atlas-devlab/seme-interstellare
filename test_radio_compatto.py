"""Transport invariants for the compact binary radio format."""
import struct
import unittest
import zlib

import radio_compatto as compact
import trasporto


class CompactFrameTests(unittest.TestCase):
    def setUp(self):
        self.payload = bytes(range(256)) * 12 + bytes(range(194))
        encoded = trasporto.encode(self.payload, block_bytes=128, seed=42)
        self.manifest = encoded['manifest']
        self.packets = encoded['packets']
        self.k = self.manifest['block_count']

    def test_round_trip_and_exact_wire_size(self):
        self.assertEqual(self.k, 26)
        for packet in self.packets:
            with self.subTest(packet=packet['id']):
                wire = compact.frame(self.manifest, packet)
                self.assertEqual(len(wire), 183 if packet['kind']=='source' else 187)
                self.assertEqual(compact.unpack_frame(wire), (self.manifest, packet))

    def test_full_rank_recovers_without_original_equations(self):
        repairs = [compact.frame(self.manifest, p) for p in self.packets[self.k:2*self.k]]
        outcome = compact.recover(repairs)
        self.assertEqual(outcome['status'], 'complete')
        self.assertEqual(outcome['rank'], self.k)
        self.assertEqual(bytes.fromhex(outcome['payload_hex']), self.payload)
        outcome = compact.recover(repairs[:-1])
        self.assertEqual(outcome['status'], 'need_more')
        self.assertEqual(outcome['rank'], self.k-1)
        self.assertIsNone(outcome['payload_hex'])

    def test_corrupt_frame_crc_and_declared_length_are_rejected(self):
        original = compact.frame(self.manifest, self.packets[0])
        changed = original[:24] + bytes([original[24] ^ 1]) + original[25:]
        for wire, reason in [(changed, 'crc_mismatch'), (original[:-1], 'compact_body_length_mismatch')]:
            with self.subTest(reason=reason):
                invalid = compact.recover([wire])
                self.assertEqual(invalid['status'], 'need_more')
                self.assertIsNone(invalid['payload_hex'])
                self.assertEqual(invalid['framing']['rejection_reasons'], {reason: 1})

    def test_recomputed_crc_does_not_bypass_structural_validation(self):
        frame = compact.frame(self.manifest, self.packets[self.k])
        mask_offset = 8 + compact.HEADER.size
        bad = frame[:mask_offset] + b'\x00'*4 + frame[mask_offset+4:-4]
        bad += struct.pack('>I', zlib.crc32(bad[8:]))
        with self.assertRaisesRegex(ValueError, 'invalid_compact_repair_mask'):
            compact.unpack_frame(bad)

    def test_conflicting_manifests_fail_and_oversized_list_is_bounded(self):
        original = compact.frame(self.manifest, self.packets[0])
        other = trasporto.encode(self.payload + b'x')
        result = compact.recover([original, compact.frame(other['manifest'], other['packets'][0])])
        self.assertEqual(result['status'], 'metadata_conflict')
        self.assertIsNone(result['payload_hex'])
        bounded = compact.recover([original]*(trasporto.MAX_PACKETS+1))
        self.assertEqual(bounded['status'], 'scan_limit')
        self.assertIsNone(bounded['payload_hex'])


if __name__=='__main__':
    unittest.main()
