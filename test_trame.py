"""Channel, repeated-metadata and complete receiver tests."""
import copy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import demo
import macchina
import trame
import trasporto


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = macchina.make_seed()
        cls.transmission = trasporto.encode(cls.payload)
        cls.manifest = cls.transmission['manifest']
        cls.packets = cls.transmission['packets']
        cls.frames = [trame.frame(cls.manifest,p) for p in cls.packets]

    def test_all_metadata_arrives_inside_received_frames(self):
        k = self.manifest['block_count']
        wire = b''.join(trame.frame(self.manifest,p) for p in self.packets[k:2*k][::-1])
        result = trame.recover(wire)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(bytes.fromhex(result['payload_hex']),self.payload)
        self.assertEqual(result['manifest'],self.manifest)

    def test_insufficient_rank_returns_no_payload(self):
        k = self.manifest['block_count']
        result = trame.recover(b''.join(self.frames[:k-1]))
        self.assertEqual(result['status'],'need_more')
        self.assertIsNone(result['payload_hex'])

    def test_empty_and_truncated_header_no_guess(self):
        for wire in [b'',b'noise',b'SIF1',b'SIF1\x00\x00']:
            with self.subTest(wire=wire):
                self.assertEqual(trame.recover(wire)['status'],'need_more')

    def test_corrupt_crc_skipped_and_later_frames_recovered(self):
        changed = bytearray(self.frames[0])
        changed[10] ^= 1
        result = trame.recover(bytes(changed)+b''.join(self.frames[1:]))
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['framing']['rejection_reasons']['crc_mismatch'],1)

    def test_corrupt_length_resynchronizes(self):
        changed = bytearray(self.frames[0])
        changed[4:8] = struct.pack('>I',0xffffffff)
        result = trame.recover(bytes(changed)+b''.join(self.frames[1:]))
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['framing']['rejection_reasons']['invalid_size'],1)

    def test_missing_bytes_inside_one_frame_resynchronizes(self):
        damaged = self.frames[0][:32]+self.frames[0][57:]
        result = trame.recover(damaged+b''.join(self.frames[1:]))
        self.assertEqual(result['status'],'complete')
        self.assertGreaterEqual(result['framing']['rejected_frames'],1)

    def test_conflicting_metadata_rejected(self):
        different = copy.deepcopy(self.manifest)
        different['payload_sha256'] = '0'*64
        wire = b''.join(self.frames)+trame.frame(different,self.packets[0])
        result = trame.recover(wire)
        self.assertEqual(result['status'],'metadata_conflict')
        self.assertIsNone(result['payload_hex'])

    def test_invalid_manifest_does_not_block_later_records(self):
        different = copy.deepcopy(self.manifest)
        different['block_count'] = 0
        result = trame.recover(trame.frame(different,self.packets[0])+b''.join(self.frames))
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['framing']['rejection_reasons']['invalid_record_or_manifest'],1)

    def test_outer_valid_inner_bad_crc_is_rejected(self):
        changed = copy.deepcopy(self.packets[0])
        changed['crc32'] = '00000000'
        result = trame.recover(trame.frame(self.manifest,changed)+b''.join(self.frames[1:]))
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['rejected_packets'],1)

    def test_bounded_input(self):
        with self.assertRaises(ValueError):
            trame.recover(b'x'*(trame.MAX_WIRE_BYTES+1))
        with self.assertRaises(ValueError):
            trame.frame(self.manifest,{'large':'x'*trame.MAX_FRAME_BYTES})

    def test_frame_count_limit_prevents_execution(self):
        result = trame.recover(self.frames[0]*(trasporto.MAX_PACKETS+1))
        self.assertEqual(result['status'],'scan_limit')
        self.assertIsNone(result['payload_hex'])

    def test_end_to_end_changed_initial_seeds(self):
        with tempfile.TemporaryDirectory(prefix='sif-test-') as directory:
            report = demo.demonstration(Path(directory),first=2,second=3)
            self.assertEqual(report['reconstruction']['status'],'complete')
            self.assertEqual(report['execution']['output'],[2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584])
            self.assertTrue(report['source_bytes_match'])

    def test_failed_channel_never_invokes_vm_and_removes_stale_result(self):
        with tempfile.TemporaryDirectory(prefix='sif-test-') as directory:
            output = Path(directory)
            demo.demonstration(output,loss=0,corruption=0)
            self.assertTrue((output/'seme_ricostruito.json').exists())
            with patch('demo.macchina.receive_capsule') as interpreter:
                report = demo.demonstration(output,loss=1,corruption=0)
                interpreter.assert_not_called()
            self.assertEqual(report['reconstruction']['status'],'need_more')
            self.assertFalse((output/'seme_ricostruito.json').exists())

    def test_channel_corruption_of_every_frame_never_invokes_vm(self):
        with tempfile.TemporaryDirectory(prefix='sif-test-') as directory:
            with patch('demo.macchina.receive_capsule') as interpreter:
                report = demo.demonstration(Path(directory),loss=0,corruption=1)
                interpreter.assert_not_called()
            self.assertEqual(report['reconstruction']['status'],'need_more')

    def test_valid_transport_does_not_make_unknown_primitive_executable(self):
        capsule = macchina.read_json(self.payload)
        capsule['machine']['opcode_table'][0]['operation'] = 'RUN_SHELL'
        payload = trame.canonical(capsule)
        encoded = trasporto.encode(payload)
        result = trame.recover(b''.join(trame.frame(encoded['manifest'],p) for p in encoded['packets']))
        self.assertEqual(result['status'],'complete')
        with self.assertRaises(macchina.VMError):
            macchina.receive_capsule(bytes.fromhex(result['payload_hex']))


if __name__=='__main__':
    unittest.main()
