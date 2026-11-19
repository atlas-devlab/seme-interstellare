"""Meaningful transport controls, including corruption and missing rank."""
import copy
import random
import unittest

from trasporto import decode, encode, packet_crc


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.payload = bytes(range(256))*3+b'programma dimostrativo'
        self.capsule = encode(self.payload, block_bytes=64)
        self.manifest = self.capsule['manifest']
        self.k = self.manifest['block_count']

    def test_only_first_k_repair_packets_recover_without_sources(self):
        repair = self.capsule['packets'][self.k:2*self.k]
        self.assertTrue(all(p['kind']=='repair' for p in repair))
        result = decode(self.manifest, repair)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(bytes.fromhex(result['payload_hex']), self.payload)

    def test_random_loss_and_reordering(self):
        rng = random.Random(7)
        for seed in range(20):
            capsule = encode(self.payload,64,3,seed)
            packets = rng.sample(capsule['packets'], 3*self.k)
            rng.shuffle(packets)
            result = decode(capsule['manifest'], packets)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(bytes.fromhex(result['payload_hex']), self.payload)

    def test_missing_equation_returns_no_payload(self):
        result = decode(self.manifest,self.capsule['packets'][self.k:2*self.k-1])
        self.assertEqual(result['status'],'need_more')
        self.assertEqual(result['rank'],self.k-1)
        self.assertIsNone(result['payload_hex'])

    def test_crc_corruption_is_erasure(self):
        packets = copy.deepcopy(self.capsule['packets'][self.k:2*self.k])
        packets[0]['data_hex'] = ('00' if packets[0]['data_hex'][:2]!='00' else '01')+packets[0]['data_hex'][2:]
        result = decode(self.manifest,packets)
        self.assertEqual(result['status'],'need_more')
        self.assertEqual(result['rejected_packets'],1)
        self.assertIn('CRC mismatch',result['rejected_reasons'])

    def test_changed_header_is_crc_bound(self):
        packet = copy.deepcopy(self.capsule['packets'][self.k])
        packet['id'] += self.k
        result = decode(self.manifest,[packet])
        self.assertEqual(result['rejected_packets'],1)
        self.assertIn('CRC mismatch',result['rejected_reasons'])

    def test_reforged_crc_does_not_bypass_payload_sha(self):
        packets = copy.deepcopy(self.capsule['packets'][:self.k])
        packets[0]['data_hex'] = ('00' if packets[0]['data_hex'][:2]!='00' else '01')+packets[0]['data_hex'][2:]
        packets[0]['crc32'] = packet_crc(packets[0])
        result = decode(self.manifest,packets)
        self.assertEqual(result['status'],'integrity_failed')
        self.assertIsNone(result['payload_hex'])

    def test_contradictory_equation_is_not_ignored(self):
        packets = copy.deepcopy(self.capsule['packets'][:self.k])
        extra = dict(packets[0],id=self.k,kind='repair')
        extra['data_hex'] = ('00' if extra['data_hex'][:2]!='00' else '01')+extra['data_hex'][2:]
        extra['crc32'] = packet_crc(extra)
        result = decode(self.manifest,packets+[extra])
        self.assertEqual(result['status'],'integrity_failed')

    def test_duplicates_do_not_increase_rank(self):
        first = self.capsule['packets'][0]
        result = decode(self.manifest,[first]*7)
        self.assertEqual(result['rank'],1)
        self.assertEqual(result['duplicate_packets'],6)
        self.assertIsNone(result['payload_hex'])

    def test_manifest_and_packet_limits(self):
        mutations = [('version',2),('block_bytes',0),('block_count',257),
                     ('payload_length',262145),('payload_sha256','a'*63)]
        for key,value in mutations:
            manifest = dict(self.manifest)
            manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(manifest,[])
        with self.assertRaises(ValueError):
            decode(self.manifest,[self.capsule['packets'][0]]*1025)
        for kwargs in [dict(block_bytes=0),dict(block_bytes=True),dict(repair_multiplier=4)]:
            with self.assertRaises(ValueError):
                encode(self.payload,**kwargs)

    def test_empty_and_exact_block_payloads(self):
        for payload in [b'',b'x',b'x'*128,b'x'*129]:
            capsule = encode(payload)
            result = decode(capsule['manifest'],capsule['packets'])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(bytes.fromhex(result['payload_hex']),payload)

    def test_invalid_packet_is_rejected_without_exposing_data(self):
        for field,value in [('mask_hex','0'),('mask_hex','f'*1000),('data_hex','0'),
                            ('id',True),('kind','shell'),('crc32','xyz')]:
            packet = dict(self.capsule['packets'][0])
            packet[field]=value
            result = decode(self.manifest,[packet])
            self.assertEqual(result['rejected_packets'],1)
            self.assertIsNone(result['payload_hex'])


if __name__=='__main__':
    unittest.main(verbosity=2)
