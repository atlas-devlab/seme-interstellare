"""Exercise model transport, integrity gating and receiver-only inference."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import demo_ai


class ModelTransportTests(unittest.TestCase):
    def test_repair_only_restores_trained_model_and_test_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = demo_ai.demo(directory, repairs_only=True)
            self.assertTrue(report['exact_model_and_predictions_match'])
            self.assertEqual(report['frames_lost'], 2 * report['source_blocks'])
            self.assertEqual(report['reconstruction']['rank'], report['source_blocks'])
            metrics = report['inference']
            self.assertGreaterEqual(metrics['accuracy'], 0.80)
            with patch('demo_ai.MODEL', directory / 'absent.json'):
                status, restored, inference = demo_ai.receive(
                    (directory / 'ricevuto.bin').read_bytes(),
                    demo_ai.load_examples(demo_ai.EXAMPLES),
                )
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(inference, metrics)
            self.assertEqual(restored, demo_ai.MODEL.read_bytes())

    def test_insufficient_rank_never_loads_weights(self):
        examples = demo_ai.load_examples(demo_ai.EXAMPLES)
        manifest, frames = demo_ai.make_frames(demo_ai.MODEL.read_bytes(), seed=8)
        with patch('demo_ai.load_model') as loader:
            report, restored, inference = demo_ai.receive(
                b''.join(frames[:manifest['block_count']-1]), examples)
        loader.assert_not_called()
        self.assertEqual(report['status'], 'need_more')
        self.assertEqual(report['rank'], manifest['block_count']-1)
        self.assertIsNone(restored)
        self.assertIsNone(inference)

    def test_corrupted_frame_is_rejected_before_recovery(self):
        examples = demo_ai.load_examples(demo_ai.EXAMPLES)
        manifest, frames = demo_ai.make_frames(demo_ai.MODEL.read_bytes(), seed=9)
        corrupted = bytearray(frames[0])
        corrupted[12] ^= 1
        report, restored, inference = demo_ai.receive(
            bytes(corrupted)+b''.join(frames[1:]), examples)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['framing']['rejection_reasons']['crc_mismatch'], 1)
        self.assertEqual(restored, demo_ai.MODEL.read_bytes())
        self.assertEqual(inference['examples'], len(examples))

    def test_full_loss_removes_stale_model_and_never_runs_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            demo_ai.demo(directory, repairs_only=True)
            with patch('demo_ai.evaluate') as evaluate:
                report = demo_ai.demo(directory, loss=1)
            evaluate.assert_not_called()
            self.assertEqual(report['reconstruction']['status'], 'need_more')
            self.assertIsNone(report['inference'])
            self.assertFalse((directory / 'cifre_ricostruito.json').exists())


if __name__ == '__main__':
    unittest.main()
