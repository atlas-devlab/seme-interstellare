"""Regression tests for the local modem and the radio-to-model path."""
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

HAS_RADIO = all(find_spec(package) is not None for package in ('numpy', 'scipy'))
if HAS_RADIO:
    import numpy as np

    import demo_ai
    import confronta_radio
    import radio
    from radio_medie import same_power_control
    from radio_phy import SPS, demodulate, encode_frame


@unittest.skipUnless(HAS_RADIO, 'Optional radio dependencies are missing')
class RadioTests(unittest.TestCase):
    def test_sync_without_sample_alignment_or_known_phase(self):
        frame = demo_ai.make_frames(demo_ai.MODEL.read_bytes(), 1)[1][0]
        samples = encode_frame(frame) * np.complex64(np.exp(1.3j))
        for phase in range(SPS):
            with self.subTest(phase=phase):
                observation = np.concatenate((np.zeros(81 + phase, dtype=np.complex64), samples,
                                              np.zeros(80, dtype=np.complex64)))
                self.assertEqual(demodulate(observation)['frames'], [frame])

    def test_crc_rejects_a_corrupted_body_and_noise_only_has_no_frame(self):
        good = demo_ai.make_frames(demo_ai.MODEL.read_bytes(), 2)[1][0]
        # Corrupt interior body bytes. Flipping only the final radio symbol can
        # be undone incidentally by a timing candidate that overlaps the guard.
        corrupted = good[:120] + bytes(value ^ 255 for value in good[120:124]) + good[124:]
        rng = np.random.default_rng(10)
        noise = (rng.standard_normal(13000) + 1j * rng.standard_normal(13000)).astype(np.complex64)
        self.assertEqual(demodulate(noise)['frames'], [])
        self.assertEqual(demodulate(np.concatenate((noise, encode_frame(corrupted), noise)))['frames'], [])

    def test_independent_radio_samples_reconstruct_model(self):
        examples = demo_ai.load_examples(demo_ai.EXAMPLES)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = radio.run(output, 8, True, 19770815, 140, examples)
            self.assertTrue(report['exact_model_and_predictions_match'])
            self.assertLess(report['modem']['crc_valid_frames'], report['frames_sent'])
            self.assertEqual(report['reconstruction']['rank'], 26)
            self.assertEqual((output/'cifre_ricostruito.json').read_bytes(), demo_ai.MODEL.read_bytes())

    def test_average_power_does_not_identify_symbols(self):
        control = same_power_control(radio.SAMPLE_RATE)
        self.assertGreater(control['different_symbols'], 0)
        self.assertEqual(len(control['first_mean_power']), 6)
        self.assertEqual(control['first_mean_power'], control['second_mean_power'])

    def test_compact_format_needs_no_new_packet_id_to_recover(self):
        examples = demo_ai.load_examples(demo_ai.EXAMPLES)
        report = confronta_radio.compare(examples)
        self.assertTrue(report['same_noise_samples_gaps_start_times_and_packet_ids'])
        self.assertGreater(report['fraction_reduction'], 0.60)
        self.assertTrue(report['simulation']['SIC1']['receiver']['exact_model_and_predictions_match'])
        self.assertTrue(report['same_received_ids_control']['exact_model_and_predictions_match'])


if __name__ == '__main__':
    unittest.main()
