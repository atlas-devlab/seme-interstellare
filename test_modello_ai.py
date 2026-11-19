"""Controlli del formato e dell'inferenza del ricevitore SIF-NN/1."""

import base64
import json
import struct
import unittest

import modello_ai


def known_model():
    # Il solo primo ingresso attiva il primo neurone nascosto.
    hidden = [4] + [0] * (64 * 24 - 1)
    output = [0] * (24 * 10)
    output[3 * 24] = 4
    output[7 * 24] = -8

    def layer(values, scale, bias):
        return dict(int8_b64=base64.b64encode(struct.pack(f'{len(values)}b', *values)).decode('ascii'),
                    scale=scale, bias=bias)

    return dict(format='SIF-NN/1', kind='digit_classifier', architecture=[64, 24, 10],
                activation='relu', input_max=16,
                weights=[layer(hidden, 0.5, [-0.5] + [0] * 23),
                         layer(output, 0.25, [0] * 7 + [1] + [0] * 2)])


def load(document=None):
    if document is None:
        document = known_model()
    return modello_ai.load_model(json.dumps(document).encode('ascii'))


class ModelTests(unittest.TestCase):
    def test_signed_weights_normalization_relu_and_output_order(self):
        model = load()
        for pixel, expected in [(0, (0.0, 1.0, 7)), (8, (0.5, 0.0, 3)), (16, (1.5, -2.0, 3))]:
            with self.subTest(pixel=pixel):
                pixels = [pixel] + [0] * 63
                scores = modello_ai.logits(model, pixels)
                self.assertEqual(len(scores), 10)
                self.assertEqual((scores[3], scores[7], modello_ai.predict(model, pixels)), expected)

    def test_invalid_dimensions_and_explicit_schema(self):
        changes = [('architecture', [64, 25, 10]), ('architecture', [64.0, 24, 10]),
                   ('kind', 'other'), ('activation', 'unknown'), ('input_max', 16.0),
                   ('optional_code', 'ignored'), ('weights', [])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                document = known_model()
                document[key] = value
                with self.assertRaises(modello_ai.ModelError):
                    load(document)

    def test_rejects_wrong_base64_length_and_invalid_bytes(self):
        for value in ('AA==', '!' * 2048, 'é' * 2048, 'A' * 2047 + '='):
            with self.subTest(value=value[:4]):
                document = known_model()
                document['weights'][0]['int8_b64'] = value
                with self.assertRaises(modello_ai.ModelError):
                    load(document)

    def test_rejects_bad_scale_and_bias(self):
        for value in (0, -1, 2, True, float('inf'), '1', float('nan')):
            with self.subTest(scale=value):
                document = known_model()
                document['weights'][0]['scale'] = value
                with self.assertRaises(modello_ai.ModelError):
                    load(document)
        for value in ([], [0] * 9 + [200], [0] * 9 + [False], [0] * 9 + [float('nan')]):
            with self.subTest(bias=value):
                document = known_model()
                document['weights'][1]['bias'] = value
                with self.assertRaises(modello_ai.ModelError):
                    load(document)

    def test_rejects_bad_pixel_count_and_value(self):
        model = load()
        values = ([0] * 63, [0] * 65, [-1] + [0] * 63, [17] + [0] * 63,
                  [True] + [0] * 63, [2.5] + [0] * 63)
        for pixels in values:
            with self.subTest(count=len(pixels), first=pixels[0]):
                with self.assertRaises(modello_ai.ModelError):
                    modello_ai.predict(model, pixels)

    def test_rejects_non_json_and_oversize_payload(self):
        values = (b'\xff', b'{"kind":1,"kind":2}', b'[]',
                  b' ' * (modello_ai.MAX_MODEL_BYTES + 1))
        for value in values:
            with self.subTest(size=len(value)):
                with self.assertRaises(modello_ai.ModelError):
                    modello_ai.load_model(value)
