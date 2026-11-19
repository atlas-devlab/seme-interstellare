"""Train the Seme Interstellare digit classifier using the original UCI split.

NumPy is a training dependency. The resulting model is data and is compatible
with the standard-library inference runtime.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET = 'Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. DOI: 10.24432/C50P49.'
DATASET_URL = 'https://doi.org/10.24432/C50P49'
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/'
ARCHITECTURE = [64, 24, 10]
SEED = 19770815
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.003


def load_split(path, expected_count):
    raw = np.loadtxt(path, delimiter=',', dtype=np.int64)
    if raw.shape != (expected_count, 65):
        raise ValueError(f'{path.name}: expected {expected_count} rows with 65 columns')
    pixels, labels = raw[:, :64], raw[:, 64]
    if np.any((pixels < 0) | (pixels > 16)) or np.any((labels < 0) | (labels > 9)):
        raise ValueError(f'{path.name}: out-of-range pixel or label')
    return pixels, labels


def logits(data, params):
    first, first_bias, second, second_bias = params
    return np.maximum(data @ first + first_bias, 0) @ second + second_bias


def train(pixels, labels):
    rng = np.random.default_rng(SEED)
    inputs, hidden, outputs = ARCHITECTURE
    params = [rng.normal(0, np.sqrt(2 / inputs), (inputs, hidden)), np.zeros(hidden),
              rng.normal(0, np.sqrt(2 / hidden), (hidden, outputs)), np.zeros(outputs)]
    means = [np.zeros_like(param) for param in params]
    squares = [np.zeros_like(param) for param in params]
    data = pixels / 16.0
    step = 0
    for _ in range(EPOCHS):
        order = rng.permutation(len(data))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start+BATCH_SIZE]
            x, y = data[batch], labels[batch]
            first, first_bias, second, second_bias = params
            z = x @ first + first_bias
            activations = np.maximum(z, 0)
            scores = activations @ second + second_bias
            probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            probabilities[np.arange(len(batch)), y] -= 1
            gradient_scores = probabilities / len(batch)
            gradient_hidden = (gradient_scores @ second.T) * (z > 0)
            gradients = [x.T @ gradient_hidden, gradient_hidden.sum(axis=0),
                         activations.T @ gradient_scores, gradient_scores.sum(axis=0)]
            step += 1
            for param, grad, mean, square in zip(params, gradients, means, squares):
                mean *= 0.9
                mean += 0.1 * grad
                square *= 0.999
                square += 0.001 * np.square(grad)
                bias_corrected_mean = mean / (1 - 0.9 ** step)
                bias_corrected_square = square / (1 - 0.999 ** step)
                param -= LEARNING_RATE * bias_corrected_mean / (np.sqrt(bias_corrected_square) + 1e-8)
    return params


def model_object(params):
    packed_layers = []
    for matrix, bias in [(params[0], params[1]), (params[2], params[3])]:
        scale = float(np.abs(matrix).max()) / 127
        quantized = np.clip(np.rint(matrix.T / scale), -127, 127).astype(np.int8)
        packed_layers.append(dict(scale=scale, int8_b64=base64.b64encode(quantized.tobytes()).decode('ascii'),
                                  bias=bias.tolist()))
    return dict(format='SIF-NN/1', kind='digit_classifier', architecture=ARCHITECTURE,
                activation='relu', input_max=16, weights=packed_layers)


def unpack_for_evaluation(model):
    params = []
    for layer, inputs, outputs in zip(model['weights'], ARCHITECTURE, ARCHITECTURE[1:]):
        data = base64.b64decode(layer['int8_b64'], validate=True)
        matrix = np.frombuffer(data, dtype=np.int8).reshape(outputs, inputs).T * layer['scale']
        params.extend((matrix, np.array(layer['bias'])))
    return params


def save_json(path, data):
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True,
                     separators=(',', ':')).encode('utf-8') + b'\n'
    path.write_bytes(raw)
    return raw


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(data_dir, output_dir):
    train_path, test_path = data_dir / 'optdigits.tra', data_dir / 'optdigits.tes'
    train_pixels, train_labels = load_split(train_path, 3823)
    params = train(train_pixels, train_labels)
    model = model_object(params)

    # The test split is never used for training or hyperparameter selection.
    test_pixels, test_labels = load_split(test_path, 1797)
    test_inputs = test_pixels / 16.0
    float_labels = logits(test_inputs, params).argmax(axis=1)
    quantized_labels = logits(test_inputs, unpack_for_evaluation(model)).argmax(axis=1)
    float_correct = int(np.count_nonzero(float_labels == test_labels))
    quantized_correct = int(np.count_nonzero(quantized_labels == test_labels))
    train_correct = int(np.count_nonzero(logits(train_pixels / 16.0, params).argmax(axis=1) == train_labels))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_bytes = save_json(output_dir / 'cifre.json', model)
    examples = [dict(index=int(index), pixels=test_pixels[index].tolist(), label=digit)
                for digit in range(10) for index in np.flatnonzero(test_labels == digit)[:2]]
    save_json(output_dir / 'cifre-esempi.json', dict(
        source=f'{DATASET} CC BY 4.0 ({LICENSE_URL}). First two optdigits.tes rows per label. Pixels and labels preserved.',
        examples=examples,
    ))
    report = dict(dataset=DATASET, dataset_url=DATASET_URL, dataset_license='CC BY 4.0',
                  dataset_license_url=LICENSE_URL, train_source='optdigits.tra', test_source='optdigits.tes',
                  train_sha256=digest(train_path.read_bytes()), test_sha256=digest(test_path.read_bytes()),
                  train_count=len(train_labels), test_count=len(test_labels),
                  train_correct=train_correct, train_accuracy=train_correct / len(train_labels),
                  float_test_correct=float_correct, float_test_accuracy=float_correct / len(test_labels),
                  quantized_test_correct=quantized_correct,
                  quantized_test_accuracy=quantized_correct / len(test_labels),
                  test_predictions_agree=int(np.count_nonzero(float_labels == quantized_labels)),
                  quantized_test_predictions_u8_sha256=digest(quantized_labels.astype(np.uint8).tobytes()),
                  model_bytes=len(model_bytes), model_sha256=digest(model_bytes), architecture=ARCHITECTURE,
                  seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, optimizer='Adam', learning_rate=LEARNING_RATE,
                  training_precision='float64', weight_encoding='symmetric int8 per matrix, row-major output by input',
                  bias_encoding='float64 as JSON numbers', numpy_version=np.__version__)
    save_json(output_dir / 'cifre-addestramento.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('data_dir', type=Path, help='Directory with official optdigits.tra and optdigits.tes')
    cli.add_argument('--output', type=Path, default=ROOT / 'models')
    args = cli.parse_args()
    run(args.data_dir, args.output)
