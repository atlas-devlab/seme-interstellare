"""Inferenza locale per il classificatore di cifre SIF-NN/1."""

import base64
import binascii
from dataclasses import dataclass
import math
import struct

from macchina import read_json


ARCHITECTURE = (64, 24, 10)
INPUT_MAX = 16
MAX_MODEL_BYTES = 16 * 1024
MAX_WEIGHT_SCALE = 1.0
MAX_ABS_BIAS = 64.0
MODEL_KEYS = {'format', 'kind', 'architecture', 'activation', 'input_max', 'weights'}
LAYER_KEYS = {'scale', 'int8_b64', 'bias'}


class ModelError(ValueError):
    """Dati del modello o dell'immagine fuori dal formato supportato."""


@dataclass(frozen=True)
class Layer:
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]


@dataclass(frozen=True)
class DigitClassifier:
    hidden: Layer
    output: Layer


def _finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ModelError(f'{label}: expected finite number')
    return float(value)


def _load_layer(document, inputs, neurons):
    if type(document) is not dict or document.keys() != LAYER_KEYS:
        raise ModelError('Invalid model layer fields')

    scale = _finite_number(document['scale'], 'scale')
    if not 0 < scale <= MAX_WEIGHT_SCALE:
        raise ModelError('Weight scale out of range')

    weight_count = inputs * neurons
    encoded = document['int8_b64']
    if type(encoded) is not str or len(encoded) != 4 * ((weight_count + 2) // 3):
        raise ModelError('Unexpected length of Base64 weights')
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ModelError('Invalid Base64 weights') from error
    if len(packed) != weight_count or base64.b64encode(packed).decode('ascii') != encoded:
        raise ModelError('Weights must use canonical Base64 with the expected byte count')

    values = struct.unpack(f'{weight_count}b', packed)
    weights = tuple(
        tuple(value * scale for value in values[offset:offset + inputs])
        for offset in range(0, weight_count, inputs)
    )
    bias = document['bias']
    if type(bias) is not list or len(bias) != neurons:
        raise ModelError('Unexpected bias dimension')
    bias = tuple(_finite_number(value, 'bias') for value in bias)
    if any(abs(value) > MAX_ABS_BIAS for value in bias):
        raise ModelError('Bias out of range')
    return Layer(weights, bias)


def load_model(payload: bytes) -> DigitClassifier:
    """Carica i pesi int8 e i bias dal documento ricevuto."""
    if type(payload) is not bytes or len(payload) > MAX_MODEL_BYTES:
        raise ModelError('Model payload exceeds the size or type limit')
    try:
        document = read_json(payload)
    except ValueError as error:
        raise ModelError(f'Invalid model JSON: {error}') from error
    if type(document) is not dict or document.keys() != MODEL_KEYS:
        raise ModelError('Invalid model fields')
    if (document['format'] != 'SIF-NN/1' or document['kind'] != 'digit_classifier'
            or document['activation'] != 'relu' or type(document['input_max']) is not int
            or document['input_max'] != INPUT_MAX):
        raise ModelError('Unsupported model format, activation or input encoding')
    dimensions = document['architecture']
    if (type(dimensions) is not list or any(type(value) is not int for value in dimensions)
            or tuple(dimensions) != ARCHITECTURE):
        raise ModelError('Unsupported neural network dimensions')
    layers = document['weights']
    if type(layers) is not list or len(layers) != 2:
        raise ModelError('Exactly two layers are required')
    input_count, hidden_count, output_count = ARCHITECTURE
    return DigitClassifier(
        hidden=_load_layer(layers[0], input_count, hidden_count),
        output=_load_layer(layers[1], hidden_count, output_count),
    )


def _dense(layer, inputs):
    return tuple(
        bias + sum(weight * value for weight, value in zip(weights, inputs))
        for weights, bias in zip(layer.weights, layer.bias)
    )


def logits(model: DigitClassifier, pixels: list[int] | tuple[int, ...]) -> tuple[float, ...]:
    """Calcola dieci punteggi, in ordine dalla cifra 0 alla cifra 9."""
    if type(pixels) not in (list, tuple) or len(pixels) != ARCHITECTURE[0]:
        raise ModelError('Expected 64 pixel counts in row order')
    if any(type(pixel) is not int or not 0 <= pixel <= INPUT_MAX for pixel in pixels):
        raise ModelError('Pixel values must be integers in 0..16')
    inputs = tuple(pixel / INPUT_MAX for pixel in pixels)
    hidden = tuple(max(0.0, value) for value in _dense(model.hidden, inputs))
    return _dense(model.output, hidden)


def predict(model: DigitClassifier, pixels: list[int] | tuple[int, ...]) -> int:
    scores = logits(model, pixels)
    return max(range(len(scores)), key=scores.__getitem__)
