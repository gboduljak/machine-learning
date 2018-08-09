from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS

import tensorflow as tf
import json

import numpy

from keras.models import load_model
from keras.utils import plot_model

from pretrained_models import *
from predict import get_models_predictions, compute_words_importance, set_models_and_graph
from preprocess import preprocess_text

server = Flask(__name__)
CORS(server)

config = {
    'use_only_cpu_compatible_models': True
}
models_paths = get_serialized_models(config['use_only_cpu_compatible_models'])

global models
global graph
models = [*map(load_model, models_paths)]
graph = tf.get_default_graph()

set_models_and_graph(models, graph)


@server.route('/predict', methods=['POST'])
def predict():
    text = json.loads(request.data)['text']

    predictions = get_models_predictions(text)
    importances = compute_words_importance(text, predictions['most_probable_category'])

    dto = predictions
    dto['word_importances'] = importances

    return jsonify(dto)
