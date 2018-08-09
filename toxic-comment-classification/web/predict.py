import tensorflow as tf
import json

import numpy

from keras.models import load_model
from keras.utils import plot_model

from pretrained_models import *
from preprocess import preprocess_text, clean

global graph
global models


def set_models_and_graph(new_models, new_graph):
    global models
    global graph

    models = new_models
    graph = new_graph
    pass


def compute_category_difference(previous_result, new_result):
    return -(new_result['probability'] - previous_result['probability'])


def compute_words_importance(text, averaged_most_probable_category):
    cleaned_text = clean(text)
    words = list(set(cleaned_text.split(' ')))

    words_with_texts = [
        *map(
            lambda word: (word, ' '.join([*filter(lambda text_word: text_word != word, words)])),
            words
        )
    ]
    results_without_each_word = [
        *map(
            lambda group: (group[0], get_models_predictions(group[1])['most_probable_category']),
            words_with_texts
        )
    ]

    return [
        *map(
            lambda group: {
                'word': group[0],
                'importance': compute_category_difference(averaged_most_probable_category, group[1])
            },
            results_without_each_word
        )
    ]


def get_models_predictions(text):
    preprocessed_text = preprocess_text(text)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    with graph.as_default():
        raw_probabilities = [
            *map(
                lambda model: numpy.squeeze(model.predict(preprocessed_text), axis=0).tolist(),
                models
            )
        ]
        probabilities_with_labels = [
            *map(
                lambda probability: [
                    *map(
                        lambda i: {'label': labels[i], 'probability': probability[i]},
                        range(0, 6)
                    )],
                raw_probabilities
            )
        ]

    averaged_probabilities = numpy.average(raw_probabilities, axis=0).tolist()

    return {
        'probabilities_of_models': raw_probabilities,
        'probabilities_of_models_with_labels': probabilities_with_labels,
        'models_averaged_probabilities': averaged_probabilities,
        'most_probable_category': {
            'label': labels[numpy.argmax(averaged_probabilities)],
            'probability': numpy.max(averaged_probabilities)
        }
    }
