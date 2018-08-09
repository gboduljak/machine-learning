import re
import numpy as np
import pickle

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing import sequence

from keras.preprocessing import text, sequence
import itertools

max_length_in_words = 400

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def preprocess_text(text):
    cleaned_text = clean(text)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    padded_text = sequence.pad_sequences(tokenized_text, maxlen=max_length_in_words)
    return padded_text


def clean(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ')

    return text
