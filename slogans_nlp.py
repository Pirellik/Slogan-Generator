from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io
import pandas as pd
import string


def read_data_file(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    df = df.drop('Unnamed: 4', 1)
    return df

def drop_unwanted_categories(dataframe, unwanted_categories):
    for cat in unwanted_categories:
        dataframe = dataframe[df["CATEGORY"] != cat]
    return dataframe

def convert_to_plain_text(dataframe):
    return '|'.join(list(dataframe["SLOGAN"]))

def get_chars(plain_text):
    return sorted(list(set(plain_text)))

def get_char_and_indices_dicts(chars):
    print('total chars: ', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char

def get_slogan_lengths(df):
    return [len(elem) for elem in list(df["SLOGAN"])]

def get_max_len(is_average, slogan_lengths, value=40):
    if is_average:
        return int(np.mean(slogan_lengths))
    else:
        return int(value)

def get_x_and_y(plain_text, maxlen, step, chars, char_indices):
    sentences = []
    next_chars = []
    for i in range(0, len(plain_text) - maxlen, step):
        sentences.append(plain_text[i: i + maxlen])
        next_chars.append(plain_text[i + maxlen])
    print('nb sequences:', len(sentences))

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return x, y

def build_model(maxlen, chars):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def train_network(model, x, y):
    from keras.callbacks import ModelCheckpoint
    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    model.fit(x, y, batch_size=128, epochs=5, callbacks=[checkpoint, reduce_lr])


if __name__ == "__main__":
    df = read_data_file('input_data.csv')
    df = drop_unwanted_categories(df, ["Apparel slogans"])

    plain_text = convert_to_plain_text(df)
    chars = get_chars(plain_text)
    char_indices, indices_char = get_char_and_indices_dicts(chars)
    slg_lengths = get_slogan_lengths(df)
    max_len = get_max_len(True, slg_lengths)

    step = 5
    x, y = get_x_and_y(plain_text, max_len, step, chars, char_indices)
    model = build_model(max_len, chars)
    train_network(model, x, y)

