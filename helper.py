import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import random
from slogans_nlp import read_data_file, get_slogan_lengths, convert_to_plain_text, get_chars, get_char_and_indices_dicts, get_max_len


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(all_slogans_as_text, maxlen, chars, char_indices, indices_char, length, diversity, end_after_pipe_character = True):
    # Get random starting text
    num_pipes = all_slogans_as_text.count('|')
    end_index = random.randint(3, num_pipes)

    def find(str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    end_index = list(find(all_slogans_as_text, '|'))[end_index]

    sentence = all_slogans_as_text[end_index - maxlen + 1:end_index + 1]
    generated = ''
    for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            if next_char == '|' and end_after_pipe_character:
                return generated

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated

def get_saved_model(maxlen, chars, filepath):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


if __name__ == "__main__":
    df = read_data_file('input_data.csv')
    slogan_lengths = get_slogan_lengths(df)
    all_slogans_as_text = convert_to_plain_text(df)
    chars = get_chars(all_slogans_as_text)
    char_indices, indices_char = get_char_and_indices_dicts(chars)
    maxlen = get_max_len(True, slogan_lengths)
    model = get_saved_model(maxlen, chars, "weights.hdf5")

    number_of_slogans = 100
    max_slogan_length = 50
    diversity = 0.3

    for _ in range(number_of_slogans):
        print(generate_text(all_slogans_as_text, maxlen, chars, char_indices, indices_char, max_slogan_length, diversity))
