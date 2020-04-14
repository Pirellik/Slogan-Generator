import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import random

df = pd.read_csv('input_data.csv', delimiter=';')
df = df.drop('Unnamed: 4', 1)
slogan_lengths = [len(elem) for elem in list(df["SLOGAN"])]
argmin = np.argmin(slogan_lengths)
argmax = np.argmax(slogan_lengths)
min_len = slogan_lengths[argmin]
max_len = slogan_lengths[argmax]
# df = df[df.CATEGORY == 'Apparel slogans']
# print(list(df['SLOGAN']))
all_slogans_as_text = '|'.join(list(df["SLOGAN"]))
chars = sorted(list(set(all_slogans_as_text)))
print('total chars: ', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = int(np.mean(slogan_lengths))
print('maxlen = ', maxlen)


model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)

model.load_weights("weights.hdf5")

model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, diversity, end_after_pipe_character = True):
    # Get random starting text
    num_pipes = all_slogans_as_text.count('|')
    end_index = random.randint(3, num_pipes)

    def find(str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    end_index = list(find(all_slogans_as_text, '|'))[end_index]

    sentence = all_slogans_as_text[end_index - maxlen + 1:end_index + 1]
    # print(sentence)
    # print('PASKO LEN', len(sentence))
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

for _ in range(100):
    print(generate_text(85, 0.2))