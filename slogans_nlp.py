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


if __name__ == "__main__":
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
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(all_slogans_as_text) - maxlen, step):
        sentences.append(all_slogans_as_text[i: i + maxlen])
        next_chars.append(all_slogans_as_text[i + maxlen])
    print('nb sequences:', len(sentences))

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print(x[:3])
    print(y[:3])

    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(epoch, logs):
        # Function invoked at end of each epoch. Prints generated all_slogans_as_text.
        print()
        print('----- Generating all_slogans_as_text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(all_slogans_as_text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = all_slogans_as_text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    from keras.callbacks import ModelCheckpoint

    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')

    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)

    model.fit(x, y, batch_size=128, epochs=5, callbacks=[print_callback, checkpoint, reduce_lr])

