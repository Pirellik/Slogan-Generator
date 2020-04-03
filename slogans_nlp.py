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
    print(df.head())

    whole_text = ''.join(list(df["SLOGAN"]))
    chars = sorted(list(set(whole_text)))
    print('total chars: ', len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    slogan_lengths = [len(elem) for elem in list(df["SLOGAN"])]
    argmin = np.argmin(slogan_lengths)
    argmax = np.argmax(slogan_lengths)
    min_len = slogan_lengths[argmin]
    max_len = slogan_lengths[argmax]

    print(min_len, df["SLOGAN"][argmin])
    print(max_len, df["SLOGAN"][argmax])