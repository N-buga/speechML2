import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split


def rnn(X_data, mfcc_data, y):
    train_index, test_index = train_test_split(list(range(X_data.shape[0])))

    main_input = Input(shape=(X_data.shape[1], 1), dtype='float', name='main_input')
    lstm_out = LSTM(32)(main_input)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=(mfcc_data.shape[1],), name='aux_input')
    x = concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  loss_weights=[1., 0.2])
    model.fit([X_data.iloc[train_index], mfcc_data.iloc[train_index]], [y.iloc[train_index], y.iloc[train_index]],
          epochs=50, batch_size=32)

def main():
    DATA_FEATURE_PATH = "features_data.csv"
    MFCC_PATH = "mfcc.csv"
    nrows = 300
    data_df = pd.read_csv(DATA_FEATURE_PATH, nrows=nrows)
    mfcc_df = pd.read_csv(MFCC_PATH, nrows=nrows)
    data_nfeatures = data_df.shape[1] - 2
    mfcc_nfeatures = mfcc_df.shape[1] - 2
    only_data_features = data_df.iloc[:, :data_nfeatures]
    only_mfcc_features = mfcc_df.iloc[:, :mfcc_nfeatures]
    # data_X = only_data_features.as_matrix()
    # data_mfcc = only_mfcc_features.as_matrix()
    y = data_df.IS_LAUGHTER
    rnn(only_data_features, only_mfcc_features, y)

if __name__ == '__main__':
    main()
