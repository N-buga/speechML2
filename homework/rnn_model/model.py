import numpy as np
import pandas as pd

from keras.layers import Input, LSTM, Dense, concatenate, TimeDistributed
from keras.models import Model
import keras
from functools import partial

from .sspnet_data_sampler import SSPNetDataSampler


class RNNPredictor:
    NUM_CLASSES = 2

    def __init__(self):
        self.model = None

    def fit(self, dir_data, path_to_labels, frame_sec, cnt_audio, epochs=15, batch_size=16):
        data, mfcc_names, filter_names = SSPNetDataSampler(dir_data, path_to_labels).create_train_df(frame_sec, cnt_audio)
        data_filter = self._extract_filter_bank(data, filter_names)
        data_mfcc = self._extract_mfcc(data, mfcc_names)
        data_y = self._extract_labels(data)
        self._create_model((data_mfcc.shape[1], data_mfcc.shape[2]), (data_filter.shape[1], data_filter.shape[2]))
        self.model = self._create_model((data_mfcc.shape[1], data_mfcc.shape[2]), (data_filter.shape[1], data_filter.shape[2]))
        self.model.fit([data_mfcc, data_filter], [data_y, data_y], epochs=epochs, batch_size=batch_size)

    def fit_and_estimate(self, dir_data, path_to_labels, frame_sec, cnt_audio, validation_split=0.3, epochs=15, batch_size=16):
        data, mfcc_names, filter_names = SSPNetDataSampler(dir_data, path_to_labels).create_train_df(frame_sec, cnt_audio)
        data_filter = self._extract_filter_bank(data, filter_names)
        data_mfcc = self._extract_mfcc(data, mfcc_names)
        data_y = self._extract_labels(data)
        self.model = self._create_model((data_mfcc.shape[1], data_mfcc.shape[2]), (data_filter.shape[1], data_filter.shape[2]))
        self.model.fit([data_mfcc, data_filter], [data_y, data_y], validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    @staticmethod
    def get_labels(path_to_labels, wav_path, frame_sec):
        return SSPNetDataSampler(None, path_to_labels).get_labels_for_file(wav_path, frame_sec)

    def predict_proba(self, wav_path, frame_sec):
        data, mfcc_names, filter_names = SSPNetDataSampler.features_from_file(wav_path, frame_sec)
        data = [pd.DataFrame(data)]
        data_filter = self._extract_filter_bank(data, filter_names)
        data_mfcc = self._extract_mfcc(data, mfcc_names)
        return self.model.predict([data_mfcc, data_filter])[0]

    def predict_classes(self, wav_path, frame_sec):
        data, mfcc_names, filter_names = SSPNetDataSampler.features_from_file(wav_path, frame_sec)
        data = [pd.DataFrame(data)]
        data_filter = self._extract_filter_bank(data, filter_names)
        data_mfcc = self._extract_mfcc(data, mfcc_names)
        return self.model.predict_classes([data_mfcc, data_filter])

    def _extract_filter_bank(self, data, filter_names):
        return np.array([df.loc[:, filter_names].values for df in data])

    def _extract_mfcc(self, data, mfcc_names):
        return np.array([df.loc[:, mfcc_names].values for df in data])

    def _extract_labels(self, data):
        data_y = np.array([df.loc[:, [SSPNetDataSampler.LABEL_NAME]].values for df in data])
        data_y = keras.utils.to_categorical(data_y, self.NUM_CLASSES).reshape(data_y.shape[0], -1, self.NUM_CLASSES)
        return data_y

    def _create_model(self, main_shape, auxiliary_shape):
        main_input = Input(shape=main_shape, dtype='float', name='main_input')
        auxiliary_input = Input(shape=auxiliary_shape, dtype='float', name='aux_input')

        lstm_out = LSTM(100, return_sequences=True)(main_input)
        lstm_aux_out = LSTM(100, return_sequences=True)(auxiliary_input)

        auxiliary_output = TimeDistributed(Dense(2, activation='softmax', name='aux_output'), name='feature_td')(lstm_out)

        x = concatenate([lstm_out, lstm_aux_out])

        main_output = TimeDistributed(Dense(2, activation='softmax', name='aux_output'), name='main_td')(x)

        model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return model
