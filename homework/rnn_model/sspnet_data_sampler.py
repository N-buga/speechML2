import os
from os.path import join

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from .utils import chunks, get_sname
from .feature_extractors import FeatureExtractor


class SSPNetDataSampler:
    """
    Class for loading and sampling audio data by frames for SSPNet Vocalization Corpus
    """
    SAMPLE_RATE = 16000
    DURATION = 11
    AUDIO_LEN = SAMPLE_RATE * DURATION
    LABEL_NAME = "IS_LAUGHTER"
    S_NAME = "SNAME"

    @staticmethod
    def read_labels(labels_path):
        def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
        label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
        def_cols.extend(label_cols)
        labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)
        return labels

    def __init__(self, data_dir, labels_path):
        self.data_dir = data_dir
        self.labels = self.read_labels(labels_path)

    @staticmethod
    def most(l):
        return int(sum(l) > len(l) / 2)

    @classmethod
    def _interval_generator(cls, incidents):
        for itype, start, end in chunks(incidents, 3):
            if itype == 'laughter':
                yield int(start * cls.SAMPLE_RATE), int(end * cls.SAMPLE_RATE)

    def get_labels_for_file(self, wav_path, frame_sec):
        sname = get_sname(wav_path)
        sample = self.labels[self.labels.Sample == sname]

        incidents = sample.loc[:, 'type_voc_0':'end_voc_5']
        incidents = incidents.dropna(axis=1, how='all')
        incidents = incidents.values[0]

        rate, audio = wav.read(wav_path)

        laughts = self._interval_generator(incidents)
        laught_along = np.zeros((self.AUDIO_LEN,))
        for beg, end in laughts:
            laught_along[beg:end] = 1

        frame_size = int(frame_sec * self.SAMPLE_RATE)
        frame_step = frame_size//2

        is_laughter = np.array(
            [self.most(laught_along[i:i + frame_size]) for i in range(0, len(audio) - frame_size, frame_step)])

        df = pd.DataFrame({self.LABEL_NAME: is_laughter,
                           self.S_NAME: sname})
        return df

    @classmethod
    def features_from_file(cls, wav_path, frame_sec):
        frame_size = int(frame_sec * cls.SAMPLE_RATE)
        frame_step = frame_size//2

        mfcc_features, mfcc_columns, filter_features, filter_columns = \
            FeatureExtractor.extract_features(wav_path, frame_size, frame_step)
        df_data = pd.concat([mfcc_features, filter_features], axis=1)
        df_data[cls.S_NAME] = get_sname(wav_path)
        return df_data, mfcc_columns, filter_columns

    def labeled_df_from_file(self, wav_path, frame_sec):
        """
        Returns sampled data by path to audio file
        :param wav_path: string, .wav file path
        :param frame_sec: int, length of each frame in sec
        :return: pandas.DataFrame with sampled audio and target labels, mfcc column names, filter column names
        """
        labels = self.get_labels_for_file(wav_path, frame_sec)

        frame_size = int(frame_sec * self.SAMPLE_RATE)
        frame_step = frame_size//2

        mfcc_features, mfcc_columns, filter_features, filter_columns = \
            FeatureExtractor.extract_features(wav_path, frame_size, frame_step)
        df_data = pd.concat([mfcc_features, filter_features, labels], axis=1)
        return df_data, mfcc_columns, filter_columns

    def get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.AUDIO_LEN]

    def create_train_df(self, frame_sec, naudio=None):
        """
        Returns sampled data for whole corpus
        :param frame_sec: int, length of each frame in sec
        :param naudio: int, number of audios to parse, if not defined parses all
        :return: dataframes, mfcc column names, filter bank column names
        """
        fullpaths = self.get_valid_wav_paths()[:naudio]

        datas = [self.labeled_df_from_file(wav_path, frame_sec) for wav_path in fullpaths]
        dataframes = [data[0] for data in datas]

        # colnames = datas[0][1] + datas[0][2]
        # colnames.append(self.LABEL_NAME)
        # colnames.append(self.S_NAME)
        # for df in dataframes:
        #     df.columns = colnames

        return dataframes, datas[0][1], datas[0][2]