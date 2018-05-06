import os
from os.path import join
from random import shuffle

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from homework.laughter_classification.utils import chunks, in_any, interv_to_range, get_sname
from homework.laughter_prediction.feature_extractors import FeatureExtractor

from homework.laughter_prediction.sample_audio import sample_wav_by_time


class SSPNetDataSampler:
    """
    Class for loading and sampling audio data by frames for SSPNet Vocalization Corpus
    """

    @staticmethod
    def read_labels(labels_path):
        def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
        label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
        def_cols.extend(label_cols)
        labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)
        return labels

    def __init__(self, data_dir, labels_path):
        self.sample_rate = 16000
        self.duration = 11
        self.default_len = self.sample_rate * self.duration
        self.data_dir = data_dir
        self.labels = self.read_labels(labels_path)

    @staticmethod
    def most(l):
        return int(sum(l) > len(l) / 2)

    @staticmethod
    def _interval_generator(incidents):
        for itype, start, end in chunks(incidents, 3):
            if itype == 'laughter':
                yield start, end

    def get_labels_for_file(self, wav_path, frame_sec):
        sname = get_sname(wav_path)
        sample = self.labels[self.labels.Sample == sname]

        incidents = sample.loc[:, 'type_voc_0':'end_voc_5']
        incidents = incidents.dropna(axis=1, how='all')
        incidents = incidents.values[0]

        rate, audio = wav.read(wav_path)

        laughts = self._interval_generator(incidents)
        laughts = [interv_to_range(x, len(audio), self.duration) for x in laughts]
        laught_along = [1 if in_any(t, laughts) else 0 for t, _ in enumerate(audio)]

        frame_size = int(self.sample_rate * frame_sec)
        is_laughter = np.array([self.most(la) for la in chunks(laught_along, frame_size)])

        df = pd.DataFrame({'IS_LAUGHTER': is_laughter,
                           'SNAME': sname})
        return df

    def df_from_file(self, wav_path, frame_sec):
        """
        Returns sampled data by path to audio file
        :param wav_path: string, .wav file path
        :param frame_sec: int, length of each frame in sec
        :return: pandas.DataFrame with sampled audio
        """
        data = sample_wav_by_time(wav_path, frame_sec)
        labels = self.get_labels_for_file(wav_path, frame_sec)
        mfcc_features, mfcc_columns, filter_features, filter_columns = FeatureExtractor().extract_features(wav_path, frame_sec)
        df_data = pd.concat([data, filter_features, mfcc_features, labels], axis=1)
        return df_data, data.shape[1], filter_columns, mfcc_columns

    def get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.default_len]

    def create_sampled_df(self, frame_sec, naudio=None, save_data_path=None, save_mfcc_path=None, force_save=False):
        """
        Returns sampled data for whole corpus
        :param frame_sec: int, length of each frame in sec
        :param naudio: int, number of audios to parse, if not defined parses all
        :param save_path: string, path to save parsed corpus
        :param force_save: boolean, if you want to override file with same name
        :return:
        """
        fullpaths = self.get_valid_wav_paths()[:naudio]
        datas = [self.df_from_file(wav_path, frame_sec) for wav_path in fullpaths]
        dataframes = [data[0] for data in datas]
        data_df = pd.concat(dataframes)

        frames_name = ["V{}".format(i) for i in range(datas[0][1])]
        colnames = frames_name + datas[0][2] + datas[0][3]
        colnames.append("IS_LAUGHTER")
        colnames.append("SNAME")
        data_df.columns = colnames

        df = data_df.sample(frac=1)

        if save_data_path is not None:
            if not os.path.isfile(save_data_path) or force_save:
                print("saving data df: ", save_data_path)
                df.loc[:, frames_name + datas[0][2] + ["IS_LAUGHTER"] + ["SNAME"]].to_csv(save_data_path, index=False)

        if save_mfcc_path is not None:
            if not os.path.isfile(save_mfcc_path) or force_save:
                print("saving mfcc df: ", save_mfcc_path)
                df.loc[:, datas[0][3] + ["IS_LAUGHTER"] + ["SNAME"]].to_csv(save_mfcc_path, index=False)

        return df, frames_name + datas[0][2], datas[0][3], "IS_LAUGHTER", "SNAME"


SSPNetDataSampler('vocalizationcorpus/train', 'vocalizationcorpus/train_labels.csv').create_sampled_df(0.01, 6,
                    save_data_path='features_data.csv', save_mfcc_path="mfcc.csv", force_save=True)
