import scipy.io.wavfile as wav
import librosa
import pandas as pd
import numpy as np


class FeatureExtractor:
    @staticmethod
    def extract_features(wav_path, frame_size, frame_step):
        """
        Extracts features for classification ny frames for .wav file

        :param frame_step: step of sliding window
        :param frame_size: window size
        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """

        rate, audio = wav.read(wav_path)

        # Let's make and display a mel-scaled power (energy-squared) spectrogram

        filterbanks = []
        mfccs = []

        for i in range(0, len(audio) - frame_size, frame_step):
            end = i + frame_step
            # Convert to log scale (dB). We'll use the peak power (max) as reference.
            cur_filterbank = librosa.feature.melspectrogram(y=audio.astype(np.float)[i: end], sr=rate)
            filterbanks.append(np.mean(cur_filterbank, axis=1))
            # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
            cur_mfcc = librosa.feature.mfcc(y=audio.astype(np.float)[i: end], sr=rate)
            mfccs.append(np.mean(cur_mfcc, axis=1))

        filterbank = np.vstack(filterbanks)
        mfcc = np.vstack(mfccs)

        columns_mfcc = list(map(lambda num: 'mfcc_' + str(num), list(range(mfcc.shape[1]))))
        columns_filter = list(map(lambda num: 'filterbank_' + str(num), list(range(filterbank.shape[1]))))
        return pd.DataFrame(mfcc, columns=columns_mfcc), columns_mfcc, \
               pd.DataFrame(filterbank, columns=columns_filter), columns_filter