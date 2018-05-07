import argparse
import random
import os
import textwrap

import numpy as np
import librosa as lr


def add_bck(file_in, noise, snr=14):
    sound, sr_sound = lr.core.load(file_in, sr=16000)
    noise, sr_noise = lr.core.load(noise, sr=16000)
    while noise.shape[0] < sound.shape[0]:  # loop in case noise is shorter than
        noise = np.concatenate((noise, noise), axis=0)
    noise = noise[0:sound.shape[0]]
    rms_noise = np.sqrt(np.mean(np.power(noise, 2)))
    rms_sound = np.sqrt(np.mean(np.power(sound, 2)))

    snr_linear = 10 ** (snr / 20.0)
    snr_linear_factor = rms_sound / rms_noise / snr_linear
    y = sound + noise * snr_linear_factor
    rms_y = np.sqrt(np.mean(np.power(y, 2)))
    y = y * rms_sound / rms_y

    return y, sr_sound


def add_noise(in_path, out_path, noise_path, cnt=None):
    in_files_names = list(filter(lambda file_name: file_name[-4:] == '.wav' or file_name[-5:] == '.flac', os.listdir(in_path)))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    noise_files_names = list(filter(lambda file_name: file_name[-4:] == '.wav' or file_name[-5:] == '.flac', os.listdir(noise_path)))
    if cnt is None:
        cnt = len(in_files_names)
    for file_name in in_files_names[:cnt]:
        abs_file_in_path = os.path.join(in_path, file_name)
        noise_file_name = random.choice(noise_files_names)
        abs_file_noise_path = os.path.join(noise_path, noise_file_name)
        s, sr = add_bck(abs_file_in_path, abs_file_noise_path)
        lr.output.write_wav(
            os.path.join(out_path, file_name),
            s, sr=sr, norm=False)

if __name__ == '__main__':
    # add_noise('./example_audio', 'result', './bg_noise/ASTERISK_MUSIC_gsm/dev')

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Add background noise to audio files.
        '''))

    parser.add_argument('-in_corpus_path', help='Path to the original audio corpus to corrupt ',
                        required=True,
                        dest='in_corpus_path')
    parser.add_argument('-out_corpus_path', help='Output path to the corrupted corpus ',
                        required=True,
                        dest='out_corpus_path')
    parser.add_argument('-noise_dir', help='Directory with noise sounds',
                        required=True,
                        dest='noise_dir')
    args = parser.parse_args()
    add_noise(args.in_corpus_path, args.out_corpus_path, args.noise_dir)
