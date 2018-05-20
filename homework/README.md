## Laughter detection tool

This tool detects laughter interval in audio files.

### Repo structure
rnn_model - realization of rnn model for laugh prediction.

Usage:

`from rnn_model import RNNPredictor`

- `fit(self, dir_data, path_to_labels, frame_sec, cnt_audio, epochs=15, batch_size=16)`
- `fit_and_estimate(self, dir_data, path_to_labels, frame_sec, cnt_audio, validation_split=0.3, epochs=15, batch_size=16)`
- `@staticmethod get_labels(path_to_labels, wav_path, frame_sec)`
- `predict_proba(self, wav_path, frame_sec)`
- `predict_classes(self, wav_path, frame_sec)`

Example of usage can be found in `example.ipynb`.

### Data
Audio corpus available at 
http://www.dcs.gla.ac.uk/vincia/?p=378 (vocalizationcorpus.zip)

