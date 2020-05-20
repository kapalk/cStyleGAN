"""https://github.com/syang1993/gst-tacotron/blob/master/util/audio.py"""
from scipy import signal
from scipy.optimize import nnls
from scipy.io import wavfile
import librosa
import librosa.filters
import numpy as np


class MelspectrogramBase:

    def __init__(self, num_mels, num_freq, sample_rate, frame_length_ms, frame_shift_ms):
        self.num_mels = num_mels
        self.num_freq = num_freq
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self._mel_basis = None

    def _stft_parameters(self):
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000 * self.sample_rate)
        return n_fft, hop_length, win_length

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann", center=True,
                            pad_mode="constant")

class AudioToMel(MelspectrogramBase):

    def __init__(self, num_mels=80, num_freq=1025, sample_rate=16000, frame_length_ms=50, frame_shift_ms=12.5,
                 preemphasis_param=0.97, min_level_db=-100, ref_level_db=20, power=1.5):
        super(AudioToMel, self).__init__(
            num_mels=num_mels,
            num_freq=num_freq,
            sample_rate=sample_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms,
        )

        self.preemphasis_param = preemphasis_param
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.power = power

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sample_rate)[0]

    @staticmethod
    def save_spectrogram(path, spectogram):
        with open(path, "wb") as fout:
            np.save(fout, spectogram)

    def _preemphasis(self, x):
        return signal.lfilter([1, -self.preemphasis_param], [1], x)

    def spectrogram(self, y):
        D = self._stft(self._preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        D = self._stft(self._preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def _normalize(self, S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)

    def _linear_to_mel(self, spectrogram):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        return np.dot(self._mel_basis, spectrogram)

    def _build_mel_basis(self):
        n_fft = (self.num_freq - 1) * 2
        return librosa.filters.mel(self.sample_rate, n_fft, n_mels=self.num_mels)

    @staticmethod
    def _amp_to_db(x):
        return 20 * np.log10(np.maximum(1e-5, x))


class MelToAudio(MelspectrogramBase):

    def __init__(self, num_mels=80, num_freq=1025, sample_rate=16000, frame_length_ms=50, frame_shift_ms=12.5,
                 preemphasis_param=0.97, min_level_db=-100, ref_level_db=20, power=1.5, griffin_lim_iters=100):
        super(MelToAudio, self).__init__(
            num_mels=num_mels,
            num_freq=num_freq,
            sample_rate=sample_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms
        )

        self.preemphasis_param = preemphasis_param
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        self._inv_mel_basis = None

    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.sample_rate, wav.astype(np.int16))

    def _inv_preemphasis(self, x):
        return signal.lfilter([1], [1, -self.preemphasis_param], x)

    def inv_melspectrogram(self, melspec, use_nnls=False):
        spec_amp = self._db_to_amp(self._denormalize(melspec) + self.ref_level_db)
        if use_nnls:
            spec_lin = self._mel_to_linear_nnls(spec_amp)
        else:
            spec_lin = self._mel_to_linear(spec_amp)
        return self._inv_preemphasis(self._griffin_lim(spec_lin ** self.power))

    def _build_mel_basis(self):
        n_fft, _, _ = self._stft_parameters()
        return librosa.filters.mel(self.sample_rate, n_fft, n_mels=self.num_mels)

    def _mel_to_linear(self, melspec):
        ''' Mel spectrum to linear scale spectrum using pseudoinverse'''
        if self._inv_mel_basis is None:
            mel_basis = self._build_mel_basis()
            self._inv_mel_basis = np.linalg.pinv(mel_basis)
        spec = np.dot(self._inv_mel_basis, melspec)
        return np.maximum(spec, 1e-6)

    def _mel_to_linear_nnls(self, melspec):
        ''' Mel spectrum to linear scale spectrum using NNLS'''
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        spec_list = []
        for m in melspec.T:
            s, _ = nnls(self._mel_basis, m)
            spec_list.append(s)
        spec = np.stack(spec_list).T
        return np.maximum(spec, 1e-6)

    def _griffin_lim(self, spec):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))
        spec_complex = np.abs(spec).astype(np.complex)
        y = self._istft(spec_complex * angles)
        for i in range(self.griffin_lim_iters):
            spec_new = self._stft(y)
            mag = np.maximum(np.abs(spec_new), 1e-6).astype(np.complex)
            angles = spec_new / mag
            y = self._istft(spec_complex * angles)
        return y

    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=hop_length, win_length=win_length, window='hann', center=True)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)
