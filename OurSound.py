import librosa
import librosa.feature
import numpy as np
import soundfile
from enum import Enum
import soundfile as sf

class SpectrogramAxis:
    frequency_LOG:str = 'log'
    frequency_linear:str = 'linear'
    frequency_chroma = 'chroma'
    time_minutes_seconds = 'm'
    time_seconds = 's'
    time_milliseconds = 'ms'

class SpectrogramType(Enum):
    mel_scaled_spectrogram = 1
    stft_chromagram = 2

def get_spectrogram(sound, sampling_rate, number_of_bands = None, horizontal_resolution = 2048, spectrogram_type:SpectrogramType = SpectrogramType.mel_scaled_spectrogram):
    spectrogram = None
    match spectrogram_type:
        case SpectrogramType.mel_scaled_spectrogram:
            spectrogram = librosa.feature.melspectrogram(y = sound, sr = sampling_rate, n_fft=horizontal_resolution, n_mels = 128 if number_of_bands is None else number_of_bands)
        case SpectrogramType.stft_chromagram:
            spectrogram = librosa.feature.chroma_stft(y = sound, sr = sampling_rate, n_fft=horizontal_resolution, n_chroma = 16 if number_of_bands is None else number_of_bands)
        case _:
            raise RuntimeError("Unexpected spectrogram type")
    return spectrogram

def get_spectral_features(sound = None, spectrogram = None, sound_sampling_rate = 22050):

    frame_length = 255
    root_mean_square = librosa.feature.rms(y=sound, S = spectrogram, frame_length= 254)
    spectral_centroid = librosa.feature.spectral_centroid(y=sound, S = spectrogram, sr = sound_sampling_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=sound, S = spectrogram, sr = sound_sampling_rate)
    if sound is not None:
        spectral_contrast = librosa.feature.spectral_contrast(y=sound, sr = sound_sampling_rate)
    else: spectral_contrast = None


    return root_mean_square, spectral_centroid, spectral_bandwidth, spectral_contrast

