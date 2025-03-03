import librosa
import librosa.feature
import librosa.display
import numpy as np
import soundfile
from matplotlib import pyplot as plt
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

def display_spectrogram(spectrogram, sampling_rate = 22050,
                        time_axis:str = SpectrogramAxis.time_seconds, frequency_axis:str = SpectrogramAxis.frequency_linear,
                        spectral_centroid = None, spectral_bandwidth = None):
    figure, axis = plt.subplots()
    spectogram_decibels = librosa.power_to_db(spectrogram, ref=np.max)
    image = librosa.display.specshow(spectogram_decibels, sr=sampling_rate, ax = axis, x_axis=time_axis, y_axis=frequency_axis)
    figure.colorbar(image, ax = axis, format='%+2.0f dB')
    axis.set(title="spectrogram")


    times = librosa.times_like(spectrogram, sr=sampling_rate)
    if spectral_centroid is not None:
        axis.plot(times, spectral_centroid.T, color='red')

        if spectral_bandwidth is not None:
            axis.fill_between(times, np.maximum(0, spectral_centroid[0] - spectral_bandwidth[0]),
                                 np.minimum(spectral_centroid[0] + spectral_bandwidth[0], sampling_rate/2), alpha = 0.5, color='blue')

    figure.show()

def display_contrast(spectrogram, sampling_rate = 22050,
                        time_axis:str = SpectrogramAxis.time_seconds, frequency_axis:str = SpectrogramAxis.frequency_linear,
                       spectral_contrast = None):

    figure, axis = plt.subplots(nrows=2)
    spectogram_decibels = librosa.power_to_db(spectrogram, ref=np.max)
    image = librosa.display.specshow(spectogram_decibels, sr=sampling_rate, ax = axis[0], x_axis=time_axis, y_axis=frequency_axis)
    figure.colorbar(image, ax=axis[0], format='%+2.0f dB')
    axis[0].set(title="spectrogram")

    figure.show()

def get_spectral_features(sound = None, spectrogram = None, sound_sampling_rate = 22050):

    frame_length = 255
    root_mean_square = librosa.feature.rms(y=sound, S = spectrogram, frame_length= 254)
    spectral_centroid = librosa.feature.spectral_centroid(y=sound, S = spectrogram, sr = sound_sampling_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=sound, S = spectrogram, sr = sound_sampling_rate)
    if sound is not None:
        spectral_contrast = librosa.feature.spectral_contrast(y=sound, sr = sound_sampling_rate)
    else: spectral_contrast = None


    return root_mean_square, spectral_centroid, spectral_bandwidth, spectral_contrast

music, sampling_rate = librosa.load("Sounds/07 Ophelia's Lament.mp3", sr=88000
                                    )
new_specto = get_spectrogram(music, sampling_rate, horizontal_resolution=28600, number_of_bands=None, spectrogram_type=SpectrogramType.mel_scaled_spectrogram)
features = get_spectral_features(sound = music, spectrogram = new_specto, sound_sampling_rate=sampling_rate)
display_spectrogram(new_specto, sampling_rate=sampling_rate, frequency_axis=SpectrogramAxis.frequency_linear, spectral_centroid=features[1], spectral_bandwidth=features[2])

figure, axis = plt.subplots()
img2 = librosa.display.specshow(features[3], x_axis='time', y_axis = 'linear', ax = axis)
figure.colorbar(img2, ax=axis)
plt.show()

