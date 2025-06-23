import random
from collections import defaultdict
from copy import deepcopy
from itertools import groupby

import librosa.feature
from librosa.feature import rms
from librosa.display import specshow
import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame

from OurSound import get_spectrogram
from matplotlib import pyplot as plt

class Track:
    @staticmethod
    def get_class_balance(tracks:list):
        tracks.sort(key=lambda x: x.original_track)
        grouped = groupby(tracks, key= lambda x: x.original_track)

        label_counter = defaultdict(int)
        current_song = None
        for original_song, group in grouped:
            if original_song == current_song:
                continue
            else:
                label_counter[list(group)[0].label] += 1
                current_song = original_song

        print(label_counter)

    @staticmethod
    def get_mismatched_original_tracks(tracks_A:list, tracks_B:list):
        mismatched_tracks = set()
        song_list_A = [track.original_track for track in tracks_A]
        song_list_B = [track.original_track for track in tracks_B]

        for name in song_list_A:
            if name not in song_list_B:
                mismatched_tracks.add(name)
        for name in song_list_B:
            if name not in song_list_A:
                mismatched_tracks.add(name)
        mismatched_tracks = list(mismatched_tracks)
        mismatched_tracks.sort()
        return mismatched_tracks

    @staticmethod
    def remove_empty_tracks_and_number_removed(tracks:list, threshold:float = 0.01):
        """Removes all tracks, where one of the streams has an average root-mean-square less than 0.01"""
        tracks.sort(key = lambda x: x.original_track)
        tracks_by_original_track = groupby(tracks, key=lambda x: x.original_track)
        cleaned_tracks = []
        summations = []
        for group, group_tracks in tracks_by_original_track:
            track_list = list(group_tracks)
            track_list.sort(key = lambda x: x.source)
            tracks_by_source = groupby(track_list, key=lambda x: x.source)

            for source, grouped_tracks in tracks_by_source:
                summed_rms = 0.0
                i = 0
                for track in grouped_tracks:
                    i += 1
                    summed_rms += np.mean(rms(y=track.sound))
                summations.append(summed_rms/i)
                if summed_rms/i < threshold:
                    print(f"Removing {group}")
                    break

            else:
                cleaned_tracks.extend(track_list)

        tracks.sort(key=lambda x: x.original_track)
        counter = set()
        for group, songs in groupby(tracks, key=lambda x: x.original_track):
            counter.add(group)

        cleaned_tracks.sort(key=lambda x: x.original_track)
        clean_counter = set()
        for group, songs in groupby(cleaned_tracks, key=lambda x: x.original_track):
            clean_counter.add(group)

        print("Original amount of songs: ", len(counter), "\nSongs after those without vocals have been removed:", len(clean_counter))
        print(f"original number of tracks: {len(tracks)}, new number of tracks: {len(cleaned_tracks)}")
        return cleaned_tracks
    @staticmethod
    def tracks_to_features_and_labels(tracks:list):
        """Output a deepcopy of the features and labels of all the tracks"""
        new_tracks = deepcopy(tracks)
        labels = [track.label for track in new_tracks]
        features = [track.features for track in new_tracks]

        return features, labels

    @staticmethod
    def separate_tracks_by_source(tracks:list):
        tracks.sort(key=lambda track: track.source)
        grouped = groupby(tracks, lambda track: track.source)

        separated_lists = []

        for source, group in grouped:
            separated_lists.append(list(group))

        return separated_lists

    @staticmethod
    def separate_tracks_by_label(tracks:list):
        tracks.sort(key=lambda track: track.label)
        grouped = groupby(tracks, lambda track: track.label)

        separated_lists = []

        for source, group in grouped:
            separated_lists.append(list(group))

        return separated_lists

    @staticmethod
    def tracks_features_to_dataframe(tracks:list, feature_names:list):
        data = defaultdict(list)
        for track in tracks:
            for i in range(len(feature_names)):
                data[feature_names[i]].append(track.features[i])
        return pd.DataFrame.from_dict(data)

    @staticmethod
    def graph_energy_in_tracks(tracks:list, title:str, opacity=1.0, show=True, axis=None, color=None):

        energy_markers = []
        for track in tracks:
            energy_markers.append(np.mean(rms(y=track.sound)))

        frame = pd.DataFrame.from_dict({'Energy': energy_markers})
        frame.hist(ax = axis, **{'alpha': opacity, 'color': color})
        print(f"Source: {tracks[0].source}")
        print(f"Mean: {frame.mean(axis='rows')}, Median: {frame.median(axis='rows')},\n STD: {frame.std(axis='rows')}, Variance: {frame.var(axis='rows')},\n Skew: {frame.skew(axis='rows')}")
        if show:
            plt.title(title)
            plt.show()

    @staticmethod
    def find_similarity_between_stems(tracks:list):
        grouped_tracks = Track.separate_tracks_by_source(tracks)

        for i in range(len(grouped_tracks[0])):

            feature_refs = []

            for j in range(len(grouped_tracks)):
                if grouped_tracks[j][i].source == "Mixed": break

                audio = grouped_tracks[j][i].sound
                feature_vector = librosa.feature.chroma_stft(y=audio)
                time_delayed = librosa.feature.stack_memory(feature_vector, n_steps=10, delay=3)
                feature_refs.append(time_delayed)

            similarity = librosa.segment.cross_similarity(feature_refs[0], feature_refs[1])

            for i in range(similarity.shape[0]):
                print(similarity[i, i])





    def __init__(self, sound = None, label = None, source = None, original_track = None, name = None):
        self.sound = sound #Must be filled
        self.label:str = label #Must be filled
        self.source:str = source #Must be filled
        self.train_or_test:str = ""
        self.original_track:str = original_track #Must be filled
        self.name: str = name
        self.features:list = [] #Must be filled

    def __str__(self):
        return f"name: {self.name}, label: {self.label}, source: {self.source}, original: {self.original_track}, length of features: {len(self.features)}"

    def __copy__(self):
        copy_track = Track(self.sound, self.label, self.source, self.original_track, self.name)
        copy_track.features = self.features
        return copy_track