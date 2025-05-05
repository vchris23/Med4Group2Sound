from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from librosa.feature import rms
from librosa.display import specshow
import pandas as pd
import numpy as np
from OurSound import get_spectrogram
from matplotlib import pyplot as plt

class Track:

    @staticmethod
    def remove_empty_tracks(tracks:list):
        """Removes all tracks, where one of the streams has a root mean square less than 0.01"""
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
                if summed_rms/i < 0.015:
                    print(f"Removing {group}")
                    break

            else:
                cleaned_tracks.extend(track_list)

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