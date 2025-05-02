from copy import deepcopy
from itertools import groupby


class Track:
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


    def __init__(self, sound = None, label = None, source = None, original_track = None, name = None):
        self.sound = sound #full path to 15 sec clips
        self.label:str = label #Q1-Q4
        self.source:str = source #Instrumental, mixed or voice
        self.train_or_test:str = ""
        self.original_track:str = original_track #name of og track before source separation and clip
        self.name: str = name #name of directory and song name
        self.features:list = [] #

    def __str__(self):
        return f"name: {self.name}, label: {self.label}, source: {self.source}, original: {self.original_track}"

    def __copy__(self):
        copy_track = Track(self.sound, self.label, self.source, self.original_track, self.name)
        copy_track.features = self.features
        return copy_track