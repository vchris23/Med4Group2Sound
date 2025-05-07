import random
import time
from collections import defaultdict
from itertools import groupby

import librosa
import numpy
from librosa.filters import chroma
from onnx.numpy_helper import from_dict
from sklearn.feature_extraction import DictVectorizer

import OurSound
from OurSound import SpectrogramType
from our_classes import Track
import csv
import os
import xml.etree.ElementTree as ET
import timeit
import soundfile
import numpy as np
import pandas as pd

emotion_lookup = {0: "amazement", 1: "solemnity", 2: "tenderness", 3: "nostalgia", 4:"calmness", 5:"power", 6:"joyful_activation", 7:"tension", 8:"sadness"}

def get_name_from_path(path):
    song_folder_path, song_title = os.path.split(path)
    song_folder = os.path.split(song_folder_path)[1]
    song_name = os.path.join(song_folder, song_title)
    return song_name

def get_original_name_and_source_from_file_name(file_name:str):
    split_name = file_name.split("_")
    source = split_name[0]
    part_with_file_type = split_name[-1]
    original_name = file_name[len(source) + 1:-(len(part_with_file_type) + 1)] #We add one to each to account for the underscore
    print("getting original name and source from: ", file_name)

    return original_name, source
def add_list(list_a, list_b):
    summed_list = list_a.copy()
    for i in range(len(list_a)):
        summed_list[i] += list_b[i]
    return summed_list

def _vote_on_emotion_label(label_lists_for_id):
    total_list = label_lists_for_id[0].copy()
    for i in range(1, len(label_lists_for_id)):
        total_list = add_list(total_list, label_lists_for_id[i])
    return total_list

def _get_labels(label_csv_path:str, number_to_take:int = None):
    reader = csv.reader(open(label_csv_path), delimiter=',')
    data_by_id = []
    for k, g in groupby(reader, lambda x: x[0]):
        data_by_id.append(list(g))

    rand = random.Random()
    rand.seed(10)

    labels_and_names = []
    for i in range(1, len(data_by_id) if number_to_take is None else min(number_to_take + 1, len(data_by_id))):
        values = list(map(lambda x: [int(x[b]) for b in range(2, 10)], data_by_id[i]))
        genre = data_by_id[i][0][1]
        song_id = ((i-1)%100)+1
        summed_values = _vote_on_emotion_label(values)

        max_vote = max(summed_values)
        if len([value for value in summed_values if value == max_vote]) > 1:
            indexes = []
            for i in range(len(summed_values)):
                if summed_values[i] == max_vote:
                    indexes.append(i)

            labels_and_names.append((emotion_lookup[rand.choice(indexes)], f"{genre}, {song_id}"))
        else:
            labels_and_names.append((emotion_lookup[summed_values.index(max(summed_values))], f"{genre}, {song_id}"))

    return labels_and_names

music_folders = ["classical", "rock", "electronic", "pop"]
def _get_tracks_with_sound_and_source(music_folder_path:str, source_types:list, sampling_rate=44100, number_to_take:int = None):
    tracks = []
    for folder in music_folders:
        genre_folder = os.path.join(music_folder_path, folder)
        folder_content = os.listdir(genre_folder)
        for content in folder_content:

            content_path = os.path.join(genre_folder, content)
            song = librosa.load(content_path, sr=sampling_rate)[0]

            source, number, part = content.split('_')
            source = source.split(".")[0]

            original_track = f"{folder}, {number}"
            name = get_name_from_path(content_path)

            new_track = Track(song, None, source, original_track, name)
            tracks.append(new_track)

            print(f"_get_tracks_with, len of tracks: {len(tracks)}, number to take: {number_to_take}")

            if number_to_take is not None and len(tracks) >= number_to_take*len(source_types): return tracks

    return tracks


def _get_tracks(music_folder_path:str, label_csv_path:str, sources:list, sampling_rate:int = 44100, amount_to_take:int = None):


    labels = _get_labels(label_csv_path, amount_to_take-1 if amount_to_take is not None else amount_to_take)
    tracks = _get_tracks_with_sound_and_source(music_folder_path, sources, sampling_rate, number_to_take=amount_to_take)

    dic = defaultdict(list)
    for pair in labels:
        dic[pair[1]].append(pair[0])  #We turn the labels and names into a dictionary,
        # so we efficiently can give the tracks corresponding labels using their original track field

    for track in tracks:
        track.label = dic[track.original_track][0] #The [0] is to prevent a weird one-length list with just the label inside

    return tracks

def _get_names_and_features_from_xml(path):
    tree = ET.parse(path)

    names_and_feature_vectors = []

    feature_sets = tree.findall("data_set")

    for sets in feature_sets:
        name_and_vector = []
        song_path = sets[0].text
        song_name = get_name_from_path(song_path)
        name_and_vector.append(song_name)

        feature_vector = []

        features = sets.findall("feature")

        for feature in features:

            for value in feature[1:]:
                feature_value = value.text
                feature_value = float(feature_value)
                feature_vector.append(feature_value)

        name_and_vector.append(feature_vector)
        names_and_feature_vectors.append(name_and_vector)

    return names_and_feature_vectors
def get_get_chroma_features(track:Track):
    chromagram = OurSound.get_spectrogram(track.sound, sampling_rate=44100, number_of_bands=12, horizontal_resolution=1024, spectrogram_type=SpectrogramType.stft_chromagram)

    features = []
    for band in chromagram:
        features.append(np.mean(band))
    return features

def _assign_features_to_tracks(tracks:list, names_and_feature_vectors:list):
    name_list = [name_and_feature[0] for name_and_feature in names_and_feature_vectors]
    for track in tracks:
        try:
            track.features = names_and_feature_vectors[name_list.index(track.name)][1]
            track.features.extend(get_get_chroma_features(track))

        except ValueError: #In case the song can't be found in our list of tracks
            continue



    return tracks

def get_feature_names(feature_xml_path:str, with_chroma:bool):

    tree = ET.parse(feature_xml_path)
    feature_sets = tree.findall("data_set")

    feature_names = []
    for feature in feature_sets[0].findall("feature"):

        if (len(feature[1:]) > 1):
            for i in range(len(feature[1:])):
                feature_names.append(f"{feature[0].text} {i + 1}") #Lets us keep track of features with the same name, by giving them a number at the end
        else:
            feature_names.append(feature[0].text)

    if (with_chroma):
        for i in range(12):
            feature_names.append(f"Chroma feature {i}")

    return feature_names

def import_tracks(music_folder_path:str, label_csv_path:str, features_xml_path:str, sources:list, amount_to_take:int = None):
    names_and_features = _get_names_and_features_from_xml(features_xml_path)
    tracks = _get_tracks(music_folder_path, label_csv_path,
                         sources, amount_to_take=amount_to_take)
    tracks = Track.remove_empty_tracks(tracks)
    tracks = _assign_features_to_tracks(tracks, names_and_features)

    return tracks

