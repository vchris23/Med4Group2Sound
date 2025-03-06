import time
from itertools import groupby

import librosa

from our_classes import Track
import csv
import os
import xml.etree.ElementTree as ET
import timeit

emotion_lookup = {0: "amazement", 1: "solemnity", 2: "tenderness", 3: "nostalgia", 4:"calmness", 5:"power", 6:"joyful_activation", 7:"tension", 8:"sadness"}

def get_name_from_path(path):
    song_folder, song_title = path.split("\\")[-2:]
    song_name = os.path.join(song_folder, song_title)
    return song_name

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

    labels = []
    for i in range(1, len(data_by_id) if number_to_take is None else min(number_to_take + 1, len(data_by_id))):
        values = list(map(lambda x: [int(x[b]) for b in range(2, 10)], data_by_id[i]))
        summed_values = _vote_on_emotion_label(values)
        labels.append(emotion_lookup[summed_values.index(max(summed_values))])

    return labels

music_folders = ["classical", "electronic", "pop", "rock"]
def _get_tracks_with_sound_and_source(music_folder_path:str, source_types:list, sampling_rate=44100, number_to_take:int = None):
    tracks = []
    for folder in music_folders:
        genre_folder = os.path.join(music_folder_path, folder)
        for i in range(1, 101):
            for source in source_types:
                song_path = os.path.join(genre_folder, f"{str(source)}_" + str(i) + '.mp3')
                print(song_path)
                song = librosa.load(song_path,  sr = sampling_rate)
                tracks.append(Track(sound=song, source=source, original_track=f"{folder}\\{i}", name=get_name_from_path(song_path)))


            if number_to_take is not None and len(tracks) >= number_to_take*len(source_types): return tracks

    return tracks


def get_tracks(music_folder_path:str, label_csv_path:str, sources:list, sampling_rate:int = 44100, amount_to_take:int = None):


    labels = _get_labels(label_csv_path, amount_to_take-1 if amount_to_take is not None else amount_to_take)
    tracks = _get_tracks_with_sound_and_source(music_folder_path, sources, sampling_rate, amount_to_take)
    print(len(labels), len(tracks))
    for i in range(0, len(labels)):
        for j in range(len(sources)):
            print(i*len(sources) + j)
            tracks[i*len(sources) + j].label = labels[i]
    return tracks

def get_names_and_features_from_xml(path):
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
            feature_value = feature[1].text
            feature_value = float(feature_value)
            feature_vector.append(feature_value)

        name_and_vector.append(feature_vector)
        names_and_feature_vectors.append(name_and_vector)

    return names_and_feature_vectors

def assign_features_to_tracks(tracks:list, names_and_feature_vectors:list):
    name_list = [name_and_feature[0] for name_and_feature in names_and_feature_vectors]
    for track in tracks:
        try:
            track.features = names_and_features[name_list.index(track.name)][1]
        except ValueError:
            continue
    return tracks

start_time = time.time()
names_and_features = get_names_and_features_from_xml("feature_values_1.xml")
tracks = get_tracks("datasets/emotify/Separated_and_mixed_versions", "datasets/emotify/emotify_data.csv", sources=["Instrumental", "Mixed", "Vocals"], amount_to_take=300)
assign_features_to_tracks(tracks, names_and_features)
end_time = time.time()
print("Time taken: ", end_time - start_time)

