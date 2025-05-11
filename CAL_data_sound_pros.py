import os.path
from copy import copy

import librosa
import numpy as np
from pandas import DataFrame
from itertools import groupby
from collections import defaultdict
from data_import import get_name_from_path, get_original_name_and_source_from_file_name, _get_names_and_features_from_xml, _assign_features_to_tracks
from our_classes import Track

i = 0
def _get_new_label(labels:list):
    is_happy:bool = labels.count('happy') > 0 or labels.count('cheerful') > 0
    is_sad:bool = labels.count('sad') > 0 or labels.count('depressed') > 0
    is_angry:bool = labels.count('angry') > 0 or labels.count('aggressive') > 0
    is_calming:bool = labels.count('calming') > 0
    is_exciting:bool = labels.count('exciting') > 0
    is_romantic:bool = labels.count('romantic') > 0

    label = ""

    if is_happy:
        label = "happy "

    if is_sad:
        label = f"{label}sad "

    if is_angry:
        label = f"{label}angry "

    if is_calming:
        label = f"{label}calming "

    if is_exciting:
        label = f"{label}exciting "

    if is_romantic:
        label = f"{label}romantic "

    if label == "":
            label = 'neutral'

    return label

def _generate_new_label_file(annoated_path:str, remove_limit:int = 4):
    file_content = np.loadtxt(fname = annoated_path, delimiter='\t', dtype=np.object_, encoding='utf-8-sig')

    songs_to_labels = defaultdict(list)

    for row in file_content:
        songs_to_labels[row[0]].append(row[1]) #Gathers each song's labels into a single list

    label_counter = defaultdict(int)
    for song in songs_to_labels.keys():
        new_label = _get_new_label(songs_to_labels[song])
        label_counter[new_label] += 1
        songs_to_labels[song] = new_label

    for song in copy(songs_to_labels).keys():
        if label_counter[songs_to_labels[song]] <= remove_limit or songs_to_labels[song] == 'neutral':
            del songs_to_labels[song]

    annotated_dir = os.path.split(annoated_path)[0]
    new_annotated_path = os.path.join(annotated_dir, "new_annotated.txt")
    df = DataFrame.from_dict({"Title": list(songs_to_labels.keys()), "Label": list(songs_to_labels.values())})
    np.savetxt(new_annotated_path, df, delimiter=';', fmt='%s')

def remake_csv(annotated_path:str, list_of_removed:list):
    content = np.loadtxt(fname = annotated_path, delimiter=';', dtype=np.object_, encoding='utf-8-sig')
    print("List of removed:", list_of_removed)
    for i in range(content.shape[0]):
        row = content[i]
        name = row[0]
        print(name)
        if name in list_of_removed:
            print(f"Removed name: ", name)
            np.delete(content, i, 0)

    np.savetxt("test.csv", content, delimiter=';', fmt='%s')


def _make_track(path:str):
    track = Track()
    print(path)
    track.sound = np.float16(librosa.load(path)[0])
    print(track.sound.shape)
    track.name = get_name_from_path(path)
    track.original_track, track.source = get_original_name_and_source_from_file_name(os.path.split(path)[1])
    return track

def _get_tracks_without_features_or_labels(sound_folder_path:str):
    """Only sound files can be in the sound_folder_path directory"""

    files = os.listdir(sound_folder_path)
    paths = [os.path.join(sound_folder_path, file) for file in files]
    tracks = map(_make_track, paths)

    return list(tracks)

def _assign_labels_to_tracks(tracks:list, annotated_path:str):
    csv_content:np = np.loadtxt(annotated_path, delimiter=';', dtype=np.object_, encoding='utf-8-sig')

    names:list = list(csv_content[:, 0])

    for track in tracks:
        try:
            name_index = names.index(track.original_track)
            track.label = csv_content[name_index, 1]
            print(f"{track.name} was assigned the label: '{track.label}'")
        except ValueError: #Handles the fact that some songs won't be on the list, since they had rare labels
            continue

    return csv_content

def get_cal_tracks(annoated_path:str, sound_folder_path:str, feature_xml_path):
    """Only sound files can be in the sound_folder_path directory"""

    names_and_features = _get_names_and_features_from_xml(feature_xml_path)
    tracks = _get_tracks_without_features_or_labels(sound_folder_path)
    _assign_labels_to_tracks(tracks, annoated_path)
    tracks = _assign_features_to_tracks(tracks, names_and_features)

    for track in tracks:
        print(track)

    return tracks



#_generate_new_label_file("datasets/New dataset/cal_annotations2.txt")
#tracks = get_cal_tracks("datasets/New dataset/new_annotated.txt", "datasets/New dataset/Clips", "datasets/New dataset/feature_values_1.xml")
#cleaned_tracks = Track.remove_empty_tracks_and_number_removed(tracks)
#mismatched = Track.get_mismatched_original_tracks(tracks, cleaned_tracks)
#remake_csv("datasets/New dataset/new_annotated.txt", mismatched)

