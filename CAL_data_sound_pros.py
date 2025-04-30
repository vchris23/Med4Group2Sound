import os.path

import librosa
import numpy as np
from pandas import DataFrame
from itertools import groupby
from collections import defaultdict
from data_import import get_name_from_path, get_original_name_and_source_from_file_name, _get_names_and_features_from_xml, _assign_features_to_tracks
from our_classes import Track


def _get_new_label(labels:list):
    is_happy:bool = labels.count('happy') > 0
    is_sad:bool = labels.count('sad') > 0
    is_angry:bool = labels.count('angry') > 0
    is_calming:bool = labels.count('calming') > 0

    label = ""

    if is_happy:
        label = "happy "

    if is_sad:
        label = f"{label}sad "

    if is_angry:
        label = f"{label}angry "

    if is_calming:
        label = f"{label}calming"

    if label == "":
        label = 'neutral'

    return label

def _generate_new_label_file(annoated_path:str):
    file_content = np.loadtxt(fname = annoated_path, delimiter=';', dtype=np.object_, encoding='utf-8-sig')

    songs_to_labels = defaultdict(list)

    for row in file_content:
        songs_to_labels[row[0]].append(row[1]) #Gathers each song's labels into a single list

    for song in songs_to_labels.keys():
        songs_to_labels[song] = _get_new_label(songs_to_labels[song])

    annotated_dir = os.path.split(annoated_path)[0]
    new_annotated_path = os.path.join(annotated_dir, "new_annotated.csv")
    df = DataFrame.from_dict({"Title": list(songs_to_labels.keys()), "Label": list(songs_to_labels.values())})
    print(df)
    np.savetxt(new_annotated_path, df, delimiter=';', fmt='%s')

def _get_tracks_without_features_or_labels(sound_folder_path:str):
    """Only sound files can be in the sound_folder_path directory"""
    tracks = []

    files = os.listdir(sound_folder_path)
    for file in files:
        track = Track()
        track.sound = librosa.load(os.path.join(sound_folder_path, file))[0]
        track.name = get_name_from_path(os.path.join(sound_folder_path, file))
        track.original_track, track.source = get_original_name_and_source_from_file_name(file)
        tracks.append(track)

    return tracks

def _assign_labels_to_tracks(tracks:list, annotated_path:str):
    csv_content:np = np.loadtxt(annotated_path, delimiter=';', dtype=np.object_, encoding='utf-8-sig')

    names:list = list(csv_content[:, 0])
    print(names)

    for track in tracks:
        print(names.index(track.original_track))

    return csv_content

def get_cal_tracks(annoated_path:str, sound_folder_path:str, feature_xml_path):
    """Only sound files can be in the sound_folder_path directory"""

    #names_and_features = _get_names_and_features_from_xml(feature_xml_path)
    tracks = _get_tracks_without_features_or_labels(sound_folder_path)
    _assign_labels_to_tracks(tracks, annoated_path)
    #tracks = _assign_features_to_tracks(tracks, names_and_features)




#_generate_new_label_file("datasets/New dataset/annotated data for cal500.csv")
get_cal_tracks("datasets/New dataset/new_annotated.csv", "datasets/New dataset/Clips", None)