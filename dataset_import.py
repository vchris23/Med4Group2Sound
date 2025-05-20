from os import listdir
from os.path import isfile, join
import numpy as np
from decorator import append
import librosa
import os

from our_classes import Track
from data_import import _get_names_and_features_from_xml, get_name_from_path, _assign_features_to_tracks

#OG audios list
original_audio_files = [f for f in listdir('datasets/MIREX-like_mood/Audio') if isfile(join('datasets/MIREX-like_mood/Audio', f))]

#Source separated audios list
SS_and_clipped_audio_list = [f for f in listdir('datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions') if isfile(join(
  'datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions', f))]


def make_tracks_list(path_to_clips:str, tracks_nr: int, path_to_categories: str = None, path_to_clusters: str = None,  using_clusters_instead_of_categories:bool = False):
  # Labelsss lists
  categories_list = np.loadtxt(path_to_categories, dtype=str, delimiter="@") if path_to_categories is not None else None
  clusters_list = np.loadtxt(path_to_clusters, dtype=str, delimiter="@") if path_to_clusters is not None else None

  tracks_list = []
  i = 0
  for a in SS_and_clipped_audio_list:
    sound_full_path = os.path.join(path_to_clips, a)
    sound, sampling_rate = librosa.load(sound_full_path)
    index = int(a.split("_")[1]) #casting to an int, splitting each element and returning the first index
    correct_index = index - 1
    audio_source = a.split("_")[0] #splitting each element and returning the zero'th index

    name_from_path = get_name_from_path(sound_full_path)

    if not using_clusters_instead_of_categories:
        label = categories_list[correct_index]
    else:
        label = clusters_list[correct_index]

    t = Track(sound = sound, label = label,  source = audio_source, original_track=original_audio_files[correct_index], name = name_from_path)
    tracks_list.append(t)

    i += 1
    if i > tracks_nr: break

  xml_names_and_features = _get_names_and_features_from_xml('datasets/MIREX-like_mood/feature_values_1.xml')
  tracks = _assign_features_to_tracks(tracks_list, xml_names_and_features)

  return tracks

#path to source separated clips (14.64s)
path_to_ss_clips = 'datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions/'
categories = 'datasets/MIREX-like_mood/categories.txt'
clusters = 'datasets/MIREX-like_mood/clusters.txt'

#final_list = make_tracks_list(path_to_ss_clips, 10, path_to_categories=categories, using_clusters_instead_of_categories=False)
#print(final_list[0])
