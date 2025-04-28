from os import listdir
from os.path import isfile, join
import numpy as np
from decorator import append
import librosa

from our_classes import Track
from data_import import _get_names_and_features_from_xml

#Labelsss lists
categories_list = np.loadtxt('datasets/MIREX-like_mood/categories.txt', dtype = str, delimiter="@")
clusters_list = np.loadtxt('datasets/MIREX-like_mood/clusters.txt', dtype = str, delimiter="@")

#OG audios list
audio_files = [f for f in listdir('datasets/MIREX-like_mood/Audio') if isfile(join('datasets/MIREX-like_mood/Audio', f))]

#Source separated audios list
SS_clips_list = [f for f in listdir('datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions') if isfile(join(
  'datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions', f))]


xml_names_and_features = _get_names_and_features_from_xml('datasets/MIREX-like_mood/feature_values_1.xml')

tracks_list = []
for a in SS_clips_list:
  index = int(a.split("_")[1]) #casting to an int, splitting each element and returning the first index
  correct_index = index - 1
  audio_source = a.split("_")[0] #splitting each element and returning the zero'th index

  t = Track(sound = a, label = categories_list[correct_index], original_track=audio_files[correct_index], source = audio_source)
  tracks_list.append(t)

print(xml_names_and_features)

