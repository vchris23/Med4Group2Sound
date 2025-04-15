from os import listdir
from os.path import isfile, join
import numpy as np

categories_list = np.loadtxt('Dataset/categories.txt', dtype = str, delimiter="@")
clusters_list = np.loadtxt('Dataset/clusters.txt', dtype = str, delimiter="@")

audio_files = [f for f in listdir('Dataset/Audio') if isfile(join('Dataset/Audio', f))]
print(audio_files)