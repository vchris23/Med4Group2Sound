import shutil
import os
from collections import defaultdict
import math
import soundfile
import numpy as np
from audio_separator.separator import Separator
from separator import _execute_separation


list = np.fromtxt("CAL500_32kps")
dict = defaultdict(list)
for row in list:
    dict[row[0]] = row[1]


separate(input_path="datasets/emotify/emotify_music", output_path="datasets/emotify/very_separated", should_override= False, model_file_name="htdemucs_6s.yaml")
#split_mp3_file_into_excerpts("datasets/emotify/emotify_music/classical/1.mp3","datasets/emotify/clips", 15)
#split_sound_files_in_folders_into_excerpts("datasets/emotify/Separated_and_mixed_versions", "datasets/emotify/clips", 15)

