import os
import librosa
from os import listdir
from os.path import isfile, join
import numpy as np
from our_classes import Track
from data_import import _get_names_and_features_from_xml, get_name_from_path, _assign_features_to_tracks, _get_tracks_without_features_or_labels, _make_track, get_original_name_and_source_from_file_name
#from SVM_test import labels
from data_import import _get_names_and_features_from_xml

original_folder_path= "MERGE-datas/AllSongsQ1-4"
path_to_SS_clips = "MERGE-datas/AllSongs15Sec"

songs=[]

def removing_redundant_characters():
    song_and_quadrant = []
    load_song_name_and_quadrant = np.loadtxt('MERGE-datas/merge_audio_balanced_metadataMIN2(1).csv', delimiter= ';', ndmin=2, dtype=np.object_)
    return load_song_name_and_quadrant[:, 0:2]


SS_and_clipped_audio_list = [f for f in listdir('MERGE-datas/AllSongs15Sec') if isfile(join('MERGE-datas/AllSongs15Sec', f))]

def separating_source_name_part():
    names_and_quadrants_list = removing_redundant_characters()
    names = list(names_and_quadrants_list[:,0])
    print(names)
    print(names_and_quadrants_list[names.index('A013'), 1])

test = _get_tracks_without_features_or_labels("MERGE-datas/AllSongs15Sec")
print(test)

#testtwo, _ = get_original_name_and_source_from_file_name("Instrumental_A001_PT1.mp3")
#print(testtwo)

""""
# Sti til XML-fil med features
xml_path = "MERGE-datas/feature_values_15sPT2.xml"
# Hent alle features og filnavne
features = _get_names_and_features_from_xml(xml_path)
# Vis hvor mange tracks der er fundet
#print("Number of feature sets loaded:", len(features))
# Vis de første 3
for name, vector in features[:3]:
    print("\nTrack:", name)
    print("First 5 features:", vector[:5])


#loop køres igennem mappen af sangenee (så vi kan arbejde med enkelte sange)
for file in os.listdir(folder_path): #os.listdir returnere en liste med alle filnavne i mappen
    if file.endswith(".mp3"): #if statement, som tjekker om filen slutter på .mp3
        file_path = os.path.join(folder_path, file) #sætter file og folder_path sammen, gør dem til én sti/path i filen


        try: #tester den kommende kode for fejl/erorrs (hvis der er, fanges den senere i except)
            #loader audio filerne
            audio = librosa.load(file_path) #lydfilen bliver indlæst med librosa - vi får sr og audio(numpy array med alle lyddataerne)
            #sr=none betyder at vi bruger den OG sr fra mappen

            #gemmer det hele i listen tidligere lavet
            songs.append(file)
            songs.append(audio)

        except Exception as e: #except håndterer fejlen
            print("Could not load", file_path)
            print("Error:", e)


print("Done. Total mp3 songs loaded:", len(songs))
"""