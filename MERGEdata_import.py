import os
import librosa
from os import listdir
from os.path import isfile, join
import numpy as np
from our_classes import Track
from data_import import _get_names_and_features_from_xml, get_name_from_path, _assign_features_to_tracks, _get_tracks_without_features_or_labels, _make_track, get_original_name_and_source_from_file_name
#from SVM_test import labels
from data_import import _get_names_and_features_from_xml


def removing_redundant_characters():
    song_and_quadrant = []
    load_song_name_and_quadrant = np.loadtxt('MERGE-datas/merge_audio_balanced_metadataMIN2(1).csv', delimiter= ';', ndmin=2, dtype=np.object_)
    return load_song_name_and_quadrant[:, 0:2]


def separating_source_name_part(clip_name):
    names_and_quadrants_list = removing_redundant_characters()
    names = list(names_and_quadrants_list[:,0]) #a list with all the og names
    og_name = (names_and_quadrants_list[names.index(clip_name), 1]) #Q corresponding to the og name in the paranthesis

    return og_name

def filling_track_list(clips_folder_path: str, xml_file_path, tracks_amount: int = None):
    tracks_list = _get_tracks_without_features_or_labels(clips_folder_path, tracks_amount)

    for track in tracks_list:
        quadrant = separating_source_name_part(track.original_track)
        track.label = quadrant

    features = _get_names_and_features_from_xml(xml_file_path)
    tracks_with_features = _assign_features_to_tracks(tracks_list, features)
    return tracks_with_features

filled_tracks_list = filling_track_list('MERGE-datas/AllSongs15Sec', 'MERGE-datas/feature_values_1.xml', None)
print(filled_tracks_list[0])
""""
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