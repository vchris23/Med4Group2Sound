import os
import librosa

#from SVM_test import labels
from data_import import _get_names_and_features_from_xml
import numpy as np
#kald nu så denne function for at bruge den heri det her script
#se onenote for mere hjælp/noter


#mappe hvor alle sangene er (definere så vi kan bruge den nemt/hurtigt senere)
folder_path= "MERGE-datas/AllSongsQ1-4"

#laver en liste for informationer om alle sangene (som vi kan bruge senere
songs=[]
"""
def removing_redundant_characters(characters_to_remove):
    song_and_quadrant = []
    load_song_name_and_quadrant = np.loadtxt('MERGE-datas/merge_audio_balanced_metadataMIN2(1).csv', delimiter= ';', ndmin=2, dtype=str)

    for item in load_song_name_and_quadrant:
        updated_item = item[:-characters_to_remove]
        song_and_quadrant.append(updated_item)
    return song_and_quadrant


labels_list = []
original_tracks_list = []
names_and_quadrants_list = removing_redundant_characters(7)
index = 0
for item in names_and_quadrants_list:
    for name_or_label in item:
        if index == name_or_label[0]:
            original_tracks_list.append(name_or_label)
        else:
            labels_list.append(name_or_label)

print(labels_list)


"""
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
        print(file_path)
        """
        try: #tester den kommende kode for fejl/eroors (hvis der er, fanges den senere i except)
            #loader audio filerne
            audio, sr = librosa.load(file_path) #lydfilen bliver indlæst med librosa - vi får sr og audio(numpy array med alle lyddataerne)
            #sr=none betyder at vi bruger den OG sr fra mappen


            #gemmer det hele i listen tidligere lavet
            songs.append({  #.append vi ændrer på listen
                "filename": file, #navnet på filen i mappen
                "audio": audio, #lyddataen som en array
                "sample_rate": sr #hvor mange samples der er pr sek
            })

            print("loaded:", file_path)
            
        except Exception as e: #except håndterer fejlen
            print("Could not load", file_path)
            print("Error:", e)
        """

#print("Done. Total mp3 songs loaded:", len(songs))
