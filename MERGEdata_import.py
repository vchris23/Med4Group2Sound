import os
import librosa
from data_import import _get_names_and_features_from_xml
#kald nu så denne function for at bruge den heri det her script


#mappe hvor alle sangene er (definere så vi kan bruge den nemt/hurtigt senere)
folder_path= "MERGE-datas/AllSongsQ1-4"

#laver en liste for informationer om alle sangene (som vi kan bruge senere
songs=[]

#loop køres igennem mappen af sangenee (så vi kan arbejde med enkelte sange)
for file in os.listdir(folder_path): #os.listdir returnere en liste med alle filnavne imappen
    if file.endswith(".mp3"): #if statement, som tjekker om filen slutter på .mp3
        file_path = os.path.join(folder_path, file) #sætter file og folder_path sammen, gør dem til én sti/path i filen

        try: #tester den kommende kode for fejl/eroors (hvis der er, fanges den senere i except)
            #loader audio filerne
            audio, sr = librosa.load(file_path, sr=None) #lydfilen bliver indlæst med librosa - vi får sr og audio(numpy array med alle lyddataerne)
            #sr=none betyder at vi bruger den OG sr fra mappen


            #gemmer det hele i listen tideligere lavet
            songs.append({  #.append vi ændrer på listen
                "filename": file, #navnet på filen i mappen
                "audio": audio, #lyddataen som en array
                "sample_rate": sr #hvor mange samples der er pr sek
            })

            print("loaded:", file_path)

        except Exception as e: #except håndterer fejlen
            print("Could not load", file_path)
            print("Error:", e)


print("Done. Total mp3 songs loaded:", len(songs))
