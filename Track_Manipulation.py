import librosa

def split_tracks_into_excerpts(tracks):
    new_tracks = []  #laver en ny liste, til de nye tracks

    for track in tracks:
        sound = track["sound"]  #lyden bliver hentet, fra tracket
        chunk_size = 15  #størrelsen af hver chunk sættes

        #lyder bliver delt op i chunks af 15 sek
        for start in range(0, len(sound), chunk_size):
            excerpt = sound[start:start + chunk_size]  #laver en bid af lyden
            new_track = track.copy()  #laver en kopi af tracket
            new_track["sound"] = excerpt  #opdater den kopierede track, med den nye lyd
            new_tracks.append(new_track)  #tilføjer det nye track til listen (fra toppen)

    return new_tracks  #returner listen med alle de nye tracks

#eksempel på input
tracks = [
    {"id": 1, "sound": list(range(60))},  # 60 elementer i lyd
    {"id": 2, "sound": list(range(45))}   # 45 elementer i lyd
]

#kalder funktionen og resultatet printes
output = split_tracks_into_excerpts(tracks)

for track in output:
    print("ID:", track["id"], "Sound:", track["sound"])