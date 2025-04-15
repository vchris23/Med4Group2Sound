import librosa

def split_tracks_into_excerpts(tracks, sampling_rate):
    new_tracks = []  #laver en ny liste, til de nye tracks

    for track in tracks:
        sound = track.sound #lyden bliver hentet, fra tracket
        chunk_size = sampling_rate * 15

        #lyder bliver delt op i chunks af 15 sek
        for start in range(0, len(sound), chunk_size):
            excerpt = sound[start:start + chunk_size]  #laver en bid af lyden på 15 sek
            new_track = track.copy()  #laver en kopi af tracket
            new_track.sound = excerpt  #opdaterer den kopierede track, med den nye lyd
            new_tracks.append(new_track)  #tilføjer det nye track til listen (fra toppen)

    return new_tracks  #returner listen med alle de nye tracks