pip install pydub



new_tracks=[] #makes new list to store new tracks

for track in tracks:
    sound = track["sound"]  #gets the sound from tracks

for start in range(0, duration, chunk_size):
    excerpt = sound[start:start + chunk_size]  # Lav et udsnit af lyden


