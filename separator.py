from turtledemo.penrose import start

from audio_separator.separator import Separator
import soundfile
import os
import shutil
from humanfriendly.terminal import output


def convert_folder_to_wav(should_override):
    if should_override: shutil.rmtree("datasets\emotify\Emotify_music_wavs")
    os.makedirs("datasets\emotify\Emotify_music_wavs", exist_ok=True)

    i = 0
    genre_folders = os.listdir("datasets\emotify\emotify_music")
    for genre in genre_folders:
        genre_path = os.path.join("datasets\emotify\emotify_music", genre)
        songs = os.listdir(genre_path)
        os.makedirs(os.path.join("datasets\emotify\Emotify_music_wavs", genre), exist_ok=True)
        for song in songs:
            i += 1
            print(i)
            sound, sr = soundfile.read(os.path.join(genre_path, song))
            soundfile.write(os.path.join("datasets\emotify\Emotify_music_wavs", genre, song[0] + ".wav"), sound, sr, format="WAV")

def _execute_separation(separator, song_path, output_path):

    song_name = os.path.split(song_path)[1]
    song_name, song_format = song_name.split(".")

    song, sr = soundfile.read(song_path)
    soundfile.write(os.path.join(output_path, "Mixed_" + song_name + f".{song_format}"), song, sr)

    source_names = {
        "Vocals": os.path.join(output_path, "Vocals_" + song_name),
        "Instrumental": os.path.join(output_path, "Instrumental_" + song_name)
    }
    separator.separate(song_path, source_names)

def separate(input_path, output_path, should_override):
    output_folder = os.path.join(output_path, "Separated_and_mixed_versions")

    if should_override: shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    sep = Separator(output_format="MP3")
    sep.load_model()
    i = 0
    input_sub_folders = os.listdir(input_path)
    for sub_folder in input_sub_folders:
        sub_item_path = os.path.join(input_path, sub_folder)
        if os.path.isdir(sub_item_path):
            output_sub_folder = os.path.join(output_folder, sub_folder)
            os.makedirs(output_sub_folder, exist_ok=True)


            songs = os.listdir(sub_item_path)
            print(sub_item_path)
            for song in songs:
                song_path = os.path.join(sub_item_path, song)
                print(sep.output_dir)
                i += 1
                print(i)
                if not should_override and os.path.exists(os.path.join(output_sub_folder, "Vocals_" + song)): continue
                _execute_separation(sep, song_path, output_sub_folder)
        else:
            if os.path.split(sub_item_path)[1][:4] != ".mp3" or ".wav": continue
            _execute_separation(sep, sub_item_path, output_folder)

separate(input_path="datasets/emotify/emotify_music", output_path="datasets/emotify", should_override= False)
