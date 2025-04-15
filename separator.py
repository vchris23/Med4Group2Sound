from turtledemo.penrose import start

import audio_separator.utils.cli
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
        "Instrumental": os.path.join(output_path, "Instrumental_" + song_name),
        "Drums": os.path.join(output_path, "Drums_" + song_name),
        "Bass": os.path.join(output_path, "Bass_" + song_name),
        "Guitar": os.path.join(output_path, "Guitar_" + song_name),
        "Piano": os.path.join(output_path, "Piano_" + song_name),
        "Other": os.path.join(output_path, "Other_" + song_name)
    }
    separator.separate(song_path, source_names)

def separate(input_path, output_path, should_override, model_file_name = None):
    output_folder = os.path.join(output_path, "Separated_and_mixed_versions")

    if should_override: shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    sep = Separator(output_format="MP3", use_soundfile=True)
    sep.load_model() if model_file_name is None else sep.load_model(model_file_name)
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
            file_type = os.path.split(sub_item_path)[1][-4:]
            if file_type != ".mp3" and file_type != ".wav": continue
            if not should_override and os.path.exists(os.path.join(output_folder, "Vocals_" + os.path.split(sub_item_path)[1])): continue
            _execute_separation(sep, sub_item_path, output_folder)

def _split_mp3_file_into_excerpts(song_path, output_folder_path, clip_length_in_seconds):
    song:soundfile.SoundFile = soundfile.SoundFile(song_path)
    clips = song.blocks(blocksize=song.samplerate * clip_length_in_seconds)
    i = 0
    file_folder, song_name = os.path.split(song_path)
    title, format = os.path.splitext(song_name)

    for clip in clips:
        clip_duration = len(clip)/(song.samplerate)
        if clip_duration < clip_length_in_seconds/3: continue #Avoids too short clips
        i += 1
        print("input:",song_path, "output",os.path.join(output_folder_path, f"{title}_PT{i}{format}"))
        print(clip_duration, "samplerate: ", song.samplerate)
        soundfile.write(file = os.path.join(output_folder_path, f"{title}_PT{i}{format}"), data = clip, samplerate=song.samplerate)

def split_sound_files_in_folders_into_excerpts(input_folder_path, output_folder_path, clip_length_in_seconds):

    input_folder_content = os.listdir(input_folder_path)

    input_folder_mp3s = [input_content for input_content in input_folder_content if input_content[-3:] == "mp3"]

    for mp3_file in input_folder_mp3s:

        _split_mp3_file_into_excerpts(song_path=os.path.join(input_folder_path, mp3_file),
                                      output_folder_path=output_folder_path,
                                      clip_length_in_seconds=clip_length_in_seconds)
        print(os.path.join(input_folder_path, mp3_file))


    input_folder_sub_folders = [input_content for input_content in input_folder_content if
                                os.path.isdir(os.path.join(input_folder_path, input_content))] #Gets folders for recursion

    for sub_folder in input_folder_sub_folders:
        os.makedirs(os.path.join(output_folder_path, sub_folder), exist_ok=True)
        split_sound_files_in_folders_into_excerpts(os.path.join(input_folder_path, sub_folder),
                                                   os.path.join(output_folder_path, sub_folder),
                                                   clip_length_in_seconds)



#separate(input_path="MERGE-datas/AllSongsQ1-4", output_path="MERGE-datas/AllSongsSourceSep", should_override= False, model_file_name="htdemucs_6s.yaml")
#split_mp3_file_into_excerpts("datasets/emotify/emotify_music/classical/1.mp3","datasets/emotify/clips", 15)
split_sound_files_in_folders_into_excerpts("MERGE-datas/AllSongsSourceSep/Separated_and_mixed_versions", "MERGE-datas/AllSongs15Sec", 15)