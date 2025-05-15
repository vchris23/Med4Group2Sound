import random
from itertools import groupby


from our_classes import Track
random.seed(42)
def split_into_train_and_test(track_list:list, ratio:float, seed:int = None, stratify:bool = False):

    track_list.sort(key = lambda track: track.original_track)
    track_list = [([track for track in cont]) for cat, cont in groupby(track_list, lambda track: track.original_track)]
    #groups the tracks based on their original track, since they need to be grouped properly for testing

    randomizer = random.Random()
    if seed is not None: randomizer.seed(seed)
    randomizer.shuffle(track_list)

    tracks_nr = len(track_list)
    tracks_ratio = tracks_nr*ratio
    rounded_tracks_ratio = round(tracks_ratio)

    if stratify == False:
        training_set:list = [track for track_list in track_list[0:rounded_tracks_ratio] for track in track_list] #Flattens the list
        test_set:list = [track for track_list in track_list[-(tracks_nr - rounded_tracks_ratio):] for track in track_list]
    else:
        training_set:list = []
        test_set:list = []

        for tracks in Track.separate_tracks_by_label([track for track_list in track_list for track in track_list]):
            part_training_set, part_test_set = split_into_train_and_test(tracks, ratio, seed, stratify=False)
            training_set.extend(part_training_set)
            test_set.extend(part_test_set)

    return training_set, test_set