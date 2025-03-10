import random
from itertools import groupby


from our_classes import Track

def split_into_train_and_test(track_list:list, ratio:float):

    track_list = [([track for track in cont]) for cat, cont in groupby(track_list, lambda track: track.original_track)]
    #groups the tracks based on their original track, since they need to be grouped properly for testing

    print(track_list)
    print(len(track_list))

    random.shuffle(track_list)

    tracks_nr = len(track_list)
    tracks_ratio = tracks_nr*ratio
    rounded_tracks_ratio = round(tracks_ratio)

    training_set:list = [track for track_list in track_list[0:rounded_tracks_ratio] for track in track_list] #Flattens the liost
    print(training_set)
    test_set:list = [track for track_list in track_list[-rounded_tracks_ratio:] for track in track_list]

    print(f"training len {len(training_set)}, test len {len(test_set)}")

    for track in training_set:
     track.train_or_test = "train"
    #for track in training_set:
       # print(track.train_or_test)

    for track in test_set:
        track.train_or_test = "test"

    return training_set, test_set