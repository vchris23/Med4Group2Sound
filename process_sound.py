import random
from our_classes import Track

tracks = [Track() for i in range (0, 100)]
random.shuffle(tracks)

def set_into_train_or_test(track_list, ratio):

    tracks_nr = len(track_list)
    tracks_ratio = tracks_nr*ratio
    rounded_tracks_ratio = round(tracks_ratio)

    training_set = track_list[0:rounded_tracks_ratio]
    test_set = track_list[rounded_tracks_ratio:]

    for track in training_set:
     track.train_or_test = "train"
    #for track in training_set:
       # print(track.train_or_test)

    for track in test_set:
        track.train_or_test = "test"
    #for track in test_set:
        #print(track.train_or_test
    complete_set = []
    complete_set.extend(training_set)
    complete_set.extend(test_set)

set_into_train_or_test(tracks, 0.8)
