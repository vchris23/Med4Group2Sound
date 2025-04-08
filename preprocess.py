from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler


def scale_features(track_list:list):
    new_track_list = track_list.copy()

    features = [track.features for track in track_list]

    scaler = StandardScaler()

    scaler.fit(features)

    for track in new_track_list:
        track.features = scaler.transform([track.features])[0]

    return new_track_list

