from enum import Enum

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler

class Scalers(Enum):
    STANDARD = 0
    ROBUST = 1
    MAXABS = 2
    MINMAX = 3

def scale_features(track_list:list, scaler:Scalers | int) -> list:
    new_track_list = track_list.copy()

    features = [track.features for track in track_list]


    match scaler:
        case Scalers.STANDARD: chosen_scaler = StandardScaler()

        case Scalers.ROBUST: chosen_scaler = RobustScaler()

        case Scalers.MAXABS: chosen_scaler = MaxAbsScaler()

        case Scalers.MINMAX: chosen_scaler = MinMaxScaler()

        case _: chosen_scaler = None

    chosen_scaler.fit(features)

    for track in new_track_list:
        track.features = chosen_scaler.transform([track.features])[0]

    return new_track_list