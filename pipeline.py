from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from process_sound import split_into_train_and_test
from data_import import import_tracks
from our_classes import Track

all_tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="feature_values_1.xml",
                     sources=["Instrumental", "Mixed", "Vocals"], amount_to_take=None)

training_set, test_set = split_into_train_and_test([track for track in all_tracks if track.source == "Mixed"], 0.8)
train_features, train_labels = Track.tracks_to_labels_and_features(all_tracks)

pipeline = Pipeline([("scaler", StandardScaler()), ("PCA", PCA()), ("svm", SVC())])
