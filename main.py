from data_import import import_tracks
from our_classes import Track
from process_sound import split_into_train_and_test

tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="feature_values_1.xml",
                     sources=["Instrumental", "Mixed", "Vocals"], amount_to_take=24)
train_tracks, test_tracks = split_into_train_and_test(tracks, 0.5)