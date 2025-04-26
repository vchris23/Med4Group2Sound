from collections import defaultdict
from itertools import groupby

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sympy import false

from data_import import import_tracks, get_feature_names
from our_classes import Track
from process_sound import split_into_train_and_test
from music_svm import classify_by_original_track, combined_classification_by_original_track, get_scores, \
    get_confusion_matrix, classify_tracks, combined_classify_tracks, select_features, get_untrained_SVM, \
    classify_and_get_scores
from music_svm import get_trained_SVM
from preprocess import scale_features, Scalers
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import pandas as pd
from Analysis import _get_best_dimensionality, get_fitted_pca, transform_features_of_tracks, \
    save_correlation_of_pca_and_og_features, get_pca_tracks_and_correlation, interpret_principal_components

all_tracks = import_tracks("datasets/emotify/very_separated_clips/Separated_and_mixed_versions", "datasets/emotify/emotify_data.csv", features_xml_path="feature_values_1.xml",
                     sources=["Bass", "Drums", "Guitar", "Mixed", "Other", "Piano", "Vocals"], amount_to_take=None)

all_tracks = [track for track in all_tracks if track.label[0] != "amazement"]

all_tracks.sort(key = lambda track: track.label)
for grouping in groupby(all_tracks, lambda track: track.label):
    print(grouping[0], len(list(grouping[1])))

all_tracks.sort(key = lambda track: track.source)

for feature_vector in [track.features for track in all_tracks]:
    plt.scatter(x=range(len(feature_vector)), y=feature_vector)
plt.title("Before")
plt.show()

for i in range(40, 60):
    val_list = [track.features[i] for track in all_tracks]
    print(f"Max in index {i}: {max(val_list)}")

scaler = Scalers.MINMAX
new_all_tracks = scale_features(all_tracks, scaler)
print("Using this scaler: ", scaler)

features = [track.features for track in new_all_tracks]

for feature_vector in features:
    plt.scatter(x=range(len(feature_vector)), y=feature_vector)
plt.title("After")
plt.show()

#indices, names = select_features(get_untrained_SVM(use_ova=True), training_tracks, number_of_features_to_select=5, feature_names=get_feature_names("feature_values_1.xml", with_chroma=True))

ratio = 0.96
oversampler_do = True
use_pca = True
feature_names = get_feature_names("feature_values_1.xml", True)

score_dict = defaultdict(dict)

for i in range(0, 220, 40):
    print(f"{i}/{220}")
    tracks = new_all_tracks

    training_tracks, test_tracks = split_into_train_and_test(tracks, 0.8, i, stratify=True)
    training_tracks_by_source = Track.separate_tracks_by_source(training_tracks)
    test_tracks_by_source = Track.separate_tracks_by_source(test_tracks)

    for source_i in range(len(training_tracks_by_source)):
        source_name = training_tracks_by_source[source_i][0].source

        source_training_tracks = training_tracks_by_source[source_i]
        source_test_tracks = test_tracks_by_source[source_i]

        if use_pca: source_training_tracks, source_test_tracks = get_pca_tracks_and_correlation(source_training_tracks, source_test_tracks, feature_names=feature_names, save_file_name=f"{source_name}", variance_ratio=ratio)

        svm = get_trained_SVM(source_training_tracks, use_oversampler=oversampler_do, rnd_state=i)

        acc, pre, rec = classify_and_get_scores(svm, source_test_tracks, plot_confusion_matrix=False)

        if score_dict[source_name].__contains__("acc") is False:
            score_dict[source_name]["acc"] = []
            score_dict[source_name]["pre"] = []
            score_dict[source_name]["rec"] = []

        score_dict[source_name]["acc"].append(acc)
        score_dict[source_name]["pre"].append(pre)
        score_dict[source_name]["rec"].append(rec)

for key in score_dict.keys():
    mean_scores = []
    stds = []
    for scorer in score_dict[key].keys():
        mean_scores.append(f"{scorer}: {np.mean(score_dict[key][scorer])}")
        stds.append(f"{scorer}: {np.std(score_dict[key][scorer])}")

    print(f"Scores for {key}: \n Mean: {mean_scores} \n Standard deviation {stds}")

if use_pca: print(f"This was done with PCA ratio at {ratio}")
else: print("This was done without PCA")

if oversampler_do: print("Using oversampler")

