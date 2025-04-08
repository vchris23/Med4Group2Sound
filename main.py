from collections import defaultdict
from itertools import groupby

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

from data_import import import_tracks, get_feature_names
from our_classes import Track
from process_sound import split_into_train_and_test
from music_svm import classify_by_original_track, combined_classification_by_original_track, get_scores, \
    get_confusion_matrix, classify_tracks, combined_classify_tracks, select_features, get_untrained_SVM
from music_svm import get_trained_SVM
from preprocess import scale_features
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import pandas as pd

all_tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="feature_values_1.xml",
                     sources=["Instrumental", "Mixed", "Vocals"], amount_to_take=None)

all_tracks = [track for track in all_tracks if track.label[0] != "amazement"]

all_tracks.sort(key = lambda track: track.label)
for grouping in groupby(all_tracks, lambda track: track.label):
    print(grouping[0], len(list(grouping[1])))

all_tracks.sort(key = lambda track: track.source)

for feature_vector in [track.features for track in all_tracks]:
    plt.scatter(x=range(len(feature_vector)), y=feature_vector)
plt.title("Before")
plt.show()


new_all_tracks = scale_features(all_tracks)

features = [track.features for track in new_all_tracks]

for feature_vector in features:
    plt.scatter(x=range(len(feature_vector)), y=feature_vector)
plt.title("After")
plt.show()

data_frame = pd.DataFrame(features, columns=get_feature_names("feature_values_1.xml", True))
data_frame.head(5)
cor = data_frame.corr()
cor.to_csv("Corr tester.csv", header = True)
data_frame.to_csv("Pandas tester.csv", header = True)

1/0

training_tracks, test_tracks = split_into_train_and_test(new_all_tracks, 0.8, 42)

training_tracks = [track for track in training_tracks if track.source == "Instrumental"]

indices, names = select_features(get_untrained_SVM(use_ova=True), training_tracks, number_of_features_to_select=5, feature_names=get_feature_names("feature_values_1.xml", with_chroma=True))

print(names, indices)
print("For Instrumental")
1/0
mixed_scores = []
late_scores = []

mixed_labels = []
mixed_predictions = []
late_labels = []
late_predictions = []
for i in range(0, 220, 40):
    print(f"{i}/{220}")
    training_tracks, test_tracks = split_into_train_and_test(new_all_tracks, 0.8, i)

    mixed_test_tracks = [track for track in test_tracks if track.source == "Mixed"]
    mixed_training_tracks = [track for track in training_tracks if track.source == "Mixed"]

    vocal_test_tracks = [track for track in test_tracks if track.source == "Vocals"]
    vocal_training_tracks = [track for track in training_tracks if track.source == "Vocals"]

    instrumental_test_tracks = [track for track in test_tracks if track.source == "Instrumental"]
    instrumental_training_tracks = [track for track in training_tracks if track.source == "Instrumental"]

    mixed_svm = get_trained_SVM(mixed_training_tracks, use_oversampler=True)
    vocal_svm = get_trained_SVM(vocal_training_tracks, use_oversampler=True)
    instrumental_svm = get_trained_SVM(instrumental_training_tracks, use_oversampler=True)

    mixed_dic = classify_tracks(mixed_svm, mixed_test_tracks)
    vocal_dic = classify_tracks(vocal_svm, vocal_test_tracks)
    instrumental_dic = classify_tracks(instrumental_svm, instrumental_test_tracks)
    late_fusion_dict = combined_classify_tracks([vocal_svm, instrumental_svm],
                                                                 [vocal_test_tracks, instrumental_test_tracks])

    mixed_scores.append(get_scores([prediction for prediction in list(mixed_dic.values())], [track.label for track in list(mixed_dic.keys())])[1])
    mixed_labels.extend([track.label for track in list(mixed_dic.keys())])
    mixed_predictions.extend([prediction for prediction in list(mixed_dic.values())])

    late_scores.append(get_scores([prediction for prediction in list(late_fusion_dict.values())],
                                   [track.label for track in list(late_fusion_dict.keys())])[1])
    late_labels.extend([track.label for track in list(late_fusion_dict.keys())])
    late_predictions.extend([prediction for prediction in list(late_fusion_dict.values())])

get_confusion_matrix(mixed_predictions, mixed_labels)
get_confusion_matrix(late_predictions, late_labels)
print(f"mean of mixed precision: {np.mean(mixed_scores)} (STD: {np.std(mixed_scores)})")
print(f"mean of late precision: {np.mean(late_scores)} (STD: {np.std(late_scores)})")
1/0

print("Got mixed svm")
mixed_dic = classify_tracks(mixed_svm, mixed_test_tracks)
vocal_dic = classify_tracks(vocal_svm, vocal_test_tracks)
instrumental_dic = classify_tracks(instrumental_svm, instrumental_test_tracks)
print("got result dic")

mixed_key_label_list = [track.label for track in list(mixed_dic.keys())]
mixed_prediction_list = [prediction for prediction in list(mixed_dic.values())]
get_confusion_matrix(mixed_prediction_list, mixed_key_label_list, mixed_svm)
print(f"score for mixed clips: {get_scores(mixed_prediction_list,mixed_key_label_list)}")

vocal_key_label_list = [track.label for track in list(vocal_dic.keys())]
vocal_prediction_list = [prediction for prediction in list(vocal_dic.values())]
get_confusion_matrix(vocal_prediction_list, vocal_key_label_list, vocal_svm)
print(f"score for vocals clips: {get_scores(vocal_prediction_list,vocal_key_label_list)}")

instrumental_key_label_list = [track.label for track in list(instrumental_dic.keys())]
instrumental_prediction_list = [prediction for prediction in list(instrumental_dic.values())]
get_confusion_matrix(instrumental_prediction_list, instrumental_key_label_list, instrumental_svm)
print(f"Scores for instrumental clips: {get_scores(instrumental_prediction_list,instrumental_key_label_list)}")

comb_result_dict = combined_classify_tracks([vocal_svm, instrumental_svm], [vocal_test_tracks, instrumental_test_tracks])
comb_label_list = [track.label for track in list(comb_result_dict.keys())]
comb_pred_list = [label for label in list(comb_result_dict.values())]
get_confusion_matrix(comb_pred_list, comb_label_list)
print(f"Scores for combined clip wise: {get_scores(comb_pred_list, comb_label_list)}")

print("Did the listing of results")


mixed_dic = classify_by_original_track(mixed_svm, mixed_test_tracks)
vocal_dic = classify_by_original_track(vocal_svm, vocal_test_tracks)
instrumental_dic = classify_by_original_track(instrumental_svm, instrumental_test_tracks)

mixed_key_label_list = [track.label for track in list(mixed_dic.keys())]
mixed_prediction_list = [prediction for prediction in list(mixed_dic.values())]
get_confusion_matrix(mixed_prediction_list, mixed_key_label_list, mixed_svm)
print(f"score for mixed by original track: {get_scores(mixed_prediction_list,mixed_key_label_list)}")

vocal_key_label_list = [track.label for track in list(vocal_dic.keys())]
vocal_prediction_list = [prediction for prediction in list(vocal_dic.values())]
get_confusion_matrix(vocal_prediction_list, vocal_key_label_list, vocal_svm)
print(f"score for vocals by original track: {get_scores(vocal_prediction_list,vocal_key_label_list)}")

instrumental_key_label_list = [track.label for track in list(instrumental_dic.keys())]
instrumental_prediction_list = [prediction for prediction in list(instrumental_dic.values())]
get_confusion_matrix(instrumental_prediction_list, instrumental_key_label_list, instrumental_svm)
print(f"Scores for instrumental by original track: {get_scores(instrumental_prediction_list,instrumental_key_label_list)}")

late_fusion_dict = combined_classification_by_original_track([vocal_svm, instrumental_svm], [vocal_test_tracks, instrumental_test_tracks])
late_fusion_key_label_list = [track.label for track in list(late_fusion_dict.keys())]
late_fusion_prediction_list = [prediction for prediction in list(late_fusion_dict.values())]
get_confusion_matrix(late_fusion_prediction_list, late_fusion_key_label_list)
print(f"scores for late fusion original track: {get_scores(late_fusion_prediction_list, late_fusion_key_label_list)}")

print("Note, balancing")