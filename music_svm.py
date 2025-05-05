import random
import time
from array import array
from pyexpat import features
from xml.sax.handler import feature_namespaces

import numpy
import pandas.plotting
import scipy
from sklearn import metrics
from sklearn import svm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sympy import false
from torch.utils.data import RandomSampler

from our_classes import Track
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from data_import import _vote_on_emotion_label, add_list, get_feature_names
from pandas import DataFrame
from pandas.plotting import table
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE

def _result_dic_to_predictions_and_labels(result_dic:dict):
    predictions = [prediction for prediction in result_dic.values()]
    labels = [track.label for track in result_dic.keys()]

    return labels, predictions

def get_untrained_SVM(use_ova:bool = False, kernel:str = 'rbf', rnd_state:int = 42, C:int = 10000, gamma = 0.01, probability:bool = True, class_weight_strategy:str = 'balanced', cache_size:int = 5012):
    classifier = svm.SVC(kernel = kernel, random_state = rnd_state, C = C, gamma = gamma, probability = probability, class_weight = class_weight_strategy, cache_size = cache_size)
    classifier = OneVsRestClassifier(classifier) if use_ova else classifier
    return classifier

def get_trained_SVM(training_tracks: list, use_oversampler = False, rnd_state = 42):

    classifier = get_untrained_SVM(use_ova=True, rnd_state=rnd_state)

    training_data, training_labels = Track.tracks_to_features_and_labels(training_tracks)


    if use_oversampler:
        before_count = len(training_data)
        oversampler = RandomOverSampler(random_state=rnd_state)
        training_data, training_labels = oversampler.fit_resample(training_data, training_labels)
        after_count = len(training_data)
        print(f"Oversampler added {after_count-before_count} more samples")

    # train model using the training data
    classifier = classifier.fit(training_data, training_labels)


    return classifier

def select_features(untrained_classifier, training_tracks: list, direction:str = "forward", feature_names:list[str] = None, number_of_features_to_select = None, number_of_cross_validations = None):
    """direction can either be \"forward\" or \"backward\" \n
    First output is a list of integers, the second output is the name of the features
    """

    training_data, training_labels = Track.tracks_to_features_and_labels(training_tracks)
    Track.tracks_features_to_dataframe(training_data, get_feature_names("feature_values_1.xml", True))

    selector_output = SequentialFeatureSelector(untrained_classifier, direction=direction, n_features_to_select=number_of_features_to_select, cv=number_of_cross_validations, n_jobs=-1)
    print("Got to the fitting")
    selector_output.fit(training_data, training_labels)

    print("Done with the fitting")
    i_ar = selector_output.get_support(indices=True)
    name_ar = [feature_names[i] for i in range(len(feature_names)) if i in i_ar]

    return i_ar, name_ar

def grid_param_search(svm, tracks):
    print("Doing grid param")
    grid = {'C': [1, 10, 100, 1000, 10000], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0001], 'kernel': ['rbf']}
    searcher = GridSearchCV(svm, grid, n_jobs=-1)

    training_data = [track.features for track in tracks]
    training_labels = [track.label[0] if len(track.label) != 0 else "none" for track in tracks] #flattens list

    searcher.fit(training_data, training_labels)
    frame = DataFrame.from_dict(searcher.cv_results_)
    frame.to_csv(f"balanced_oversampling_rbf_{tracks[0].source}.csv")

def random_search(svm, tracks):
    parameters = {'C': scipy.stats.expon(scale=10), 'gamma': scipy.stats.expon(scale=0.1), 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
    searcher = RandomizedSearchCV(svm, parameters, random_state=42, n_iter=10,n_jobs=-1)

    training_data = [track.features for track in tracks]
    training_labels = [track.label[0] if len(track.label) != 0 else "none" for track in tracks] #flattens list

    searcher.fit(training_data, training_labels)
    frame = DataFrame.from_dict(searcher.cv_results_)
    frame.to_csv(f"random_oversampling_rbf_{tracks[0].source}.csv")


def classify_and_get_conf_scores(svm:OneVsRestClassifier, track:Track | list, feature_names = None):

    feature_names = get_feature_names("feature_values_1.xml", True) if feature_names is None else feature_names
    if type(track) == list:
        dataframe_features = Track.tracks_features_to_dataframe(track, feature_names)

        y_pred_ova: list = svm.predict(dataframe_features)
        y_pred_ova: list = list(map(lambda x: str(x), y_pred_ova))

        y_confidence: list = svm.predict_proba(dataframe_features)
    else:
        dataframe_features = Track.tracks_features_to_dataframe([track], feature_names)
        y_pred_ova: str = str(svm.predict(dataframe_features)[0])
        y_confidence: list = svm.predict_proba(dataframe_features)

    return y_pred_ova, y_confidence


def get_scores(prediction_results, actual_results):
    """Return accuracy, precision, and recall"""

    acc = metrics.accuracy_score(actual_results, prediction_results)
    pre = metrics.precision_score(actual_results, prediction_results, average='macro', zero_division=np.nan)
    rec = metrics.recall_score(actual_results, prediction_results, average='macro', zero_division=np.nan)

    return acc, pre, rec

def get_confusion_matrix(prediction_results, actual_results, title:str = "Confusion matrix"):
    matrix = confusion_matrix(actual_results, prediction_results)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot()
    plt.title(title)
    plt.show()

def train_and_test_new_SVM(training_tracks: list, test_tracks: list):
    SVM = get_trained_SVM(training_tracks)

    predictions_and_confidences = [classify_and_get_conf_scores(SVM, track) for track in test_tracks]
    y_pred = [prediction_results[0] for prediction_results in predictions_and_confidences]
    y_actual = [track.label for track in test_tracks]

    scores = get_scores(y_pred, y_actual)
    return scores


def _get_conf_scores_by_original_track(svm:OneVsRestClassifier, tracks:list, feature_names:list = None):

    grouped_tracks = defaultdict(list)
    for track in tracks:
        grouped_tracks[track.original_track].append(track)

    result_dictionary = {}
    for original_track in grouped_tracks.keys():
        conf_scores = classify_and_get_conf_scores(svm, grouped_tracks[original_track], feature_names)[1]
        summed = list(sum(conf_scores)[0])
        result_dictionary[original_track] = summed

    return result_dictionary

def _class_from_confidence_scores(svm:OneVsRestClassifier, confidence_scores:list):
    classes = svm.classes_

    index_of_highest = confidence_scores.index(max(confidence_scores))
    return classes[index_of_highest]

def classify_by_original_track(svm:OneVsRestClassifier, tracks:list):

    results = _get_conf_scores_by_original_track(svm, tracks)

    lookup_table = defaultdict(list)

    for track in tracks:
        lookup_table[track.original_track].append(track)

    classification_results = {}

    for key in results.keys():
        classification_results[lookup_table[key][0]] = _class_from_confidence_scores(svm, results[key])

    return _result_dic_to_predictions_and_labels(classification_results)

def combined_classification_by_original_track(svms:list, track_lists:list):
    result_dictionaries = []
    for i in range(len(svms)):
        result_dictionaries.append(_get_conf_scores_by_original_track(svms[i], track_lists[i]))

    lookup_table = defaultdict(list)

    for track in track_lists[0]:
        lookup_table[track.original_track].append(track)

    combined_result_dict = {}

    for key in result_dictionaries[0].keys():
        scores = []
        for dictionary in result_dictionaries:
            scores.append(np.array(dictionary[key]))
        summed_scores:list = list(sum(scores))
        combined_result_dict[lookup_table[key][0]] = _class_from_confidence_scores(svms[0], summed_scores)

    return _result_dic_to_predictions_and_labels(combined_result_dict)

def classify_tracks(svm:OneVsRestClassifier, tracks:list, feature_names:list = None):
    """Returns list of predictions and list of labels"""

    result_dict = {}
    predictions = classify_and_get_conf_scores(svm, tracks, feature_names)[0] #[0] means we only get the predicted class

    for i in range(len(predictions)):
        result_dict[tracks[i]] = predictions[i]

    return _result_dic_to_predictions_and_labels(result_dict)


def combined_classify_tracks(svms:list, track_lists:list, feature_names:list = None):
    """Returns list of predictions and list of labels"""
    svm_dic = {}
    for i in range(len(track_lists)):
        svm_dic[track_lists[i][0].source] = svms[i]

    track_dic = defaultdict(list)

    for tracks in track_lists:
        for track in tracks:
            track_name:str = track.name
            track_genre:str = track_name.split("\\")[0]
            shared_name = f"{track_genre}\\{track_name.split('_', 1)[1]}"
            track_dic[shared_name].append(track)

    conf_dic = defaultdict(list)
    for shared_name in track_dic.keys():
        for track in track_dic[shared_name]:
            conf_dic[shared_name].append(classify_and_get_conf_scores(svm_dic[track.source], track, feature_names)[1])

    result_dict = {}
    for shared_name in conf_dic.keys():

        score_lists = conf_dic[shared_name]

        summed_list = score_lists[0][0]

        for i in range(1, len(score_lists)):
            summed_list = add_list(summed_list, score_lists[i][0])
            result_dict[track_dic[shared_name][0]] = _class_from_confidence_scores(svms[0], list(summed_list))


    return _result_dic_to_predictions_and_labels(result_dict)

def classify_and_get_scores(svm:OneVsRestClassifier | list, tracks:list, plot_confusion_matrix:bool = False, confusion_matrix_title:str = None, feature_names:list = None):
    """ If svm is a list, ensemble classification is assumed, so tracks must be a list of lists of tracks \n
    Returns accuracy, precision, and recall"""

    if type(svm) == list:
        predictions, labels = combined_classify_tracks(svm, tracks, feature_names)

    else:
        predictions, labels = classify_tracks(svm, tracks, feature_names)

    if plot_confusion_matrix:
        get_confusion_matrix(predictions, labels, confusion_matrix_title)

    return get_scores(predictions, labels)

def classify_by_original_track_and_get_scores(svm:OneVsRestClassifier | list, tracks:list, plot_confusion_matrix:bool = False, confusion_matrix_title:str = None, feature_names:list = None):
    """ If svm is a list, ensemble classification is assumed, so tracks must be a list of lists of tracks \n
    Returns accuracy, precision, and recall"""

    for track in tracks:
        print(track)

    if type(svm) == list:
        predictions, labels = combined_classification_by_original_track(svm, tracks, feature_names)

    else:
        predictions, labels = classify_by_original_track(svm, tracks, feature_names)

    if plot_confusion_matrix:
        get_confusion_matrix(predictions, labels, confusion_matrix_title)

    return get_scores(predictions, labels)