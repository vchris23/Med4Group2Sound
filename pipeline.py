import time
from collections import defaultdict
from copy import copy
from enum import Enum
import traceback
from random import random, Random

import numpy as np
import numpy.random
from imblearn.over_sampling import RandomOverSampler
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC

from dataframe_test import dataframe
from music_svm import grid_param_search, get_scores, classify_and_get_scores, classify_by_original_track_and_get_scores
from process_sound import split_into_train_and_test
from data_import import import_tracks, get_feature_names
from our_classes import Track
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import warnings
warnings.filterwarnings('always')

class Searchers(Enum):
    RANDOM = 0
    GRID = 1

def pipeline_search(pipeline, training_tracks, search_attributes:dict | list, search_type:Searchers, use_oversampler = False, feature_names:list = None):

    feature_names = get_feature_names("feature_values_1.xml", True) if feature_names is None else feature_names

    features, labels = Track.tracks_to_features_and_labels(training_tracks)
    features = Track.tracks_features_to_dataframe(training_tracks, feature_names)
    if use_oversampler:
        features, labels = RandomOverSampler(random_state=42).fit_resample(features, labels)

    match search_type:
        case Searchers.RANDOM:
            searcher = RandomizedSearchCV(pipeline, search_attributes, cv=3, n_jobs=-1)
        case Searchers.GRID:
            searcher = GridSearchCV(pipeline, search_attributes, cv=3, n_jobs=-1, scoring='accuracy')
        case _:
            raise ValueError("Searcher wasn't a handled type")

    searcher.fit(features, labels)
    print(searcher.best_params_)
    print(searcher.best_score_)
    pd.DataFrame.from_dict(searcher.cv_results_).to_csv(f"{training_tracks[0].source}_grid_search.csv")
    return searcher

def get_optimal_estimators(list_of_steps:list, training_tracks, search_attributes: dict | list, search_type:Searchers, number_to_take = 20, use_oversampler = False, feature_names:list = None):

    pipeline = Pipeline(list_of_steps, memory="cache")

    best_pipeline = pipeline_search(pipeline, training_tracks, search_attributes, search_type, use_oversampler, feature_names)
    dataframe = pd.DataFrame.from_dict(best_pipeline.cv_results_)

    dataframe = dataframe.sort_values(['rank_test_score'], axis=0)
    dataframe = dataframe.head(number_to_take)
    best_parameters = dataframe['params']
    print("length", len(best_parameters))
    best_estimators = []
    for i in range(len(best_parameters)):
        parameters = best_parameters.take([i]).values[0]
        print(parameters)
        estimator = Pipeline(list_of_steps)
        estimator.set_params(**parameters)
        best_estimators.append(clone(estimator)) #I don't feel like we need a clone here, but for some reason it is; else we are appending the same estimator every time

    return best_estimators


def get_and_test_optimal_pipelines_for_every_source(tracks:list, list_of_steps:list, search_attributes:dict | list, search_type:Searchers, use_oversampler = False, feature_names:list = None):

    estimators_by_source = defaultdict(list)
    training_tracks, test_tracks = split_into_train_and_test(tracks, 0.8, seed=42, stratify=True)

    data = defaultdict(list)
    try:
        for tracks_by_source in Track.separate_tracks_by_source(training_tracks):

            best_estimators = get_optimal_estimators(list_of_steps, tracks_by_source, search_attributes, search_type, number_to_take=20, use_oversampler=use_oversampler, feature_names= feature_names)
            for best_estimator in best_estimators:
                best_estimator.set_params(**{"Classifier__SVC__estimator__probability": True})

                features, labels = Track.tracks_to_features_and_labels(tracks_by_source)
                features = Track.tracks_features_to_dataframe(tracks_by_source, feature_names)
                if use_oversampler: features, labels = RandomOverSampler(random_state = 42).fit_resample(features, labels)
                best_estimator.fit(features, labels)
                estimators_by_source[tracks_by_source[0].source].append(best_estimator)
        i = 0
        for source in estimators_by_source.keys():
            for estimator in estimators_by_source[source]:

                print(source, id(estimator))
                results = classify_by_original_track_and_get_scores(estimator, [track for track in test_tracks if track.source == source], feature_names= feature_names, plot_confusion_matrix=False, confusion_matrix_title=f"{source} {i}")
                data['source(s)'].append(source)
                data['Oversampling'].append(use_oversampler)
                data['First parameters'].append(estimator.get_params())
                data['First features'].append(estimator[:-1].get_feature_names_out())
                data['Second parameters'].append(None)
                data['Second features'].append(None)
                data['Accuracy'].append(results[0])
                data['Precision'].append(results[1])
                data['Recall'].append(results[2])
                i += 1

        combinations = []
        i = 0
        for source in estimators_by_source.keys():
            for source_peer in estimators_by_source.keys():
                if source_peer == source: continue
                if (source, source_peer) in combinations or (source_peer, source) in combinations: continue
                combinations.append((source, source_peer))
                for estimator in estimators_by_source[source]:
                    for estimator_peer in estimators_by_source[source_peer]:
                        if estimator_peer == estimator: continue
                        if (estimator, estimator_peer) in combinations or (estimator_peer, estimator) in combinations: continue
                        print(source, source_peer, id(source_peer), f"{i}/{(len(estimators_by_source[source]) * len(list(estimators_by_source.keys())) * (len(list(estimators_by_source.keys())) - 1))*10}")
                        results = classify_by_original_track_and_get_scores([estimator, estimator_peer], [[track for track in test_tracks if track.source == source], [track for track in test_tracks if track.source == source_peer]], feature_names=feature_names, plot_confusion_matrix=False, confusion_matrix_title=f"{source} + {source_peer} {i + 60}")

                        data['source(s)'].append(f"{source} + {source_peer}")
                        data['Oversampling'].append(use_oversampler)
                        data['First parameters'].append(estimator.get_params())
                        data['First features'].append(estimator[:-1].get_feature_names_out())
                        data['Second parameters'].append(estimator_peer.get_params())
                        data['Second features'].append(estimator_peer[:-1].get_feature_names_out())
                        data['Accuracy'].append(results[0])
                        data['Precision'].append(results[1])
                        data['Recall'].append(results[2])
                        i += 1
    except Exception as error:
        print("An error occured: ", error)
        traceback.print_exc()
        pd.DataFrame.from_dict(data).to_csv("Pipeline results errored.csv")

    pd.DataFrame.from_dict(data).to_csv("Pipeline results.csv")

def make_confusion_matrices(pipeline:Pipeline, csv_file, all_tracks, feature_names):
    pipeline = clone(pipeline)
    training, test = split_into_train_and_test(all_tracks, 0.8, seed=42, stratify=True)
    training_source_lists = Track.separate_tracks_by_source(training)
    training_tracks_by_source = {}
    for source_list in training_source_lists:
        training_tracks_by_source[source_list[0].source] = Track.tracks_features_to_dataframe(source_list, feature_names), Track.tracks_to_features_and_labels(source_list)[1]

    test_source_lists = Track.separate_tracks_by_source(test)
    test_tracks_by_source = {}
    for source_list in test_source_lists:
        test_tracks_by_source[source_list[0].source] = source_list

    dataframe = pd.read_csv(csv_file)
    for row in dataframe.iterrows():
        row_content = row[1]
        is_ensemble = row_content['source(s)'].count('+') != 0
        if is_ensemble:
            sources = "".join(row_content['source(s)'].replace('*', "").split())
            sources = sources.split('+')

            pipelines = [clone(pipeline), clone(pipeline)]
            pipelines[0] = pipelines[0].set_params(**eval(row_content['First parameters']))
            pipelines[1] = pipelines[1].set_params(**eval(row_content['Second parameters']))
            for i in range(len(sources)):
                source = sources[i]

                features, labels = training_tracks_by_source[source]
                pipelines[i].fit(features, labels)
                pipelines[0] = pipelines[0].set_params(**eval(row_content['First parameters']))

            scores = classify_by_original_track_and_get_scores(pipelines, [[track for track in test if track.source == sources[0]], [track for track in test if track.source == sources[1]]],
                                                      plot_confusion_matrix=True, confusion_matrix_title=str(row[0]),
                                                      feature_names=feature_names)
            print(row[0], scores)

        else:
            source = row_content['source(s)'] if row_content['source(s)'].count('*') == 0 else row_content['source(s)'].replace('*', '').replace(' ', '')
            features, labels = training_tracks_by_source[source]
            parameters = eval(row_content['First parameters'])
            pipeline = clone(pipeline).set_params(**parameters)
            pipeline = pipeline.fit(features, labels)
            print(pipeline[:-1].get_feature_names_out())
            scores = classify_by_original_track_and_get_scores(pipeline, test_tracks_by_source[source], plot_confusion_matrix=True, confusion_matrix_title=str(row[0]), feature_names = feature_names)
            print(row[0], scores)

all_tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="datasets/emotify/emotify_values.xml",
                     sources=["Mixed", "Instrumentals", "Vocals"], amount_to_take=None)

all_tracks = Track.remove_empty_tracks_and_number_removed(all_tracks)

classifier = Pipeline([("SVC", OneVsRestClassifier(SVC(cache_size=500, max_iter=1000000, probability=False, random_state=42, class_weight='balanced', decision_function_shape='ovr')))])

search_attributes = [{"Classifier__SVC__estimator__C": [1, 10, 100, 1000], "Classifier__SVC__estimator__kernel": ["linear"], "Scaler": [MinMaxScaler(), StandardScaler()], 'Selector__max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]},
                     {"Classifier__SVC__estimator__C": [10, 100, 1000, 10000], "Classifier__SVC__estimator__gamma": [0.0001, 0.001, 0.01, 0.1],"Classifier__SVC__estimator__kernel": ["rbf"], "Scaler": [MinMaxScaler(), StandardScaler()], 'Selector__max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]},
                     {"Classifier__SVC__estimator__C": [0.0001, 0.001, 0.01, 0.1], "Classifier__SVC__estimator__gamma": [0.0001, 0.001, 0.01, 0.1], "Classifier__SVC__estimator__degree": [3, 5], "Classifier__SVC__estimator__coef0": [2, 4, 6],"Classifier__SVC__estimator__kernel": ["poly"], "Scaler": [MinMaxScaler(), StandardScaler()], 'Selector__max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]}]

#steps = [("Imputer", SimpleImputer(strategy="mean")), ("Scaler", StandardScaler()), ("PCA", PCA(random_state=42)), ('Selector', SelectFromModel(LinearSVC())), ("Classifier", classifier)]
steps = [("Imputer", SimpleImputer(strategy="mean")), ("Scaler", MinMaxScaler()), ('Selector', SelectFromModel(LinearSVC(random_state=42))), ("Classifier", classifier)]

feature_names = get_feature_names("datasets/emotify/emotify_values.xml", True)

#make_confusion_matrices(Pipeline(steps=steps), "Pipeline results.csv", all_tracks, feature_names)

start_time = time.time()
get_and_test_optimal_pipelines_for_every_source(all_tracks, steps, search_attributes, Searchers.GRID, use_oversampler = False, feature_names = feature_names)
print(f"Took {time.time()-start_time}")


