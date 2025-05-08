import time
from collections import defaultdict
from enum import Enum
from random import random
import traceback

from imblearn.over_sampling import RandomOverSampler
from pandas.core.common import random_state
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from music_svm import grid_param_search, get_scores, classify_and_get_scores
from process_sound import split_into_train_and_test
from data_import import import_tracks, get_feature_names
from our_classes import Track
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import warnings
warnings.filterwarnings('always')

all_tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="datasets/emotify/emotify_values.xml",
                     sources=["Mixed", "Instrumentals", "Vocals"], amount_to_take=None)
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

def get_optimal_estimators(list_of_steps:list, training_tracks, search_attributes: dict | list, search_type:Searchers, number_to_take = 10, use_oversampler = False, feature_names:list = None):

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
        estimator = Pipeline(list_of_steps)
        estimator.set_params(**parameters)
        print("CLassifier", parameters["Classifier__SVC__C"])
        best_estimators.append(clone(estimator)) #I don't feel like we need a clone here, but for some reason it is; else we are appending the same estimator every time

    return best_estimators


def get_and_test_optimal_pipelines_for_every_source(tracks:list, list_of_steps:list, search_attributes:dict | list, search_type:Searchers, use_oversampler = False, feature_names:list = None):

    estimators_by_source = defaultdict(list)
    training_tracks, test_tracks = split_into_train_and_test(tracks, 0.8, seed=42)
    data = defaultdict(list)
    try:
        for tracks_by_source in Track.separate_tracks_by_source(training_tracks):

            best_estimators = get_optimal_estimators(list_of_steps, tracks_by_source, search_attributes, search_type, number_to_take=10, use_oversampler=use_oversampler, feature_names= feature_names)
            for best_estimator in best_estimators:
                best_estimator.set_params(**{"Classifier__SVC__probability": True})

                features, labels = Track.tracks_to_features_and_labels(tracks_by_source)
                features = Track.tracks_features_to_dataframe(tracks_by_source, feature_names)
                if use_oversampler: features, labels = RandomOverSampler(random_state = 42).fit_resample(features, labels)
                best_estimator.fit(features, labels)
                estimators_by_source[tracks_by_source[0].source].append(best_estimator)

        for source in estimators_by_source.keys():
            for estimator in estimators_by_source[source]:
                print(source, id(estimator))
                results = classify_and_get_scores(estimator, [track for track in test_tracks if track.source == source], feature_names= feature_names)
                data['source(s)'].append(source)
                data['Oversampling'].append(use_oversampler)
                data['First parameters'].append(estimator.get_params())
                data['First features'].append(estimator[:-1].get_feature_names_out())
                data['Second parameters'].append(None)
                data['Second features'].append(None)
                data['Accuracy'].append(results[0])
                data['Precision'].append(results[1])
                data['Recall'].append(results[2])

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
                        i += 1
                        print(source, source_peer, id(source_peer), f"{i}/{len(estimators_by_source[source]) * len(list(estimators_by_source.keys())) * (len(list(estimators_by_source.keys())) - 1)}")
                        results = classify_and_get_scores([estimator, estimator_peer], [[track for track in test_tracks if track.source == source], [track for track in test_tracks if track.source == source_peer]], feature_names=feature_names)

                        data['source(s)'].append(f"{source} + {source_peer}")
                        data['Oversampling'].append(use_oversampler)
                        data['First parameters'].append(estimator.get_params())
                        data['First features'].append(estimator[:-1].get_feature_names_out())
                        data['Second parameters'].append(estimator_peer.get_params())
                        data['Second features'].append(estimator_peer[:-1].get_feature_names_out())
                        data['Accuracy'].append(results[0])
                        data['Precision'].append(results[1])
                        data['Recall'].append(results[2])
    except Exception as error:
        print("An error occured: ", error)
        traceback.print_exc()
        pd.DataFrame.from_dict(data).to_csv("Pipeline results errored.csv")

    pd.DataFrame.from_dict(data).to_csv("Pipeline results.csv")


classifier = Pipeline([("SVC", SVC(cache_size=500, max_iter=1000000, probability=False, random_state=42, class_weight='balanced'))])
classifier_two = Pipeline([("SVC", SVC(cache_size=500, max_iter=1000000, probability=False, random_state=40, class_weight='balanced'))])

search_attributes = [{"Selector__max_features": [4, 8, 12, 16], "Classifier__SVC__C": [1, 10, 100, 1000], "Classifier__SVC__kernel": ["linear"], "Scaler": [MinMaxScaler()], "PCA__n_components": [20, 30, 40]},
                     {"Selector__max_features": [4, 8, 12, 13], "Classifier__SVC__C": [1, 10, 100, 1000], "Classifier__SVC__kernel": ["linear"], "Scaler": [MinMaxScaler()], "PCA__n_components": [15]},
                     {"Selector__max_features": [4, 8,], "Classifier__SVC__C": [1, 10, 100, 1000], "Classifier__SVC__kernel": ["linear"], "Scaler": [MinMaxScaler()], "PCA__n_components": [10]},
                     {"Selector__max_features": [4, 8, 12, 16, 20], "Classifier__SVC__C": [10, 100, 1000], "Classifier__SVC__gamma": [0.001, 0.01, 0.1],"Classifier__SVC__kernel": ["rbf"], "Scaler": [MinMaxScaler()], "PCA__n_components": [30, 40]},
                     {"Selector__max_features": [4, 8, 12], "Classifier__SVC__C": [10, 100, 1000], "Classifier__SVC__gamma": [0.001, 0.01, 0.1],"Classifier__SVC__kernel": ["rbf"], "Scaler": [MinMaxScaler()], "PCA__n_components": [15]},
                     {"Selector__max_features": [4, 8], "Classifier__SVC__C": [10, 100, 1000], "Classifier__SVC__gamma": [0.001, 0.01, 0.1],"Classifier__SVC__kernel": ["rbf"], "Scaler": [MinMaxScaler()], "PCA__n_components": [10]},
                     {"Selector__max_features": [4, 8, 12, 16, 20], "Classifier__SVC__C": [0.001, 0.01, 0.1], "Classifier__SVC__gamma": [0.001, 0.01], "Classifier__SVC__degree": [3, 5], "Classifier__SVC__coef0": [2, 4, 6],"Classifier__SVC__kernel": ["poly"], "Scaler": [MinMaxScaler()], "PCA__n_components": [30, 40]},
                     {"Selector__max_features": [4, 8, 12], "Classifier__SVC__C": [0.001, 0.01, 0.1], "Classifier__SVC__gamma": [0.001, 0.01], "Classifier__SVC__degree": [3, 5], "Classifier__SVC__coef0": [2, 4, 6],"Classifier__SVC__kernel": ["poly"], "Scaler": [MinMaxScaler()], "PCA__n_components": [15]},
                     {"Selector__max_features": [4, 8], "Classifier__SVC__C": [0.001, 0.01, 0.1], "Classifier__SVC__gamma": [0.001, 0.01], "Classifier__SVC__degree": [3, 5], "Classifier__SVC__coef0": [2, 4, 6],"Classifier__SVC__kernel": ["poly"], "Scaler": [MinMaxScaler()], "PCA__n_components": [10]}]

search_attributes = [{"Classifier__SVC__C": [1, 10, 100, 1000], "Classifier__SVC__kernel": ["linear"], "Scaler": [MinMaxScaler()], 'Selector__max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]},
                     {"Classifier__SVC__C": [10, 100, 1000], "Classifier__SVC__gamma": [0.001, 0.01, 0.1],"Classifier__SVC__kernel": ["rbf"], "Scaler": [MinMaxScaler()], 'Selector__max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]},
                     {"Classifier__SVC__C": [0.001, 0.01, 0.1], "Classifier__SVC__gamma": [0.001, 0.01], "Classifier__SVC__degree": [3, 5], "Classifier__SVC__coef0": [2, 4, 6],"Classifier__SVC__kernel": ["poly"], "Scaler": [MinMaxScaler()], 'Selector__max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]}]

steps = [("Imputer", SimpleImputer(strategy="mean")), ("Scaler", StandardScaler()), ("PCA", PCA(random_state=42)), ('Selector', SelectFromModel(LinearSVC())), ("Classifier", classifier)]
steps = [("Imputer", SimpleImputer(strategy="mean")), ("Scaler", StandardScaler()), ('Selector', SelectFromModel(LinearSVC())), ("Classifier", classifier)]

feature_names = get_feature_names("datasets/emotify/emotify_values.xml", True)

start_time = time.time()
get_and_test_optimal_pipelines_for_every_source(all_tracks, steps, search_attributes, Searchers.GRID, use_oversampler = False, feature_names = feature_names)
print(f"Took {time.time()-start_time}")


