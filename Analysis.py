from copy import deepcopy
from random import random

import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.masked import cumsum

from our_classes import Track
import pandas as pd


def _get_best_dimensionality(tracks:list, ratio:float):

    features, _ = Track.tracks_to_labels_and_features(tracks)
    analyser = PCA(random_state=42)
    analyser.fit(features)
    culm = np.cumsum(analyser.explained_variance_ratio_)
    best_dimensional_count = np.argmax(culm >= ratio)

    return best_dimensional_count

def get_fitted_pca(tracks:list, dimensions_to_get:int):

    features, _ = Track.tracks_to_labels_and_features(tracks)
    pca = PCA(random_state=42, n_components=dimensions_to_get)
    pca.fit(features)

    return pca

def save_correlation_of_pca_and_og_features(principal_components, original_features, feature_names, file_name):

    pca_names = [f"Principal component {i}" for i in range(len(principal_components[0]))]

    pc = pd.DataFrame(principal_components, columns = pca_names)
    of = pd.DataFrame(original_features, columns = feature_names)
    pd.concat([pc, of], axis=1).corr().to_csv(f"{file_name}.csv")


def get_correlation_of_pca_and_og_features(principal_components, original_features, feature_names):

    pca_names = [f"Principal component {i}" for i in range(len(principal_components[0]))]

    pc = pd.DataFrame(principal_components, columns = pca_names)
    of = pd.DataFrame(original_features, columns = feature_names)
    return pd.concat([pc, of], axis=1).corr()

def get_pca_tracks_and_correlation(training_tracks:list, test_tracks:list, feature_names, save_file_name:str, dimensions_to_reduce_down_to:int = None, variance_ratio:float = 0.95, pca:PCA = None):
    """If dimensions_to_reduce_down_to is None, then the best dimensionality will be calculated using the variance ratio \n
        The save file name should not include the .csv file ending"""
    if pca is None:
        dimension_target = _get_best_dimensionality(training_tracks, variance_ratio) if dimensions_to_reduce_down_to is None else dimensions_to_reduce_down_to
        pca = get_fitted_pca(training_tracks, dimension_target)

    new_training_tracks = transform_features_of_tracks(training_tracks, pca)
    new_test_tracks = transform_features_of_tracks(test_tracks, pca)

    training_features = [track.features for track in training_tracks]
    pca_features = [track.features for track in new_training_tracks]

    print(f"For {new_training_tracks[0].source} PCA reduced dimensions by {len(training_features[0]) - len(pca_features[0])}, that's {round(len(pca_features[0])/len(training_features[0])*100)}%")

    save_correlation_of_pca_and_og_features(pca_features, training_features, feature_names, save_file_name)

    interpret_principal_components(pca_features, training_features, feature_names)

    return new_training_tracks, new_test_tracks




def get_LDA(tracks:list, dimensions_to_get):
    features, _ = Track.tracks_to_labels_and_features(tracks)

    lda = LinearDiscriminantAnalysis(n_components=dimensions_to_get)
    lda.fit(features)

    return lda

def transform_features_of_tracks(tracks:list, transformer):
    new_tracks = deepcopy(tracks)

    features, _ = Track.tracks_to_labels_and_features(new_tracks)
    transformed_features = transformer.transform(features)

    for i in range(len(transformed_features)):
        new_tracks[i].features = transformed_features[i]


    return new_tracks

def interpret_principal_components(principal_components, original_features, feature_names, significant_value = 0.5):

    correlation_dataframe = get_correlation_of_pca_and_og_features(principal_components, original_features, feature_names)

    just_pca = correlation_dataframe.take(range(len(principal_components[0])), 1)
    keys = just_pca.keys()

    pca_strong_correlations = {}
    for key in keys:
        correlating_features = just_pca[key].keys()
        correlation_strengths = just_pca[key].values
        significant_correlations = {}
        for i in range(len(correlating_features)):
            if abs(correlation_strengths[i]) >= significant_value and correlating_features[i] != key:
                significant_correlations[correlating_features[i]] = correlation_strengths[i]

        pca_strong_correlations[key] = significant_correlations