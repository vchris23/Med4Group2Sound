from enum import Enum

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from music_svm import grid_param_search
from process_sound import split_into_train_and_test
from data_import import import_tracks
from our_classes import Track
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

all_tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="feature_values_1.xml",
                     sources=["Instrumental", "Mixed", "Vocals"], amount_to_take=400)

class Searchers(Enum):
    RANDOM = 0
    GRID = 1

def pipeline_search(pipeline, training_tracks, search_attributes:dict, search_type:Searchers):

    features, labels = Track.tracks_to_features_and_labels(training_tracks)

    match search_type:
        case Searchers.RANDOM:
            searcher = RandomizedSearchCV(pipeline, search_attributes, cv=5, n_jobs=-1)
        case Searchers.GRID:
            searcher = GridSearchCV(pipeline, search_attributes, cv=5, n_jobs=-1)
        case _:
            raise ValueError("Searcher wasn't a handled type")

    searcher.fit(features, labels)
    print(searcher.best_params_)

def get_optimal_pipeline(list_of_steps:list, train_tracks, test_tracks, search_attributes:dict, search_type:Searchers):

    training_features, training_labels = Track.tracks_to_features_and_labels(train_tracks)
    test_features, test_labels = Track.tracks_to_features_and_labels(test_tracks)

    pipeline = Pipeline(list_of_steps, memory="cache")

    pipeline_search(pipeline, train_tracks, search_attributes, search_type)

    #pipeline.fit(training_features, training_labels)

steps = [("Scaler", StandardScaler()), ("SVC", SVC())]
search_attributes = {"SVC__C": [1, 10, 100]}
training_tracks, test_tracks = split_into_train_and_test(all_tracks, 0.8)


get_optimal_pipeline(steps, training_tracks, test_tracks, search_attributes, Searchers.GRID)
