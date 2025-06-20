import os.path
import time
from collections import defaultdict
from enum import Enum
import traceback
import numpy as np
import pandas.core.series
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from copy import copy
from matplotlib import pyplot as plt
from CAL_data_sound_pros import get_cal_tracks
from dataset_import import make_tracks_list
from music_svm import classify_by_original_track_and_get_scores
from process_sound import split_into_train_and_test
from data_import import import_tracks, get_feature_names
from our_classes import Track
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import warnings
from MERGEdata_import import filling_track_list

path_to_ss_clips = 'datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions/'
categories = 'datasets/MIREX-like_mood/categories.txt'
clusters = 'datasets/MIREX-like_mood/clusters.txt'
trackDict = {
        "Emotify": lambda : import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="datasets/emotify/emotify_values.xml", sources=["Mixed", "Instrumentals", "Vocals"], amount_to_take=None),
        "Cal500" : lambda : get_cal_tracks("datasets/New dataset/new_annotated.txt", "datasets/New dataset/Clips", "datasets/New dataset/feature_values_1.xml"),
        "Merge" : lambda : filling_track_list('MERGE-datas/AllSongs15Sec', 'MERGE-datas/feature_values_1.xml', None),
        "MirexLike_Categories" : lambda : make_tracks_list(path_to_ss_clips, 100000000, path_to_categories=categories, path_to_clusters=clusters, using_clusters_instead_of_categories=False),
        "MirexLike_Cluster" : lambda : make_tracks_list(path_to_ss_clips, 100000000, path_to_categories=categories, path_to_clusters=clusters, using_clusters_instead_of_categories=True),
        }

warnings.filterwarnings('always')


class Searchers(Enum):
    RANDOM = 0
    GRID = 1

def pipeline_search(pipeline, training_tracks, search_attributes:dict | list, search_type:Searchers, use_oversampler = False, feature_names:list = None):

    feature_names = get_feature_names("feature_values_1.xml", True) if feature_names is None else feature_names

    features, labels = Track.tracks_to_features_and_labels(training_tracks)
    features = Track.tracks_features_to_dataframe(training_tracks, feature_names)
    if use_oversampler:
        features, labels = RandomOverSampler(random_state=rng).fit_resample(features, labels)

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
        estimator = Pipeline(list_of_steps)
        estimator.set_params(**parameters)
        best_estimators.append(clone(estimator)) #I don't feel like we need a clone here, but for some reason it is; else we are appending the same estimator every time

    return best_estimators

def make_parameters_saveable(parameters:dict, seed:int = 42):
    new_parameters = copy(parameters)
    new_parameters['Selector'].estimator.set_params(**{'random_state': seed})
    new_parameters['steps'][-1][1][-1].estimator.set_params(**{'random_state': seed})
    new_parameters['steps'][-2][1].estimator.set_params(**{'random_state': seed})
    new_parameters['Classifier'][-1].estimator.set_params(**{'random_state': seed})
    new_parameters['Classifier__steps'][0][1].estimator.set_params(**{'random_state': seed})
    new_parameters['Classifier__SVC'].estimator.set_params(**{'random_state': seed})
    new_parameters['Classifier__SVC__estimator'].set_params(**{'random_state': seed})
    new_parameters['Classifier__SVC__estimator__random_state'] = seed
    new_parameters['Selector__estimator'].set_params(**{'random_state': seed})
    new_parameters['Selector__estimator__random_state'] = seed
    return new_parameters

def get_and_test_optimal_pipelines_for_every_source(tracks:list, list_of_steps:list, search_attributes:dict | list, search_type:Searchers, use_oversampler = False, feature_names:list = None, pipeline_results_file_name = "pipeline results", seed:int = 42):

    estimators_by_source = defaultdict(list)
    training_tracks, test_tracks = split_into_train_and_test(tracks, 0.8, seed=seed, stratify=True)

    data = defaultdict(list)
    try:
        for tracks_by_source in Track.separate_tracks_by_source(training_tracks):

            best_estimators = get_optimal_estimators(list_of_steps, tracks_by_source, search_attributes, search_type, number_to_take=30, use_oversampler=use_oversampler, feature_names= feature_names)
            for best_estimator in best_estimators:
                best_estimator.set_params(**{"Classifier__SVC__estimator__probability": True})

                features, labels = Track.tracks_to_features_and_labels(tracks_by_source)
                features = Track.tracks_features_to_dataframe(tracks_by_source, feature_names)
                if use_oversampler: features, labels = RandomOverSampler(random_state = rng).fit_resample(features, labels)
                best_estimator.fit(features, labels)
                estimators_by_source[tracks_by_source[0].source].append(best_estimator)
        i = 0
        for source in estimators_by_source.keys():
            for estimator in estimators_by_source[source]:

                print(source, id(estimator))
                results = classify_by_original_track_and_get_scores(estimator, [track for track in test_tracks if track.source == source], feature_names= feature_names, plot_confusion_matrix=False, confusion_matrix_title=f"{source} {i}")
                data['source(s)'].append(source)
                data['Oversampling'].append(use_oversampler)
                data['First parameters'].append(make_parameters_saveable(estimator.get_params(), seed))
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
                #if source_peer == source: continue
                if (source, source_peer) in combinations or (source_peer, source) in combinations: continue
                combinations.append((source, source_peer))
                for estimator in estimators_by_source[source]:
                    for estimator_peer in estimators_by_source[source_peer]:
                        if estimator_peer == estimator:
                            print("Skipped as estimator peer was estimator")
                            continue
                        if (estimator, estimator_peer) in combinations or (estimator_peer, estimator) in combinations:
                            print("Skipped")
                            continue
                        print(source, source_peer, id(source_peer), f"{i}/{(len(estimators_by_source[source]) * len(list(estimators_by_source.keys())) * (len(list(estimators_by_source.keys())) - 1))*10}")
                        results = classify_by_original_track_and_get_scores([estimator, estimator_peer], [[track for track in test_tracks if track.source == source], [track for track in test_tracks if track.source == source_peer]], feature_names=feature_names)

                        data['source(s)'].append(f"{source} + {source_peer}")
                        data['Oversampling'].append(use_oversampler)
                        data['First parameters'].append(make_parameters_saveable(estimator.get_params(), seed))
                        data['First features'].append(estimator[:-1].get_feature_names_out())
                        data['Second parameters'].append(make_parameters_saveable(estimator_peer.get_params(), seed))
                        data['Second features'].append(estimator_peer[:-1].get_feature_names_out())
                        data['Accuracy'].append(results[0])
                        data['Precision'].append(results[1])
                        data['Recall'].append(results[2])
                        i += 1
    except Exception as error:
        print("An error occured: ", error)
        traceback.print_exc()
        pd.DataFrame.from_dict(data).to_csv("Pipeline results errored.csv")

    pd.DataFrame.from_dict(data).to_csv(pipeline_results_file_name + ".csv")

def process_datasets(track_dictionary:dict, list_of_steps:list, search_attributes:dict | list, feature_names:list = None, dictionary_keys_list:list = None, filename_append = "", seed:int = 42):
    if dictionary_keys_list is None:
        dictionary_keys_list = track_dictionary.keys()
    for key in dictionary_keys_list:
        filename = f"{key}_results_{filename_append}"
        if os.path.exists(filename): return
        dataset_tracks = track_dictionary[key].__call__()
        dataset_tracks = Track.remove_empty_tracks_and_number_removed(dataset_tracks)
        get_and_test_optimal_pipelines_for_every_source(dataset_tracks, list_of_steps, search_attributes, search_type=Searchers.GRID, feature_names=feature_names, pipeline_results_file_name=filename, seed = seed)

def big_test(test_range:list, track_dictionary:dict, dictionary_keys_list:list = None):
    for i in test_range:
        rng = np.random.RandomState(i+1)

        classifier = Pipeline([("SVC", OneVsRestClassifier(
            SVC(cache_size=500, max_iter=1000000, probability=False, random_state=rng, class_weight='balanced',
                decision_function_shape='ovr')))])

        search_attributes = [
            {"Classifier__SVC__estimator__C": [0.01, 0.1, 1, 10, 100], "Classifier__SVC__estimator__kernel": ["linear"],
             "Scaler": [MinMaxScaler(), StandardScaler()], 'Selector__max_features': [4, 8, 12, 16, 20, 24, 28, 32]},
            {"Classifier__SVC__estimator__C": [0.01, 0.1, 1, 10, 100],
             "Classifier__SVC__estimator__gamma": [0.0001, 0.001, 0.01, 0.1],
             "Classifier__SVC__estimator__kernel": ["rbf"], "Scaler": [MinMaxScaler(), StandardScaler()],
             'Selector__max_features': [4, 8, 12, 16, 20, 24, 28, 32]},
            {"Classifier__SVC__estimator__C": [0.01, 0.1, 1, 10, 100],
             "Classifier__SVC__estimator__gamma": [0.0001, 0.001, 0.01, 0.1],
             "Classifier__SVC__estimator__degree": [2, 4, 6, 8], "Classifier__SVC__estimator__coef0": [2, 4, 6, 8],
             "Classifier__SVC__estimator__kernel": ["poly"], "Scaler": [MinMaxScaler(), StandardScaler()],
             'Selector__max_features': [4, 8, 12, 16, 20, 24, 28, 32]}]

        steps = [("Imputer", SimpleImputer(strategy="mean")), ("Scaler", StandardScaler),
                 ('Selector', SelectFromModel(LinearSVC(random_state=rng))), ("Classifier", classifier)]

        feature_names = get_feature_names("datasets/emotify/emotify_values.xml", True)

        process_datasets(track_dictionary, steps, search_attributes, feature_names, dictionary_keys_list, filename_append=f"{i+1}", seed=i+1)

def make_confusion_matrices(pipeline:Pipeline, csv_file, all_tracks, feature_names, dataset:str = ''):
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
            pipelines[0] = pipelines[0].set_params(**eval(row_content['First parameters'].replace(': nan', ': np.nan')))
            pipelines[1] = pipelines[1].set_params(**eval(row_content['Second parameters'].replace(': nan', ': np.nan')))
            for i in range(len(sources)):
                source = sources[i]
                features, labels = training_tracks_by_source[source]
                pipelines[i].fit(features, labels)
            scores = classify_by_original_track_and_get_scores(pipelines, [[track for track in test if track.source == sources[0]], [track for track in test if track.source == sources[1]]],
                                                      plot_confusion_matrix=True, confusion_matrix_title=f"{str(row[1]['source(s)'].replace('*', ''))}",
                                                      feature_names=feature_names)
            print(row[1]['source(s)'], scores)

        else:
            source = row_content['source(s)'] if row_content['source(s)'].count('*') == 0 else row_content['source(s)'].replace('*', '').replace(' ', '')
            features, labels = training_tracks_by_source[source]
            parameters = eval(row_content['First parameters'].replace(': nan', ': np.nan'))
            new_pipeline = clone(pipeline).set_params(**parameters)
            new_pipeline = new_pipeline.fit(features, labels)
            scores = classify_by_original_track_and_get_scores(new_pipeline, test_tracks_by_source[source], plot_confusion_matrix=True, confusion_matrix_title=f"{str(row[1]['source(s)'].replace('*', ''))}", feature_names = feature_names)
            print(row[1]['source(s)'], scores)

def retest(csv_file, dataset_dict, seeds:list):

    result_dict = {"dataset": [], "sources": [], "accuracy": []}

    dataframe = pd.read_csv(csv_file)
    for group in dataframe.groupby("dataset"):

        all_tracks = Track.remove_empty_tracks_and_number_removed(dataset_dict[group[0]]())

        for seed in seeds:

            rng = np.random.RandomState(seed)

            classifier = Pipeline([("SVC", OneVsRestClassifier(
                SVC(cache_size=500, max_iter=1000000, probability=False, random_state=rng, class_weight='balanced',
                    decision_function_shape='ovr')))])

            pipeline = Pipeline([("Imputer", SimpleImputer(strategy="mean")), ("Scaler", StandardScaler),
                        ('Selector', SelectFromModel(LinearSVC(random_state=rng))), ("Classifier", classifier)])

            feature_names = get_feature_names("datasets/emotify/emotify_values.xml", True)

            training, test = split_into_train_and_test(all_tracks, 0.8, seed=seed, stratify=True)
            training_source_lists = Track.separate_tracks_by_source(training)
            training_tracks_by_source = {}
            for source_list in training_source_lists:
                training_tracks_by_source[source_list[0].source] = Track.tracks_features_to_dataframe(source_list, feature_names), Track.tracks_to_features_and_labels(source_list)[1]

            test_source_lists = Track.separate_tracks_by_source(test)
            print(test_source_lists)
            test_tracks_by_source = {}
            for source_list in test_source_lists:
                test_tracks_by_source[source_list[0].source] = source_list

            dfgroup = group[1]
            for row in dfgroup.iterrows():
                row_content = row[1]
                is_ensemble = row_content['sources'].count('+') != 0
                if is_ensemble:
                    source = row_content['sources'].replace('*', "")
                    sources = "".join(source.split())
                    sources = sources.split('+')

                    pipelines = [clone(pipeline), clone(pipeline)]
                    pipelines[0] = pipelines[0].set_params(**eval(row_content['First parameters'].replace(': nan', ': np.nan')))
                    pipelines[1] = pipelines[1].set_params(**eval(row_content['Second parameters'].replace(': nan', ': np.nan')))
                    for i in range(len(sources)):
                        sourcee = sources[i]
                        features, labels = training_tracks_by_source[sourcee]
                        pipelines[i].fit(features, labels)
                    scores = classify_by_original_track_and_get_scores(pipelines, [[track for track in test if track.source == sources[0]],
                                                                                       [track for track in test if track.source == sources[1]]], feature_names=feature_names)
                    print(row[1]['sources'], scores)

                else:
                    source = row_content['sources'] if row_content['sources'].count('*') == 0 else row_content['sources'].replace('*', '').replace(' ', '')
                    features, labels = training_tracks_by_source[source]
                    parameters = eval(row_content['First parameters'].replace(': nan', ': np.nan'))
                    new_pipeline = clone(pipeline).set_params(**parameters)
                    new_pipeline = new_pipeline.fit(features, labels)
                    print(test_tracks_by_source.keys())
                    scores = classify_by_original_track_and_get_scores(new_pipeline, test_tracks_by_source[source], feature_names = feature_names)
                    print(row[1]['sources'], scores)

                result_dict["dataset"].append(group[0])
                result_dict["sources"].append(source)
                result_dict["accuracy"].append(scores[0])

    pd.DataFrame.from_dict(result_dict).to_csv("Repeat_results.csv", index=False)

def fetch_best_results(dataset_results:dict):

    collected_best ={"dataset": [], "sources": [], "accuracy": [], "first_parameters": [], "second_parameters": []}

    for key in dataset_results.keys():
        results_list = dataset_results[key]
        for path in results_list:
            df:pd.DataFrame = pd.read_csv(path)
            for group in df.groupby("source(s)"):
                df_group = group[1]
                #df_group = df_group.sort_values(["Accuracy", "Precision", "Recall"], ascending=False)
                bests = df_group.head(2)
                collected_best["dataset"].append(key)
                collected_best["dataset"].append(key)
                collected_best["sources"].append(bests["source(s)"].values[0])
                collected_best["accuracy"].append(bests["Accuracy"].values[0])
                collected_best["first_parameters"].append(bests["First parameters"].values[0])
                collected_best["first_parameters"].append(bests["First parameters"].values[1])
                collected_best["second_parameters"].append(bests["Second parameters"].values[0])
                collected_best["second_parameters"].append(bests["Second parameters"].values[1])
                collected_best["sources"].append(bests["source(s)"].values[1])
                collected_best["accuracy"].append(bests["Accuracy"].values[1])

    bests_df = pd.DataFrame.from_dict(collected_best)
    index_of_mixed = bests_df.index[bests_df["sources"]=="Mixed"][0]
    mixed_accuracy = bests_df.take([index_of_mixed]).values[0][2]
    bests_df.to_csv("Best_results.csv", index=False)

big_test([10, 20], trackDict)

results = {"Emotify": ["emotify_tracks_results_.csv"]}
fetch_best_results(results)

retest("Best_results.csv", trackDict, [42])

#path_to_ss_clips = 'datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions/'
#categories = 'datasets/MIREX-like_mood/categories.txt'
#clusters = 'datasets/MIREX-like_mood/clusters.txt'

#all_tracks = get_cal_tracks("datasets/New dataset/new_annotated.txt", "datasets/New dataset/Clips", "datasets/New dataset/feature_values_1.xml")

#Track.graph_energy_in_tracks([track for track in all_tracks if track.source == 'Vocals'], "Energy in vocal clips for Cal500 before pruning")

#all_tracks = Track.remove_empty_tracks_and_number_removed(all_tracks)

#Track.graph_energy_in_tracks([track for track in all_tracks if track.source == 'Vocals'], "Energy in vocal clips for Cal500 after pruning")


#make_confusion_matrices(Pipeline(steps=steps), "Cal500 bests.csv", all_tracks, feature_names)

#df = pd.read_csv("Energy distribution to results.csv")
#df.corr(numeric_only = True).to_csv("Correlation of energy distributions.csv")

"""path_to_ss_clips = 'datasets/MIREX-like_mood/SS_and_clipped_audio/Separated_and_mixed_versions/'
categories = 'datasets/MIREX-like_mood/categories.txt'
clusters = 'datasets/MIREX-like_mood/clusters.txt'
#here
#cal_tracks = get_cal_tracks("datasets/New dataset/new_annotated.txt", "datasets/New dataset/Clips", "datasets/New dataset/feature_values_1.xml")
#merge_tracks = filling_track_list('MERGE-datas/AllSongs15Sec', 'MERGE-datas/feature_values_1.xml', None)
mirex_cluster_tracks = make_tracks_list(path_to_ss_clips, 100000000, path_to_categories=categories, path_to_clusters=clusters, using_clusters_instead_of_categories=True)
#mirex_category_tracks = make_tracks_list(path_to_ss_clips, 100000000, path_to_categories=categories, path_to_clusters=clusters, using_clusters_instead_of_categories=False)
#emotify_tracks = import_tracks("datasets/emotify/clips", "datasets/emotify/emotify_data.csv", features_xml_path="datasets/emotify/emotify_values.xml",
#                     sources=["Mixed", "Instrumentals", "Vocals"], amount_to_take=None)

#'Cal500': cal_tracks, 'Merge': merge_tracks, 'Mirex_Cluster': mirex_cluster_tracks, 'Mirex_Categories': mirex_category_tracks,
# 'Emotify': emotify_tracks
datasets = {'MIREX-like': mirex_cluster_tracks}
bests = {'CAL500': 'Cal500 bests.csv', 'MERGE': 'Merge bests.csv', 'Mirex_Cluster': 'Mixed cluster bests.csv', 'Mirex_Categories': 'Mirex category bests.csv', 'Emotify': 'Emotify bests.csv'}

for dataset_name in datasets.keys():
    print(dataset_name)
    plt.title(dataset_name)
    plt.show()

    print("Before pruning:\n")
    tracks = datasets[dataset_name]
    Track.get_class_balance(tracks)
    mixed = [track for track in tracks if track.source == "Mixed"]
    vocals = [track for track in tracks if track.source == "Vocals"]
    instrumentals = [track for track in tracks if track.source == "Instrumental"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #Track.graph_energy_in_tracks(vocals, f"Vocal energy for {dataset_name} before\n removing songs with low energy source parts", 0.5, False, ax)
    #Track.graph_energy_in_tracks(mixed, f"Combined energy for {dataset_name} before\n removing songs with low energy source parts", 0.5, False, ax)
    #Track.graph_energy_in_tracks(instrumentals, f"Instrumental energy for {dataset_name} before\n removing songs with low energy source parts", 0.4, False, ax)
    #plt.title(f"Energy distribution of {dataset_name} vocal clips")

    print("After pruning:\n")

    tracks = Track.remove_empty_tracks_and_number_removed(tracks)
    Track.get_class_balance(tracks)
    mixed = [track for track in tracks if track.source == "Mixed"]
    vocals = [track for track in tracks if track.source == "Vocals"]
    instrumentals = [track for track in tracks if track.source == "Instrumental"]

    Track.graph_energy_in_tracks(vocals, f"Vocal energy for {dataset_name} after\n removing songs with low energy source parts", 0.4, False, ax, color='b')
    Track.graph_energy_in_tracks(mixed, f"Combined energy for {dataset_name} after\n removing songs with low energy source parts", 0.5, False, ax, color='g')
    plt.title(f"Energy distribution of vocal and mixed clips in {dataset_name}")
    plt.legend(["Vocal clips", "Mixed clips"])
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)


    Track.graph_energy_in_tracks(instrumentals, f"Instrumental energy for {dataset_name} after\n removing songs with low energy source parts", 0.4, False, ax, color='r')
    Track.graph_energy_in_tracks(mixed,f"Combined energy for {dataset_name} after\n removing songs with low energy source parts",0.5, False, ax, color='g')

    plt.title(f"Energy distribution of instrumental and mixed clips in {dataset_name}")
    plt.legend(["Instrumental clips", "Mixed clips"])
    fig.show()

    #make_confusion_matrices(Pipeline(steps=steps), bests[dataset_name], tracks, feature_names, dataset=dataset_name)

    print("\n \n")



start_time = time.time()
#get_and_test_optimal_pipelines_for_every_source(all_tracks, steps, search_attributes, Searchers.GRID, use_oversampler = False, feature_names = feature_names)
print(f"Took {time.time()-start_time}")"""


