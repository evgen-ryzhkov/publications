import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes


def get_place_segments(df_original):
    df_place_data = df_original[['city', 'neighborhood', 'location']].copy()
    df_preprocessed = _preprocess_data(df_place_data)

    # to define number of cluster, run this function
    # get_number_of_segments(df_preprocessed)
    num_clusters = 5
    place_cluster_column_name = 'cluster_place'
    df_original_place_segmented = get_segments(df_place_data, df_preprocessed, place_cluster_column_name, num_clusters)
    f_validation, df_stat = validate_cluster_sizes(df_original_place_segmented, place_cluster_column_name)

    _profile_clusters(df_original_place_segmented, df_stat, place_cluster_column_name, num_clusters)


def _preprocess_data(df):
    print('[INFO] Place data preprocessing started...')
    df_custom_features = _create_custom_features(df)

    # choosing meaningful features for segmentation
    '''
        - cat_city/suburbs - different lifestyle
        - loc_[different location attributes] - lifestyle
    '''
    df_for_segmentation = df_custom_features[['cat_city', 'cat_suburbs', 'loc_quiet_road', 'loc_busy_road',
                                              'loc_in_center', 'loc_res_district', 'loc_near_park',
                                              'loc_near_forest', 'loc_near_water', 'loc_rural']]
    print('[OK] Place data preprocessing finished.')
    return df_for_segmentation


def _create_custom_features(df):
    # in the city or suburbs
    df.loc[df['city'] == 'Amsterdam', 'cat_city'] = 1
    df.loc[df['city'] != 'Amsterdam', 'cat_suburbs'] = 1

    # get unique values for feature analysis
    # excluding nan
    loc_values_list =df['location'].loc[pd.notna(df['location'])].unique().tolist()

    # join into one string in order to convert back into list but in our conditions
    separator = ', '
    loc_values_string = separator.join(loc_values_list)
    loc_values_string_without_and = loc_values_string.replace(' and', ',')

    # convert string into list and get unique values
    loc_values_list_2 = loc_values_string_without_and.split(', ')
    loc_values_numpy = np.array(loc_values_list_2)

    # custom creating onehot encoding for location
    df.loc[df['location'].str.contains(pat='quiet road', na=False, case=False), 'loc_quiet_road'] = 1
    df.loc[df['location'].str.contains(pat='busy road', na=False, case=False), 'loc_busy_road'] = 1
    df.loc[df['location'].str.contains(pat='in center', na=False, case=False), 'loc_in_center'] = 1
    df.loc[df['location'].str.contains(pat='residential district', na=False, case=False), 'loc_res_district'] = 1
    df.loc[df['location'].str.contains(pat='alongside park', na=False, case=False), 'loc_near_park'] = 1
    df.loc[df['location'].str.contains(pat='forest|wooded surroundings', na=False, case=False), 'loc_near_forest'] = 1
    df.loc[df['location'].str.contains(pat='water|waterway|seaview', na=False, case=False), 'loc_near_water'] = 1
    df.loc[df['location'].str.contains(pat='rural', na=False, case=False), 'loc_rural'] = 1

    # fill nan for new custom features
    nan_columns = ['cat_city', 'cat_suburbs', 'loc_quiet_road', 'loc_busy_road', 'loc_in_center', 'loc_res_district', 'loc_near_park',
                   'loc_near_forest', 'loc_near_water', 'loc_rural']
    for col in nan_columns:
        df[col] = df[col].fillna(0)

    return df


def _profile_clusters(df_segmented, df_stat, cluster_col_name, n_clusters):
    # define value for calculation distribution for each column
    loc_types = ['cat_city', 'cat_suburbs',]
    loc_properties = ['loc_quiet_road', 'loc_busy_road', 'loc_in_center', 'loc_res_district', 'loc_near_park',
                   'loc_near_forest', 'loc_near_water', 'loc_rural']
    df_profiling_cols = loc_types + loc_properties

    # initiation of df_profiling
    n_columns = len(df_profiling_cols)
    init_array = np.zeros((n_clusters, n_columns))
    df_profiling = pd.DataFrame(data=init_array, columns=df_profiling_cols)

    # fill df_profiling with real values
    # value - percents each type of features in cluster
    for cluster in range(n_clusters):
        df_cluster = df_segmented.loc[df_segmented[cluster_col_name] == cluster]
        df_cluster_len = len(df_cluster)

        for col in df_profiling_cols:

            percent_val = round((len(df_cluster.loc[df_cluster[col] == 1]) / df_cluster_len) * 100)
            df_profiling.loc[cluster, col] = percent_val

    # add distribution values for each cluster
    df_profiling = pd.concat([df_stat, df_profiling], axis=1, sort=False)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # full screen mode

    sns.heatmap(df_profiling, annot=True, fmt="g")
    plt.show()
