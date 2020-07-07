import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes
from settings.secrets import API_KEY
import requests
import re


def get_place_segments(df_original):

    df_place_data = df_original[['city', 'neighborhood', 'location']].copy()

    # more detailed data about neighborhoods
    neighborhood_csv_path = 'data/neighborhoods/data.csv'
    df_neighborhood = _load_csv_file(neighborhood_csv_path)

    df_place_merged = pd.merge(df_place_data, df_neighborhood, how='left', left_on=['city', 'neighborhood'], right_on=['city', 'neighborhood'])

    df_preprocessed = _preprocess_data(df_place_merged)

    # to define number of cluster, run this function
    # get_number_of_segments(df_preprocessed)
    # num_clusters = 5
    # place_cluster_column_name = 'cluster_place'
    # df_original_place_segmented = get_segments(df_place_data, df_preprocessed, place_cluster_column_name, num_clusters)
    # f_validation, df_stat = validate_cluster_sizes(df_original_place_segmented, place_cluster_column_name)
    #
    # _profile_clusters(df_original_place_segmented, df_stat, place_cluster_column_name, num_clusters)


def _preprocess_data(df):
    print('[INFO] Place data preprocessing started...')

    df_numeric = _convert_text_data(df)

    df_custom_features = _create_custom_features(df)

    # choosing meaningful features for segmentation
    '''
        - cat_city/suburbs - different lifestyle
        - loc_[different location attributes] - lifestyle
    '''
    # df_for_segmentation = df_custom_features[['cat_city', 'cat_suburbs', 'loc_quiet_road', 'loc_busy_road',
    #                                           'loc_in_center', 'loc_res_district', 'loc_near_park',
    #                                           'loc_near_forest', 'loc_near_water', 'loc_rural']]
    df_for_segmentation = df_custom_features[['cat_city', 'cat_suburbs', 'loc_quiet', 'loc_nature', 'loc_walkability']]
    print('[OK] Place data preprocessing finished.')
    return df_for_segmentation


def _convert_text_data(df):
    count = 0
    df['commit_time_driving_mins'] = ''
    # convert commit time from hours & minutes format into minutes only
    for index, row in df.iterrows():
        count +=1
        hours_exist = False
        s = row['commit_time_driving']
        sub_hours = 'hour'
        sub_mins = 'mins'

        # not all rows contain hours
        try:
            hours = s[:s.index(sub_hours)]
            hours_exist = True
        except:
            hours = '0'

        # not all rows contain mins
        try:
            if hours_exist:
                mins = s[(s.index(sub_hours) + len(sub_hours)):s.index(sub_mins)]
            else:
                # if there isn't hours part, we get mins in another way
                mins = s[:s.index(sub_mins)]
        except:
            mins = '0'

        try:
            df.loc[index, 'commit_time_driving_mins'] = int(hours.replace(' ', '')) * 60 + int(re.sub('[s ]', '', mins))
        except:
            print('hours=', hours)
            print('mins=', mins)

    print(df)

    return df


def _create_custom_features(df):
    # in the city or suburbs
    df.loc[df['city'] == 'Amsterdam', 'cat_city'] = 1
    df.loc[df['city'] != 'Amsterdam', 'cat_suburbs'] = 1

    # print(df['city'].unique())

    # # get unique values for feature analysis
    # # excluding nan
    # loc_values_list =df['location'].loc[pd.notna(df['location'])].unique().tolist()
    #
    # # join into one string in order to convert back into list but in our conditions
    # separator = ', '
    # loc_values_string = separator.join(loc_values_list)
    # loc_values_string_without_and = loc_values_string.replace(' and', ',')
    #
    # # convert string into list and get unique values
    # loc_values_list_2 = loc_values_string_without_and.split(', ')
    # loc_values_numpy = np.array(loc_values_list_2)
    #
    # print(np.unique(loc_values_numpy))

    # custom creating onehot encoding for location
    # df.loc[df['location'].str.contains(pat='quiet road', na=False, case=False), 'loc_quiet_road'] = 1
    # df.loc[df['location'].str.contains(pat='busy road', na=False, case=False), 'loc_busy_road'] = 1
    # df.loc[df['location'].str.contains(pat='in center', na=False, case=False), 'loc_in_center'] = 1
    # df.loc[df['location'].str.contains(pat='residential district', na=False, case=False), 'loc_res_district'] = 1
    # df.loc[df['location'].str.contains(pat='alongside park', na=False, case=False), 'loc_near_park'] = 1
    # df.loc[df['location'].str.contains(pat='forest|wooded surroundings', na=False, case=False), 'loc_near_forest'] = 1
    # df.loc[df['location'].str.contains(pat='water|waterway|seaview', na=False, case=False), 'loc_near_water'] = 1
    # df.loc[df['location'].str.contains(pat='rural', na=False, case=False), 'loc_rural'] = 1

    df.loc[df['location'].str.contains(pat='quiet road|rural', na=False, case=False), 'loc_quiet'] = 1
    df.loc[df['location'].str.contains(pat='alongside park|forest|wooded surroundings|water|waterway|seaview', na=False, case=False), 'loc_nature'] = 1
    df.loc[df['location'].str.contains(pat='in center|residential district', na=False, case=False), 'loc_walkability'] = 1

    # fill nan for new custom features
    # nan_columns = ['cat_city', 'cat_suburbs', 'loc_quiet_road', 'loc_busy_road', 'loc_in_center', 'loc_res_district', 'loc_near_park',
    #                'loc_near_forest', 'loc_near_water', 'loc_rural']

    nan_columns = ['cat_city', 'cat_suburbs', 'loc_quiet', 'loc_nature', 'loc_walkability']
    for col in nan_columns:
        df[col] = df[col].fillna(0)

    return df


def _profile_clusters(df_segmented, df_stat, cluster_col_name, n_clusters):
    # define value for calculation distribution for each column
    loc_types = ['cat_city', 'cat_suburbs',]
    # loc_properties = ['loc_quiet_road', 'loc_busy_road', 'loc_in_center', 'loc_res_district', 'loc_near_park',
    #                'loc_near_forest', 'loc_near_water', 'loc_rural']
    loc_properties = ['loc_quiet', 'loc_nature', 'loc_walkability']
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


def _load_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, index_col=None, header=0)
        return df
    except FileNotFoundError:
        raise ValueError('[ERROR] CSV file not found!')
    except:
        raise ValueError('[ERROR] Something wrong with loading of CSV file!')
