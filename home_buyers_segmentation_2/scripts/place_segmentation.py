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

    df_place_merged = df_place_data.reset_index().merge(df_neighborhood, how="left", left_on=['city', 'neighborhood'],
                                right_on=['city', 'neighborhood']).set_index('index')

    df_preprocessed = _preprocess_data(df_place_merged)

    # to define number of cluster, run this function
    # get_number_of_segments(df_preprocessed)
    num_clusters = 6
    place_cluster_column_name = 'cluster_place'
    df_original_place_segmented = get_segments(df_place_merged, df_preprocessed, place_cluster_column_name, num_clusters)
    f_validation, df_stat = validate_cluster_sizes(df_original_place_segmented, place_cluster_column_name)

    # uncomment for profiling
    # _profile_clusters(df_original_place_segmented, df_stat, place_cluster_column_name, num_clusters)

    df_place_clusters = _prepare_for_overall_clustering(df_original_place_segmented)
    return df_place_clusters


def _preprocess_data(df):
    print('[INFO] Data preprocessing started...')

    df_numeric = _convert_text_data(df)

    df_custom_features = _create_custom_features(df_numeric)
    df_filled_nan = _fill_nan(df_custom_features)

    # choosing meaningful features for segmentation
    '''
        - cat_city/suburbs - different lifestyle
        - school_cat (school numbers) - good for family
        - cafe/restaraunts - lifestyle
        - nightlife - lifestyle 
        - park - family / dogs / outdoor activities
        - commit driving time - how far from Amsterdam's center
    '''
    df_for_segmentation = df_filled_nan[['loc_view', 'school_cat', 'cafe_rest_cat',
                                         'nightlife_cat', 'commit_driving_cat']]

    # encode property types
    encoder_1hot = OneHotEncoder()
    df_loc_type_1hot = pd.DataFrame(
        encoder_1hot.fit_transform(df_filled_nan[['cat_loc']]) \
            .toarray(),
        columns=['City', 'Suburbs']
    )
    # order of columns you can get by print(encoder_1hot.categories_)
    # it will help for creating profiling matrix

    # merge encoded df
    df_encode_merged = pd.concat(
        [df_loc_type_1hot.reset_index(drop=True), df_for_segmentation.reset_index(drop=True)],
        axis=1, sort=False)

    df_scaled = _scale_data(df_encode_merged, ['school_cat', 'cafe_rest_cat', 'nightlife_cat', 'commit_driving_cat'])

    print('[OK] Data preprocessing finished.')
    return df_scaled


def _convert_text_data(df):
    print('[INFO] Converting commit driving time into minutes...')
    df_converted = _convert_commit_time(df, 'commit_time_driving')

    # as I exclude transit time from segmentation features
    # it's no need to spend time for converting
    # print('[INFO] Converting commit transit time into minutes...')
    # df_converted = _convert_commit_time(df_converted, 'commit_time_transit')

    return df_converted


def _convert_commit_time(df, col_name):
    # convert commit time from hours & minutes format into minutes only
    df[col_name + '_mins'] = ''

    for index, row in df.iterrows():
        # if nan just skip it for now
        if pd.isna(row[col_name]):
            continue
        else:
            s = row[col_name]

        hours_exist = False
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
            df.loc[index, col_name+'_mins'] = int(hours.replace(' ', '')) * 60 + int(re.sub('[s ]', '', mins))
        except:
            print('hours=', hours)
            print('mins=', mins)

    return df


def _create_custom_features(df):
    # in the city or suburbs
    df.loc[df['city'] == 'Amsterdam', 'cat_loc'] = 'City'
    df.loc[df['city'] != 'Amsterdam', 'cat_loc'] = 'Suburbs'

    df.loc[df['location'].str.contains(pat='alongside park|forest|wooded surroundings|water|waterway|seaview', na=False, case=False), 'loc_view'] = 1

    df.loc[df['school_num']<6, 'school_cat'] = 1
    df.loc[(df['school_num']>5) & (df['school_num']<13), 'school_cat'] = 2
    df.loc[df['school_num']>12, 'school_cat'] = 3

    df.loc[(df['cafe_num'] + df['restaurant_num']) < 15, 'cafe_rest_cat'] = 1
    df.loc[((df['cafe_num'] + df['restaurant_num']) >= 15) & ((df['cafe_num'] + df['restaurant_num']) < 25), 'cafe_rest_cat'] = 2
    df.loc[(df['cafe_num'] + df['restaurant_num']) >= 25 , 'cafe_rest_cat'] = 3

    df.loc[(df['bar_num'] + df['night_clubs_num'] + df['movie_theatres_num']) < 6, 'nightlife_cat'] = 1
    df.loc[(df['bar_num'] + df['night_clubs_num'] + df['movie_theatres_num']) >= 6, 'nightlife_cat'] = 2
    df.loc[(df['bar_num'] + df['night_clubs_num'] + df['movie_theatres_num']) >= 10, 'nightlife_cat'] = 3

    df.loc[(df['park_num']) < 1, 'park_cat'] = 1
    df.loc[(df['park_num']) <= 2, 'park_cat'] = 2
    df.loc[(df['park_num']) > 2, 'park_cat'] = 3

    df.loc[(df['commit_time_driving_mins']) >= 60, 'commit_driving_cat'] = 1
    df.loc[(df['commit_time_driving_mins']) < 60, 'commit_driving_cat'] = 2
    df.loc[(df['commit_time_driving_mins']) <= 40, 'commit_driving_cat'] = 3
    df.loc[(df['commit_time_driving_mins']) <= 20, 'commit_driving_cat'] = 4

    return df


def _fill_nan(df):
    nan_columns = ['loc_view']
    for col in nan_columns:
        df[col] = df[col].fillna(0)

    df['nightlife_cat'] = df['nightlife_cat'].fillna(1)

    return df


def _scale_data(df, cols):
    # normalize and scale data
    df_normalized = df.copy()
    min_max_scaler = MinMaxScaler()

    for col in cols:
        # Transform Skewed Data
        df_normalized[col] = np.log(df_normalized[col])

        # scale data
        # min max scaler because there are outliers in dataset
        try:
            df_normalized[[col]] = min_max_scaler.fit_transform(df_normalized[[col]])
        except:
            print(df_normalized[col])

    return df_normalized


def _profile_clusters(df_segmented, df_stat, cluster_col_name, n_clusters):

    # replace categorical numbers by readable values
    df_segmented['loc_view'].loc[df_segmented['loc_view'] == 0] = 'No view'
    df_segmented['loc_view'].loc[df_segmented['loc_view'] == 1] = 'Good view'

    df_segmented['school_cat'].loc[df_segmented['school_cat'] == 1] = 'Few sch'
    df_segmented['school_cat'].loc[df_segmented['school_cat'] == 2] = 'A few of sch'
    df_segmented['school_cat'].loc[df_segmented['school_cat'] == 3] = 'A lot of sch'

    df_segmented['cafe_rest_cat'].loc[df_segmented['cafe_rest_cat'] == 1] = 'Few cafe'
    df_segmented['cafe_rest_cat'].loc[df_segmented['cafe_rest_cat'] == 2] = 'A few of cafe'
    df_segmented['cafe_rest_cat'].loc[df_segmented['cafe_rest_cat'] == 3] = 'A lot of cafe'

    df_segmented['nightlife_cat'].loc[df_segmented['nightlife_cat'] == 1] = 'Poor N_life'
    df_segmented['nightlife_cat'].loc[df_segmented['nightlife_cat'] == 2] = 'Good N_life'
    df_segmented['nightlife_cat'].loc[df_segmented['nightlife_cat'] == 3] = 'V Good N_life'

    # df_segmented['park_cat'].loc[df_segmented['park_cat'] == 1] = 'None parks'
    # df_segmented['park_cat'].loc[df_segmented['park_cat'] == 2] = 'A few of parks'
    # df_segmented['park_cat'].loc[df_segmented['park_cat'] == 3] = 'A lot of parks'

    df_segmented['commit_driving_cat'].loc[df_segmented['commit_driving_cat'] == 1] = '>1h driving'
    df_segmented['commit_driving_cat'].loc[df_segmented['commit_driving_cat'] == 2] = '40-60 mins driving'
    df_segmented['commit_driving_cat'].loc[df_segmented['commit_driving_cat'] == 3] = '20-40 mins driving'
    df_segmented['commit_driving_cat'].loc[df_segmented['commit_driving_cat'] == 4] = '<20 mins driving'

    # define value for calculation distribution for each column
    loc_types = ['City', 'Suburbs']
    place_views = ['No view', 'Good view']
    place_schools = ['Few sch', 'A few of sch', 'A lot of sch']
    place_cafes = ['Few cafe', 'A few of cafe', 'A lot of cafe']
    place_nightlife = ['Poor N_life', 'Good N_life', 'V Good N_life']
    # place_parks = ['None parks', 'A few of parks', 'A lot of parks']
    place_driving = ['>1h driving', '40-60 mins driving', '20-40 mins driving', '<20 mins driving']

    df_profiling_cols = loc_types + place_views + place_schools + place_cafes + place_nightlife + place_driving

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

            # there are different columns in original df
            # for different profiling columns
            # for example: columnn prop_size in df_original
            # transforms into three columns S, M, L in df_profiling
            df_col_name = ''
            if col in place_views:
                df_col_name = 'loc_view'
            elif col in loc_types:
                df_col_name = 'cat_loc'
            elif col in place_schools:
                df_col_name = 'school_cat'
            elif col in place_cafes:
                df_col_name = 'cafe_rest_cat'
            elif col in place_nightlife:
                df_col_name = 'nightlife_cat'
            elif col in place_driving:
                df_col_name = 'commit_driving_cat'

            percent_val = round((len(df_cluster.loc[df_cluster[df_col_name] == col]) / df_cluster_len) * 100)
            df_profiling.loc[cluster, col] = percent_val

    # add distribution values for each cluster
    df_profiling = pd.concat([df_stat, df_profiling], axis=1, sort=False)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # full screen mode

    sns.heatmap(df_profiling, annot=True, fmt="g")
    plt.show()


def _prepare_for_overall_clustering(df):

    # choose object features for common profiling
    df_prepared = df[['cluster_place', 'cat_loc', 'loc_view', 'school_cat', 'cafe_rest_cat', 'nightlife_cat',
                      'commit_driving_cat']].copy()

    # replace categorical numbers by readable values
    # TODO there is duplicate code here
    df_prepared['loc_view'].loc[df_prepared['loc_view'] == 0] = 'No view'
    df_prepared['loc_view'].loc[df_prepared['loc_view'] == 1] = 'Good view'

    df_prepared['school_cat'].loc[df_prepared['school_cat'] == 1] = 'Few sch'
    df_prepared['school_cat'].loc[df_prepared['school_cat'] == 2] = 'A few of sch'
    df_prepared['school_cat'].loc[df_prepared['school_cat'] == 3] = 'A lot of sch'

    df_prepared['cafe_rest_cat'].loc[df_prepared['cafe_rest_cat'] == 1] = 'Few cafe'
    df_prepared['cafe_rest_cat'].loc[df_prepared['cafe_rest_cat'] == 2] = 'A few of cafe'
    df_prepared['cafe_rest_cat'].loc[df_prepared['cafe_rest_cat'] == 3] = 'A lot of cafe'

    df_prepared['nightlife_cat'].loc[df_prepared['nightlife_cat'] == 1] = 'Poor N_life'
    df_prepared['nightlife_cat'].loc[df_prepared['nightlife_cat'] == 2] = 'Good N_life'
    df_prepared['nightlife_cat'].loc[df_prepared['nightlife_cat'] == 3] = 'V Good N_life'

    # df_prepared['park_cat'].loc[df_prepared['park_cat'] == 1] = 'None parks'
    # df_prepared['park_cat'].loc[df_prepared['park_cat'] == 2] = 'A few of parks'
    # df_prepared['park_cat'].loc[df_prepared['park_cat'] == 3] = 'A lot of parks'

    df_prepared['commit_driving_cat'].loc[df_prepared['commit_driving_cat'] == 1] = '>1h driving'
    df_prepared['commit_driving_cat'].loc[df_prepared['commit_driving_cat'] == 2] = '40-60 mins driving'
    df_prepared['commit_driving_cat'].loc[df_prepared['commit_driving_cat'] == 3] = '20-40 mins driving'
    df_prepared['commit_driving_cat'].loc[df_prepared['commit_driving_cat'] == 4] = '<20 mins driving'

    return df_prepared


def _load_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, index_col=None, header=0)
        return df
    except FileNotFoundError:
        raise ValueError('[ERROR] CSV file not found!')
    except:
        raise ValueError('[ERROR] Something wrong with loading of CSV file!')
