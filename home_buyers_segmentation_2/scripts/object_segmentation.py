import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes


def get_object_segments(df_original):
    # print(df_original.info())
    df_object_data = df_original[['ob_kind', 'ob_year', 'ob_specific', 'ob_living_area', 'ob_other_space_inside',
                                  'ob_ext_storage', 'ob_vol_cub', 'ob_room_num', 'ob_bath_num', 'ob_bath_facilities',
                                  'ob_stories', 'ob_facilities', 'ob_energy', 'ob_insulation', 'shed_storage',
                                  'garage', 'garden']].copy()
    df_preprocessed = _preprocess_data(df_object_data)

    # to define number of cluster, run this function
    # get_number_of_segments(df_preprocessed)
    num_clusters = 6
    object_cluster_column_name = 'cluster_object'
    df_original_prop_segmented = get_segments(df_object_data, df_preprocessed, object_cluster_column_name, num_clusters)
    f_validation, df_stat = validate_cluster_sizes(df_original_prop_segmented, object_cluster_column_name)

    _profile_clusters(df_original_prop_segmented, df_stat, object_cluster_column_name, num_clusters)

    return df_original


def _preprocess_data(df):

    print('[INFO] Object data preprocessing started...')

    df_numeric = _convert_text_data(df)
    df_custom_features = _create_custom_features(df_numeric)
    df_na_filled = _fill_missed_values(df_custom_features)

    # choosing meaningful features for segmentation
    '''
        - bedrooms - it can to tell us about number of family members
        - object kind - preferences in personal space
        - storage size - lifestyle (children/sport/garden stuff)
        - car friendly - lifestyle, personal space
        - garden - lifestyle, personal space
        - energy - personal beliefs (environment friendly)
        ? living area - personal space
    '''
    df_for_segmentation = df_na_filled[['ob_bedrooms', 'cat_energy', 'cat_storage', 'cat_garden']]
    # ? df_optimized = _optimize_memory_usage(df_numeric)

    # encode property types
    encoder_1hot = OneHotEncoder()
    df_prop_type_1hot = pd.DataFrame(
        encoder_1hot.fit_transform(df_na_filled[['cat_ob_type']]) \
        .toarray(),
        columns=['Flat', 'House', 'Townhouse']
    )
    # order of columns you can get by print(encoder_1hot.categories_)
    # it will help for creating profiling matrix

    # merge encoded df
    df_encode_merged = pd.concat(
        [df_prop_type_1hot.reset_index(drop=True), df_for_segmentation.reset_index(drop=True)],
        axis=1, sort=False)

    df_scaled = _scale_data(df_encode_merged, ['ob_bedrooms', 'cat_energy', 'cat_storage', 'cat_garden'])

    print('[OK] Object data preprocessing finished.')

    return df_scaled


def _convert_text_data(df):
    df['ob_living_area'] = pd.to_numeric(df['ob_living_area'].str.replace('[^0-9]', ''))
    df['ob_bedrooms'] = pd.to_numeric(df['ob_room_num'].str.partition('(')[2].str.replace('[^0-9]', ''))
    df['ob_ext_storage'] = pd.to_numeric(df['ob_ext_storage'].str.replace('[^0-9]', ''))

    return df


def _fill_missed_values(df):
    # NaN ob_kind - apartment (scrapping bag)
    df['cat_ob_type'] = df['cat_ob_type'].fillna('Flat')

    # for part of data just insert median value
    median_columns = ['ob_living_area', 'cat_energy', 'ob_bedrooms']
    for col in median_columns:
        df[col] = round(df[col].fillna(df[col].median()))

    # NaN storage - there is no storage
    df['cat_storage'] = df['cat_storage'].fillna(1)

    # NaN garden - there is no garden
    df['cat_garden'] = df['cat_garden'].fillna(1)

    return df


def _create_custom_features(df):
    # transform living area into categories
    # 0 - >100 m2
    # 1 - 100-150 m2
    # 2 - 151-250 m2
    # 3 250 m2 >
    df.loc[df['ob_living_area'] < 100, 'cat_living_area'] = 0
    df.loc[(df['ob_living_area'] >= 100) & (df['ob_living_area'] <= 150), 'cat_living_area'] = 1
    df.loc[(df['ob_living_area'] > 150) & (df['ob_living_area'] <= 250), 'cat_living_area'] = 2
    df.loc[df['ob_living_area'] > 250, 'cat_living_area'] = 3

    # Object type Category -----------------
    # 2 - House (fully detached individual house)
    # 1 - Townhouse (semi-detached property, different variations)
    # 0 - Flat
    df.loc[df['ob_kind'].str.contains(pat=', detached residential property', na=False), 'cat_ob_type'] = 'House'
    df.loc[df['ob_kind'].str.contains(pat='semi-detached', na=False), 'cat_ob_type'] = 'Townhouse'
    df.loc[df['ob_kind'].str.contains(pat='row house', na=False), 'cat_ob_type'] = 'Townhouse'
    df.loc[df['ob_kind'].str.contains(pat='corner house', na=False), 'cat_ob_type'] = 'Townhouse'
    df.loc[df['ob_kind'].str.contains(pat='double house', na=False), 'cat_ob_type'] = 'Townhouse'
    df.loc[df['ob_kind'].str.contains(pat='staggered', na=False), 'cat_ob_type'] = 'Townhouse'


    # Storage Category ----------------------
    # 1 - No storage
    # 2 - Little storage >=6m2
    # 3 - Medium storage 7-14m2
    # 4 - Large storage 14m2 >
    df.loc[(df['ob_ext_storage'] > 0) & (df['ob_ext_storage'] <= 6), 'cat_storage'] = 2
    df.loc[(df['ob_ext_storage'] > 6) & (df['ob_ext_storage'] <= 14), 'cat_storage'] = 3
    df.loc[df['ob_ext_storage'] > 14, 'cat_storage'] = 4

    # Energy efficiency ---------------------
    df.loc[df['ob_energy'].str.contains(pat='A', na=False), 'cat_energy'] = 7  # 'A'
    df.loc[df['ob_energy'].str.contains(pat='B', na=False), 'cat_energy'] = 6  # 'B'
    df.loc[df['ob_energy'].str.contains(pat='C', na=False), 'cat_energy'] = 5  # 'C'
    df.loc[df['ob_energy'].str.contains(pat='D', na=False), 'cat_energy'] = 4  # 'D'
    df.loc[df['ob_energy'].str.contains(pat='E', na=False), 'cat_energy'] = 3  # 'E'
    df.loc[df['ob_energy'].str.contains(pat='F', na=False), 'cat_energy'] = 2  # 'F'
    df.loc[df['ob_energy'].str.contains(pat='G', na=False), 'cat_energy'] = 1  # 'G'

    # Car Friendly Category -----------------
    # 0 - No Place for Car
    # 1 - Paid Parking / resident's parking permits
    # 2 - Public parking
    # 3 Parking place / Underground parking
    # 5 Parking on private property
    # 8 Garage
    # TODO wait for getting full data
    # df.loc[df['garage'].str.contains(pat='arage', na=False), 'cat_car_friendly'] = 8
    # df.loc[df['garage'].str.contains(pat='arage', na=False), 'cat_car_friendly'] = 8

    # Garden --------------------------------
    # # old version
    # # 1 - no any garden
    # # 2 - front garden or side garden or terace only
    # # 3 - front + side garden
    # # 4 - back garden
    # # 5 - back + front garden or back + side garde
    # # 6 - back + front + side garden
    # # 7 - surrounded by garden
    # # 8 - surrounded by garden + terrace
    # init_garden_category = np.ones((df.shape[0], 1))
    # df['cat_garden'] = init_garden_category
    #
    # df.loc[(df['garden'].str.contains(pat='back garden', na=False, case=False)), 'cat_garden'] = 4
    # df.loc[(df['garden'].str.contains(pat='front garden', na=False, case=False)), 'cat_garden'] += 1
    # df.loc[(df['garden'].str.contains(pat='side garden', na=False, case=False)), 'cat_garden'] += 1
    # df.loc[(df['garden'].str.contains(pat='surrounded by garden', na=False, case=False)), 'cat_garden'] = 7
    # df.loc[(df['garden'].str.contains(pat='sun terrace|atrium', na=False, case=False)), 'cat_garden'] += 1

    # new version
    # 1 - no any garden
    # 2 - Small Garden: front garden or side garden or terace only or front + side garden
    # 3 - Medium Garden: there is a back garden
    # 4 - Big Garden: Surrounded by garden
    init_garden_category = np.ones((df.shape[0], 1))
    df['cat_garden'] = init_garden_category

    df.loc[(df['garden'].str.contains(pat='sun terrace|atrium', na=False, case=False)), 'cat_garden'] = 2
    df.loc[(df['garden'].str.contains(pat='front garden|side garden', na=False, case=False)), 'cat_garden'] = 2
    df.loc[(df['garden'].str.contains(pat='back garden', na=False, case=False)), 'cat_garden'] = 3
    df.loc[(df['garden'].str.contains(pat='surrounded by garden', na=False, case=False)), 'cat_garden'] = 4


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

    print('Houses=', len(df_segmented.loc[df_segmented['cat_ob_type'] == 'House']))

    # replace categorical numbers by readable values
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 1] = 'No storage'
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 2] = 'S Storage'
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 3] = 'M Storage'
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 4] = 'B Storage'

    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 7] = 'A'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 6] = 'B'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 5] = 'C'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 4] = 'D'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 3] = 'E'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 2] = 'F'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 1] = 'G'

    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 1] = 'No Garden'
    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 2] = 'S Garden'
    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 3] = 'M Garden'
    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 4] = 'B Garden'

    # define value for calculation distribution for each column
    ob_types = ['Flat', 'House', 'Townhouse']
    ob_storages = ['No storage', 'S Storage', 'M Storage', 'B Storage']
    ob_energies = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    ob_gardens = ['No Garden', 'S Garden', 'M Garden', 'B Garden']
    df_profiling_cols = ob_types + ob_storages + ob_energies + ob_gardens

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
            if col in ob_types:
                df_col_name = 'cat_ob_type'
            elif col in ob_storages:
                df_col_name = 'cat_storage'
            elif col in ob_energies:
                df_col_name = 'cat_energy'
            elif col in ob_gardens:
                df_col_name = 'cat_garden'

            percent_val = round((len(df_cluster.loc[df_cluster[df_col_name] == col]) / df_cluster_len) * 100)
            df_profiling.loc[cluster, col] = percent_val

    # rename price columns for better reading
    # print(df_profiling.columns[3])
    # df_profiling.rename(columns={
    #     df_profiling.columns[3]: 'No storage',
    #     df_profiling.columns[4]: 'S Storage',
    #     df_profiling.columns[5]: 'M Storage',
    #     df_profiling.columns[6]: 'B Storage',
    #     df_profiling.columns[7]: 'A',
    # }, inplace=True)

    # add distribution values for each cluster
    df_profiling = pd.concat([df_stat, df_profiling], axis=1, sort=False)
    print(df_profiling)

    # rename cluster percents for better reading
    df_profiling.rename(columns={
        0: 'Cluster %'
    }, inplace=True)

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_profiling, annot=True)
    plt.show()

