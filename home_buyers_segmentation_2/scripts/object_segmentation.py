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
    df_object_clusters = _prepare_for_overall_clustering(df_original_prop_segmented)

    return df_object_clusters


def _preprocess_data(df):

    print('[INFO] Object data preprocessing started...')

    df_numeric = _convert_text_data(df)
    df_custom_features = _create_custom_features(df_numeric)
    df_na_filled = _fill_missed_values(df_custom_features)

    # choosing meaningful features for segmentation
    '''
        - bedrooms - it can to tell us about number of family members
        - ? object kind - preferences in personal space
        - storage size - lifestyle (children/sport/garden stuff)
        - car friendly - lifestyle, personal space
        - garden - lifestyle, personal space
        - ? energy - personal beliefs (environment friendly)
        - ? living area - personal space
    '''
    df_for_segmentation = df_na_filled[['ob_bedrooms', 'cat_energy', 'cat_storage', 'cat_garden',
                                        'cat_living_area', 'cat_car_friendly']]
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

    df_scaled = _scale_data(df_encode_merged, ['ob_bedrooms', 'cat_energy', 'cat_storage', 'cat_garden',
                                               'cat_living_area', 'cat_car_friendly'])

    print('[OK] Object data preprocessing finished.')

    return df_scaled


def _convert_text_data(df):
    df['ob_living_area'] = pd.to_numeric(df['ob_living_area'].str.replace('[^0-9]', ''))
    df['ob_bedrooms'] = pd.to_numeric(df['ob_room_num'].str.partition('(')[2].str.replace('[^0-9]', ''))
    df['ob_ext_storage'] = pd.to_numeric(df['ob_ext_storage'].str.replace('[^0-9]', ''))
    df['ob_vol_cub'] = pd.to_numeric(df['ob_vol_cub'].str.replace('[^0-9]', ''))

    return df


def _fill_missed_values(df):
    # NaN ob_kind - apartment (scrapping bag)
    df['cat_ob_type'] = df['cat_ob_type'].fillna('Flat')

    # for part of data just insert median value
    median_columns = ['ob_living_area', 'cat_energy', 'ob_bedrooms', 'ob_vol_cub']
    for col in median_columns:
        df[col] = round(df[col].fillna(df[col].median()))

    # NaN storage - there is no storage
    df['cat_storage'] = df['cat_storage'].fillna(1)

    # NaN garden - there is no garden
    df['cat_garden'] = df['cat_garden'].fillna(1)

    return df


def _create_custom_features(df):
    # transform living area into categories
    # 1 - >100 m2
    # 2 - 100-150 m2
    # 3 - 151-250 m2
    # 4 250 m2 >
    df['ob_living_area'] = round(df['ob_living_area'].fillna(df['ob_living_area'].median()))
    df.loc[df['ob_living_area'] < 100, 'cat_living_area'] = 1
    df.loc[(df['ob_living_area'] >= 100) & (df['ob_living_area'] <= 150), 'cat_living_area'] = 2
    df.loc[(df['ob_living_area'] > 150) & (df['ob_living_area'] <= 250), 'cat_living_area'] = 3
    df.loc[df['ob_living_area'] > 250, 'cat_living_area'] = 4

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
    df.loc[df['ob_energy'].str.contains(pat='A|B', na=False), 'cat_energy'] = 3  # Good Energy Efficiency
    df.loc[df['ob_energy'].str.contains(pat='C|D', na=False), 'cat_energy'] = 2  # Normal Energy Efficiency
    df.loc[df['ob_energy'].str.contains(pat='E|F|G', na=False), 'cat_energy'] = 1  # Poor Energy Efficiency

    # Car Friendly Category -----------------
    # 1 - Poor CF (NaN / No Place for Car / Paid Parking / resident's parking permits)
    # 2 - Usual CF (Public parking)
    # 3 - Good CF (Parking place / Underground parking)
    # 4 - VGood CF (Parking on private property / Garage)
    df.loc[df['garage'].isnull(), 'cat_car_friendly'] = 1
    df.loc[df['garage'].str.contains(pat='parking place', na=False, case=False), 'cat_car_friendly'] = 2
    df.loc[df['garage'].str.contains(pat='underground parking', na=False,
                                     case=False), 'cat_car_friendly'] = 3
    df.loc[df['garage'].str.contains(pat='garage', na=False, case=False), 'cat_car_friendly'] = 4
    # fill strange values
    df['cat_car_friendly'] = df['cat_car_friendly'].fillna(2)


    # Garden --------------------------------
    # 1 - no any garden
    # 2 - Small Garden: front garden or side garden or terace only or front + side garden
    # 3 - Medium Garden: there is a back garden
    # 4 - Big Garden: Surrounded by garden
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

    # replace categorical numbers by readable values
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 1] = 'No storage'
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 2] = 'S Storage'
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 3] = 'M Storage'
    df_segmented['cat_storage'].loc[df_segmented['cat_storage'] == 4] = 'B Storage'

    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 3] = 'Good EE'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 2] = 'Normal EE'
    df_segmented['cat_energy'].loc[df_segmented['cat_energy'] == 1] = 'Poor EE'

    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 1] = 'No Garden'
    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 2] = 'S Garden'
    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 3] = 'M Garden'
    df_segmented['cat_garden'].loc[df_segmented['cat_garden'] == 4] = 'B Garden'

    df_segmented['cat_car_friendly'].loc[df_segmented['cat_car_friendly'] == 1] = 'Poor CF'
    df_segmented['cat_car_friendly'].loc[df_segmented['cat_car_friendly'] == 2] = 'Usual CF'
    df_segmented['cat_car_friendly'].loc[df_segmented['cat_car_friendly'] == 3] = 'Good CF'
    df_segmented['cat_car_friendly'].loc[df_segmented['cat_car_friendly'] == 4] = 'VGood CF'


    # define value for calculation distribution for each column
    ob_types = ['Flat', 'House', 'Townhouse']
    ob_storages = ['No storage', 'S Storage', 'M Storage', 'B Storage']
    ob_energies = ['Good EE', 'Normal EE', 'Poor EE']
    ob_gardens = ['No Garden', 'S Garden', 'M Garden', 'B Garden']
    ob_cars = ['Poor CF', 'Usual CF', 'Good CF', 'VGood CF']
    df_profiling_cols = ob_types + ob_storages + ob_energies + ob_gardens + ob_cars

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
            elif col in ob_cars:
                df_col_name = 'cat_car_friendly'

            percent_val = round((len(df_cluster.loc[df_cluster[df_col_name] == col]) / df_cluster_len) * 100)
            df_profiling.loc[cluster, col] = percent_val

    # mean columns
    df_means = round(df_segmented.groupby(cluster_col_name)['cat_living_area', 'ob_bedrooms', 'ob_vol_cub'].mean())

    # add distribution values for each cluster
    df_profiling = pd.concat([df_stat, df_profiling, df_means], axis=1, sort=False)


    # rename cluster percents for better reading
    df_profiling.rename(columns={
        0: 'Cluster %',
        'ob_bedrooms': 'Bedrooms',
        'cat_living_area': 'Liv area',
        'ob_vol_cub': 'Cub m3'

    }, inplace=True)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # full screen mode

    # exclude some columns (like means) from heatmap
    mask = np.zeros(df_profiling.shape)
    mask[:, [19, 20, 21]] = True

    cm = sns.light_palette("green")

    sns.heatmap(df_profiling, mask=mask, annot=True, fmt="g", cmap=cm)
    sns.heatmap(df_profiling, alpha=0, cbar=False, annot=True, fmt="g", annot_kws={"color": "black"})
    plt.show()

    # transform some number value into readable
    df_profiling['Liv area'].loc[df_profiling['Liv area'] == 1] = '>100 m2'
    df_profiling['Liv area'].loc[df_profiling['Liv area'] == 2] = '100-150m2'
    df_profiling['Liv area'].loc[df_profiling['Liv area'] == 3] = '151-250m2'
    df_profiling['Liv area'].loc[df_profiling['Liv area'] == 4] = '250m2 +'

    print(df_profiling.sort_values('Cluster %', ascending=False))


def _prepare_for_overall_clustering(df):

    # choose object features for common profiling
    df_prepared = df[['cluster_object', 'cat_ob_type', 'ob_bedrooms', 'ob_living_area', 'cat_living_area', 'ob_vol_cub',
                      'cat_storage', 'cat_energy', 'cat_car_friendly', 'cat_garden']].copy()

    # transform some number value into readable
    df_prepared['cat_living_area'].loc[df_prepared['cat_living_area'] == 1] = '>100 m2'
    df_prepared['cat_living_area'].loc[df_prepared['cat_living_area'] == 2] = '100-150m2'
    df_prepared['cat_living_area'].loc[df_prepared['cat_living_area'] == 3] = '151-250m2'
    df_prepared['cat_living_area'].loc[df_prepared['cat_living_area'] == 4] = '250m2 +'

    return df_prepared
