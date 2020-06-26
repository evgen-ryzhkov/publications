import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt


def get_object_segments(df_original):
    # print(df_original.info())
    df_object_data = df_original[['ob_kind', 'ob_year', 'ob_specific', 'ob_living_area', 'ob_other_space_inside',
                                  'ob_ext_storage', 'ob_vol_cub', 'ob_room_num', 'ob_bath_num', 'ob_bath_facilities',
                                  'ob_stories', 'ob_facilities', 'ob_energy', 'ob_insulation', 'shed_storage',
                                  'garage', 'garden']].copy()
    df_preprocessed = _preprocess_data(df_object_data)

    return df_original


def _preprocess_data(df):

    # print(df.info())

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
    df_for_segmentation = df_na_filled[['cat_ob_type', 'ob_bedrooms', 'cat_energy', 'cat_storage', 'cat_garden']]
    # ? df_optimized = _optimize_memory_usage(df_numeric)

    # encode property types
    encoder_1hot = OneHotEncoder()
    df_prop_type_1hot = pd.DataFrame(
        encoder_1hot.fit_transform(df_for_segmentation[['cat_ob_type']]) \
        .toarray(),
        columns=['Flat', 'Townhouse', 'House']
    )
    # order of columns you can get by print(df_prop_type_1hot.categories_)

    # merge encoded df
    df_encode_merged = pd.concat(
        [df_prop_type_1hot.reset_index(drop=True), df_for_segmentation.reset_index(drop=True)],
        axis=1, sort=False)


    df_scaled = _scale_data(df_encode_merged, ['ob_bedrooms', 'cat_energy', 'cat_storage', 'cat_garden'])
    print(df_scaled['cat_garden'].describe())
    # df_encode_merged.hist(column='cat_energy')
    # plt.show()



    # df_normalized['ob_bedrooms'] = np.log(df_normalized['ob_bedrooms'])





    # print(df_for_segmentation)
    # print(df_na_filled['garden'].unique())
    # print(df_na_filled['cat_garden'].unique())
    # print(df_na_filled['garden'].describe())
    # print(df_na_filled['cat_garden'].describe())
    # debug= df_na_filled[['cat_garden', 'garden']].loc[df_na_filled['cat_garden'] == 1]
    # print(debug['garden'].unique())
    # print(df_custom_features['cat_car_friendly'].describe())
    # print(df_na_filled[['ob_ext_storage', 'ob_stories', 'shed_storage', 'garage']].loc[df_na_filled['ob_stories'].str.contains('basement', na=False)])


    return df


def _convert_text_data(df):
    df['ob_living_area'] = pd.to_numeric(df['ob_living_area'].str.replace('[^0-9]', ''))
    df['ob_bedrooms'] = pd.to_numeric(df['ob_room_num'].str.partition('(')[2].str.replace('[^0-9]', ''))
    df['ob_ext_storage'] = pd.to_numeric(df['ob_ext_storage'].str.replace('[^0-9]', ''))

    return df


def _fill_missed_values(df):
    # NaN ob_kind - apartment (scrapping bag)
    df['cat_ob_type'] = df['cat_ob_type'].fillna('Flat')

    # for part of data just insert median value
    median_columns = ['ob_living_area', 'cat_energy']
    for col in median_columns:
        df[col] = round(df[col].fillna(df[col].median()))

    # most objects has 4 bedrooms
    df['ob_bedrooms'] = df['ob_bedrooms'].fillna(4)

    # NaN storage - there is no storage
    df['cat_storage'] = df['cat_storage'].fillna(1)

    # NaN garden - there is no garden
    df['cat_garden'] = df['cat_garden'].fillna(10)

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
    # 1 - no any garden
    # 2 - front garden or side garden or terace only
    # 3 - front + side garden
    # 4 - back garden
    # 5 - back + front garden or back + side garde
    # 6 - back + front + side garden
    # 7 - surrounded by garden
    # 8 - surrounded by garden + terrace
    init_garden_category = np.ones((df.shape[0], 1))
    df['cat_garden'] = init_garden_category

    df.loc[(df['garden'].str.contains(pat='back garden', na=False, case=False)), 'cat_garden'] = 4
    df.loc[(df['garden'].str.contains(pat='front garden', na=False, case=False)), 'cat_garden'] += 1
    df.loc[(df['garden'].str.contains(pat='side garden', na=False, case=False)), 'cat_garden'] += 1
    df.loc[(df['garden'].str.contains(pat='surrounded by garden', na=False, case=False)), 'cat_garden'] = 7
    df.loc[(df['garden'].str.contains(pat='sun terrace|atrium', na=False, case=False)), 'cat_garden'] += 1

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
