import numpy as np
import pandas as pd


def get_object_segments(df_original):

    df_object_data = df_original[['ob_kind', 'ob_year', 'ob_specific', 'ob_living_area', 'ob_other_space_inside',
                                  'ob_ext_storage', 'ob_vol_cub', 'ob_room_num', 'ob_bath_num', 'ob_bath_facilities',
                                  'ob_stories', 'ob_facilities', 'ob_energy', 'ob_insulation', 'shed_storage',
                                  'garage', 'garden']].copy()
    df_preprocessed = _preprocess_data(df_object_data)

    return df_original


def _preprocess_data(df):

    print(df.info())
    # features for segmentation
    # - bedrooms - because for different number of family members


    df_numeric = _convert_text_data(df)

    df_na_filled = _fill_missed_values(df_numeric)
    df_custom_features = _create_custom_features(df_na_filled)

    # ? df_optimized = _optimize_memory_usage(df_numeric)

    # print(df['ob_ext_storage'])
    # print(df_na_filled['ob_ext_storage'].unique())
    print(df_na_filled['ob_ext_storage'].describe())
    print(df_na_filled['cat_storage'].describe())
    # print(df_na_filled[['ob_ext_storage', 'ob_stories', 'shed_storage', 'garage']].loc[df_na_filled['ob_stories'].str.contains('basement', na=False)])


    return df


def _convert_text_data(df):
    df['ob_living_area'] = pd.to_numeric(df['ob_living_area'].str.replace('[^0-9]', ''))
    df['ob_bedrooms'] = pd.to_numeric(df['ob_room_num'].str.partition('(')[2].str.replace('[^0-9]', ''))
    df['ob_ext_storage'] = pd.to_numeric(df['ob_ext_storage'].str.replace('[^0-9]', ''))

    return df


def _fill_missed_values(df):

    # for part of data just insert median value
    median_columns = ['ob_living_area']
    for col in median_columns:
        df[col] = df[col].fillna(df[col].median())

    # most objects has 4 bedrooms
    df['ob_bedrooms'] = df['ob_bedrooms'].fillna(4)

    # NaN storage - there is no storage
    df['ob_ext_storage'] = df['ob_ext_storage'].fillna(0)




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

    # transform website object types into a few categories
    # 2 - House
    # 1 - Townhouse (different variations)
    # 0 - Flat
    df.loc[df['ob_kind'].str.contains(pat=', detached residential property'), 'cat_ob_type'] = 2
    df.loc[df['ob_kind'].str.contains(pat='semi-detached'), 'cat_ob_type'] = 1
    df.loc[df['ob_kind'].str.contains(pat='row house'), 'cat_ob_type'] = 1
    df.loc[df['ob_kind'].str.contains(pat='corner house'), 'cat_ob_type'] = 1
    df.loc[df['ob_kind'].str.contains(pat='double house'), 'cat_ob_type'] = 1
    df.loc[df['ob_kind'].str.contains(pat='staggered'), 'cat_ob_type'] = 1

    # 0 - No storage
    # 1 - Little storage >=6m2
    # 2 - Medium storage 7-14m2
    # 3 - Large storage 14m2 >
    df.loc[df['ob_ext_storage'] == 0, 'cat_storage'] = 0
    df.loc[(df['ob_ext_storage'] > 0) & (df['ob_ext_storage'] <= 6), 'cat_storage'] = 1
    df.loc[(df['ob_ext_storage'] > 6) & (df['ob_ext_storage'] <= 14), 'cat_storage'] = 2
    df.loc[df['ob_ext_storage'] > 14, 'cat_storage'] = 3


    return df