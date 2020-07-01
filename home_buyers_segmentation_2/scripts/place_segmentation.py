import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes


def get_place_segments(df_original):
    # print(df_original['city'].loc[df_original['city'] != 'Amsterdam'].unique())
    # удаленность от city center (car min)
    # удаленость от city center (transport, min)
    df_place_data = df_original[['city', 'neighborhood', 'location']].copy()
    df_preprocessed = _preprocess_data(df_place_data)


def _preprocess_data(df):
    print('[INFO] Place data preprocessing started...')
    df_custom_features = _create_custom_features(df)
    print('[OK] Place data preprocessing finished.')


def _create_custom_features(df):
    # in the city or suburbs
    df.loc[df['city'] == 'Amsterdam', 'cat_city'] = 1
    df.loc[df['city'] != 'Amsterdam', 'cat_suburbs'] = 1

    # print(np.sort(df['city'].unique()))
    # print(len(df['city'].unique()))

    # filtered = df.groupby('city').filter(lambda x: len(x) >= 50)
    # print(np.sort(filtered['city'].unique()))
    # print(len(filtered['city'].unique()))
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
