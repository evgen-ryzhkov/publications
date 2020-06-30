import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes


def get_price_segments(df_original):
    df_price_data = df_original[['fin_price', 'fin_price_per_m']].copy()
    df_preprocessed = _preprocess_data(df_price_data)


def _preprocess_data(df):
    print('[INFO] Object data preprocessing started...')
    df_numeric = _convert_text_data(df)
    df_na_filled = _fill_missed_values(df_numeric)
    # pd.set_option('display.float_format', lambda x: '%.0f' % x)
    # print(df_na_filled['fin_price_per_m'].describe())
    # print(df_na_filled['fin_price'].describe())

    # choosing meaningful features for segmentation
    '''
        - price - just experimentation
    '''
    df_for_segmentation = df_na_filled[['fin_price']]
    df_scaled = _scale_data(df_for_segmentation, ['fin_price'])
    print(df_scaled['fin_price'].describe())

    pass


def _convert_text_data(df):
    df['fin_price'] = pd.to_numeric(df['fin_price'].str.replace('[^0-9]', ''))
    df['fin_price_per_m'] = pd.to_numeric(df['fin_price_per_m'].str.replace('[^0-9]', ''))

    return df


def _fill_missed_values(df):
    # it's just a fast solution
    # for better model price has to be depended from place and type of object
    df['fin_price'] = round(df['fin_price'].fillna(df['fin_price'].median()))

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
