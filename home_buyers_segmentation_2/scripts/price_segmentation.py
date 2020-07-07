import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes


def get_price_segments(df_original):
    df_price_data = df_original[['fin_price', 'fin_price_per_m']].copy()
    df_preprocessed = _preprocess_data(df_price_data)
    print(df_price_data)



    # to define number of cluster, run this function
    # get_number_of_segments(df_preprocessed)
    num_clusters = 5
    price_cluster_column_name = 'cluster_price'
    df_original_price_segmented = get_segments(df_price_data, df_preprocessed, price_cluster_column_name, num_clusters)
    f_validation, df_stat = validate_cluster_sizes(df_original_price_segmented, price_cluster_column_name)

    # _profile_clusters(df_original_price_segmented, df_stat, price_cluster_column_name, num_clusters)
    df_price_clusters = _prepare_for_overall_clustering(df_original_price_segmented)

    return df_price_clusters


def _preprocess_data(df):
    print('[INFO] Object data preprocessing started...')
    df_numeric = _convert_text_data(df)
    df_na_filled = _fill_missed_values(df_numeric)
    df_custom_features = _create_custom_features(df_na_filled)
   # choosing meaningful features for segmentation
    '''
        - price - take price only, because there is direct correlation with start payment and monthly payment
          start payment - how much money has to save buyer for purchase
          monthly payment - what's household income
    '''
    df_for_segmentation = df_custom_features[['fin_price']]
    df_scaled = _scale_data(df_for_segmentation, ['fin_price'])

    print('[OK] Price data preprocessing finished.')

    return df_scaled


def _convert_text_data(df):
    df['fin_price'] = pd.to_numeric(df['fin_price'].str.replace('[^0-9]', ''))
    df['fin_price_per_m'] = pd.to_numeric(df['fin_price_per_m'].str.replace('[^0-9]', ''))

    return df


def _fill_missed_values(df):
    # it's just a fast solution
    # for better model price has to be depended from place and type of object
    df['fin_price'] = round(df['fin_price'].fillna(df['fin_price'].median()))

    return df


def _create_custom_features(df):
    # affordability of object --------------------------------
    # cost of buying a house in the Netherland 6% of the house price
    df['fin_buying_cost'] = df['fin_price'] * 0.06

    # standart deposit for Netherland is 10%
    df['fin_downpayment'] = df['fin_price'] * 0.1
    df['fin_start_payment'] = df['fin_downpayment'] + df['fin_buying_cost']

    # anuitete formula https://myfin.by/wiki/term/annuitetnyj-platyozh
    n = 360 # loan for 30 years
    i = 0.00283 # loan rate 3.4%
    k = (i * (1 + i)**n) / ((1 + i)**n - 1)
    df['fin_monthly_payment'] = (df['fin_price'] - df['fin_downpayment']) * k

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
    # print(df_segmented)
    # mean columns
    print('Segmentation by price')
    df_means = round(df_segmented.groupby(cluster_col_name)['fin_price', 'fin_price_per_m', 'fin_start_payment', 'fin_monthly_payment'].mean())

    # add distribution values for each cluster
    df_profiling = pd.concat([df_stat, df_means], axis=1, sort=False)

    print(df_profiling.sort_values(0, ascending=False))

    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    print(df_segmented[['fin_price', 'fin_monthly_payment']].loc[df_segmented['cluster_price'] == 0].describe())
    # print(df_segmented[['fin_price', 'fin_price_per_m']].loc[df_segmented['cluster_price'] == 1].describe())
    # print(df_segmented[['fin_price', 'fin_price_per_m']].loc[df_segmented['cluster_price'] == 2].describe())
    # print(df_segmented[['fin_price', 'fin_price_per_m']].loc[df_segmented['cluster_price'] == 3].describe())
    # print(df_segmented[['fin_price', 'fin_price_per_m']].loc[df_segmented['cluster_price'] == 4].describe())


def _prepare_for_overall_clustering(df):
    df_prepared = df[['cluster_price', 'fin_price', 'fin_price_per_m', 'fin_start_payment', 'fin_monthly_payment']]

    return df_prepared
