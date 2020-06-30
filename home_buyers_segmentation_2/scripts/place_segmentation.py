import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes


def get_place_segments(df_original):
    # print(df_original['city'].loc[df_original['city'] == 'Amsterdam'])
    df_place_data = df_original[['city', 'neighborhood']].copy()
    df_preprocessed = _preprocess_data(df_place_data)


def _preprocess_data(df):
    print('[INFO] Object data preprocessing started...')
    df_custom_features = _create_custom_features(df)


def _create_custom_features(df):
    # in the city or suburbs
    df.loc[df['city'] == 'Amsterdam', 'cat_location'] = 'City'
    df.loc[df['city'] != 'Amsterdam', 'cat_location'] = 'Suburbs'

    print(df['city'].unique())
