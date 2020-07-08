"""
Home buyers segmentation by property and district features

Input:
    - Seattle sold properties
    - Seattle districts ratings
Output:


Written by Evgeniy Ryzhkov

------------------------------------------------------------

Usage:

    # parse data
    parse data: python -m scripts.data.py

"""
from .k_means_segmentation import get_number_of_segments, get_segments
from .segments_analysis import analyse_segments
from .object_segmentation import get_object_segments
from .price_segmentation import get_price_segments
from .place_segmentation import get_place_segments

import os
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import re


class Data:

    def run(self):

        # load data
        df_city_loaded = self._load_csv_data()

        # general data cleaning
        df_original_cleaned = self._clean_data(df_city_loaded)

        print('[INFO] Object segmentation --------------------------')
        df_object_segments = get_object_segments(df_original_cleaned)
        print('[INFO] Price segmentation ---------------------------')
        df_price_segments = get_price_segments(df_original_cleaned)
        print('[INFO] Place segmentation ---------------------------')
        df_place_segments = get_place_segments(df_original_cleaned)

        df_merged = pd.concat([df_object_segments, df_price_segments, df_place_segments], axis=1)
        df_preprocessed = self._preprocess_data(df_merged)

        # self._familiarity_with_data(df_original_cleaned)

        # df_city_loaded = self._get_test_data()



        # 1. get property segments
        # - 1.1. data preprocessing
        # - 1.2. property segmentation
        # - 1.3. get property segmented df
        # 2. get price segments
        # - 2.1. data preprocessing
        # - 2.2. price segmentation
        # - 2.3. get price segmented df
        # 3. get place segments
        # 4. get overall segments
        # 4.1. merge part segmentation
        # 4.2. overall segmentation

    @staticmethod
    def _load_csv_data():
        print('[INFO] Parsing started...')
        DATA_DIR = 'data/real_estate/'

        # csv files with city data consist of many separate files
        all_files = glob.glob(os.path.join(DATA_DIR + "/*.csv"))
        li = []
        try:
            for filename in all_files:
                df = pd.read_csv(filename, index_col=None, header=0)
                li.append(df)

            frame = pd.concat(li, axis=0, ignore_index=True)
            return frame

        except FileNotFoundError:
            raise ValueError('CSV file not found!')
        except:
            raise ValueError('Something wrong with CSV file operation!')

    @staticmethod
    def _clean_data(df):
        print('[INFO] Rows number before cleaning data = ', df.shape[0])

        # drop dublicates
        df_cleaned = df.drop_duplicates(subset="house-link-href", keep="first")

        # drop error instances
        df_cleaned = df_cleaned.loc[(pd.notnull(df_cleaned['fin_price']) & pd.notnull(df_cleaned['fin_term']))]

        # drop rudiment columns
        df_cleaned = df_cleaned.drop(columns=['web-scraper-order', 'web-scraper-start-url', 'house-link', 'house-link-href'])

        print('[INFO] Rows number  after cleaning data = ', df_cleaned.shape[0])

        return df_cleaned

    def _preprocess_data(self, df):
        # for overall segmentation choose cluster columns only
        df_for_segmentation = df[['cluster_object', 'cluster_price', 'cluster_place']]

        # for OK scale working, data has to have no zero values
        df_for_segmentation[['cluster_object', 'cluster_price', 'cluster_place']] += 1
        df_scaled = self._scale_data(df_for_segmentation, ['cluster_object', 'cluster_price', 'cluster_place'])

        return df_scaled

    @staticmethod
    def _scale_data(df, cols):
        # normalize and scale data
        df_normalized = df.copy()
        scaler = MinMaxScaler()

        for col in cols:
            # Transform Skewed Data
            df_normalized[col] = np.log(df_normalized[col])

            # scale data
            # min max scaler because there are outliers in dataset
            # TODO there is wrong comment. it's standart scaller has been affected less by outliers
            # not min max
            try:
                df_normalized[[col]] = scaler.fit_transform(df_normalized[[col]])
            except:
                print(df_normalized[col])

        return df_normalized



# ----------------------------------------------------
data_o = Data()
data_o.run()
