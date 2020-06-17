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
from .k_means_segmentation import get_segments
from .segments_analysis import analyse_segments

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

        # house feature analysis
        # self._familiarity_with_data(df_city_loaded)

        # data preprocessing -----------------------
        # data cleaning
        df_cleaned = self._clean_data(df_city_loaded)

        # feature engineering
        df_feature_processed = self._convert_text_data(df_cleaned)
        df_feature_processed = self._fill_missed_values(df_feature_processed)

        df_normalized = self._get_normalized_df(df_feature_processed)


        self._familiarity_with_data(df_feature_processed)
        self._familiarity_with_data(df_normalized)

        # customer segmentation --------------------

        # original_df, normalized_df = self._prepare_data(df_city_loaded)
        # self._familiarity_with_data(original_df)
        # self._define_correlations(original_df)
        # segmented_df = get_segments(original_df, normalized_df)
        # analyse_segments(original_df, normalized_df, segmented_df)

    @staticmethod
    def _load_csv_data():
        print('[INFO] Parsing started...')
        DATA_DIR = 'data/'

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
    def _familiarity_with_data(dataset):

        # what's size and fullness datas
        # print('Dataset size', dataset.shape)
        # print(dataset.head())
        # print(dataset.info())
        #
        # print(dataset['fin_price'])
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        print(dataset['fin_price'].describe())

        # what's columns and type of data
        # print(dataset.shape)

        # familiarity with particular column
        # pd.set_option('display.float_format', lambda x: '%.1f' % x)
        # print(dataset['Type'].describe())
        # dataset.hist(column='Type')
        # plt.show()

        # print(dataset.loc[dataset['Car'].isnull()]['Type'])

        # return df_houses

    @staticmethod
    def _clean_data(df):

        # drop dublicates
        df_cleaned = df.drop_duplicates(subset="house-link-href", keep="first")

        # drop error instances
        df_cleaned = df_cleaned.loc[(pd.notnull(df_cleaned['fin_price']) & pd.notnull(df_cleaned['fin_term']))]

        return df_cleaned

    @staticmethod
    def _convert_text_data(df):
        df['fin_price'] = pd.to_numeric(df['fin_price'].str.replace('[^0-9]', ''))

        return df

    @staticmethod
    def _prepare_data(self, dataset):

        original_df = self._get_houses(dataset)
        original_df = self._convert_text_values(original_df)
        original_df = self._get_short_df(original_df)
        original_df = self._fill_missed_values(original_df)
        normalized_df = self._get_normalized_df(original_df)
        return original_df, normalized_df


    @staticmethod
    def _get_short_df(df_houses):
        df_houses_short = df_houses[
            ['IntType', 'Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'Distance', 'Price']]
        return df_houses_short

    @staticmethod
    def _fill_missed_values(df):

        # for part of data just insert median value
        median_columns = ['fin_price']
        for col in median_columns:
            df[col] = df[col].fillna(df[col].median())


        # most single family houses have 1 or 2 stories, and their proportion is about equal
        # so NaN stories replace for 1 or 2 in random maner
        # df['Car'] = df['Car'].fillna(
        #     pd.Series(np.random.choice([0, 1, 2], size=len(df.index))))
        #
        # df_bed_0 = df.loc[df['Bedroom2'] == 0]
        # df['Bedroom2'] = pd.Series(np.random.choice([1, 2, 3], size=len(df_bed_0.index)))
        # df['Bedroom2'] = df['Bedroom2'].fillna(
        #     pd.Series(np.random.choice([1, 2], size=len(df.index))))
        #

        return df

    @staticmethod
    def _create_custom_features(df_houses):
        df_houses.loc[df_houses['ob-type-3'].eq('Townhouse'), 'ob-type'] = 1
        df_houses.loc[df_houses['ob-type-4'].eq('Multi Family'), 'ob-type'] = 2
        df_houses.loc[df_houses['ob-type-1'].eq('Single Family Home'), 'ob-type'] = 3

        return df_houses

    @staticmethod
    def _define_correlations(df_houses_short):
        corr_matrix = df_houses_short.corr()
        print(corr_matrix['Price'].sort_values(ascending=False))


    @staticmethod
    def _get_normalized_df(original_df):
        '''
            pre-processing data for k-mean segmentation:
            - transform skewed data with log tranasformation
            - scale data
        '''

        necessary_columns = ['fin_price']
        normalized_df = original_df.copy()
        min_max_scaler = MinMaxScaler()

        for col in necessary_columns:
            # Transform Skewed Data
            normalized_df[col] = np.log(normalized_df[col])

            # normalize data
            try:
                normalized_df[[col]] = min_max_scaler.fit_transform(normalized_df[[col]])
            except:
                print(normalized_df[col])

        return normalized_df




# ----------------------------------------------------
data_o = Data()
data_o.run()
