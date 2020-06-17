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
        df_chosen_features = self._choose_features(df_cleaned)
        df_feature_processed = self._convert_text_data(df_chosen_features)
        df_feature_processed = self._fill_missed_values(df_feature_processed)

        df_normalized = self._get_normalized_df(df_feature_processed)

        # self._familiarity_with_data(df_chosen_features)
        # self._familiarity_with_data(df_normalized)

        # customer segmentation --------------------
        segmented_df = get_segments(df_cleaned, df_normalized)
        analyse_segments(df_cleaned, df_normalized, segmented_df)

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
        # temp = dataset.loc[~dataset['plot_size'].str.contains('m²', na = False)]
        # print(temp['plot_size'].unique())
        # print(dataset['plot_size']) # ob_bath_num, ob_ext_storage, ob_stories, ob_vol_cub, plot_size
        # pd.set_option('display.float_format', lambda x: '%.1f' % x)
        print(dataset['ob_stories'].describe())

        # what's columns and type of data
        print(dataset.shape)

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

        # drop rudiment columns
        df_cleaned = df_cleaned.drop(columns=['web-scraper-order', 'web-scraper-start-url', 'house-link', 'house-link-href'])

        return df_cleaned

    @staticmethod
    def _choose_features(df):
        df_chosen_features = df.copy()
        df_chosen_features = df_chosen_features[['fin_price', 'ob_living_area', 'ob_room_num', 'ob_other_space_inside', 'ob_ext_storage',
                                  'ob_vol_cub', 'plot_size', 'ob_stories']]
        return df_chosen_features

    @staticmethod
    def _convert_text_data(df):
        df['fin_price'] = pd.to_numeric(df['fin_price'].str.replace('[^0-9]', ''))
        df['ob_living_area'] = pd.to_numeric(df['ob_living_area'].str.replace('[^0-9]', ''))
        df['ob_room_num'] = pd.to_numeric(df['ob_room_num'].str[0:2].replace('[^0-9]', ''))
        df['ob_other_space_inside'] = pd.to_numeric(df['ob_other_space_inside'].str.replace('[^0-9]', ''))
        df['ob_ext_storage'] = pd.to_numeric(df['ob_ext_storage'].str.replace('[^0-9]', ''))
        df['ob_vol_cub'] = pd.to_numeric(df['ob_vol_cub'].str.replace('[^0-9]', ''))
        df['plot_size'] = pd.to_numeric(df['plot_size'].str.replace('[^0-9]', ''))

        # TODO it needs to get attic / loft features
        df['ob_stories'] = pd.to_numeric(df['ob_stories'].str[0:2].replace('[^0-9]', ''))

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
        median_columns = ['fin_price', 'ob_living_area', 'ob_room_num', 'ob_vol_cub', 'plot_size']
        for col in median_columns:
            df[col] = df[col].fillna(df[col].median())

        # if NaN it means there isn't other space there
        # TODO perhaps it isn't good idia with 0.0001
        # perhaps it would be better if convert this space into categorical data (None, Small, Normal, Big)
        df['ob_other_space_inside'] = df['ob_other_space_inside'].fillna(0.0001)
        df['ob_ext_storage'] = df['ob_ext_storage'].fillna(0.0001)

        # most properties have 1, 2 or 3 stories
        # so NaN stories replace for 1 or 2 or 3 in random maner
        mask_nan = df['ob_stories'].isnull()
        l = mask_nan.sum()
        s = np.random.choice([1, 2, 3], size=l)
        df.loc[mask_nan, 'ob_stories'] = s

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

        necessary_columns = ['fin_price', 'ob_living_area', 'ob_room_num', 'ob_other_space_inside', 'ob_ext_storage',
                             'ob_stories', 'ob_vol_cub', 'plot_size']
        normalized_df = original_df.copy()
        min_max_scaler = MinMaxScaler()

        for col in necessary_columns:
            # Transform Skewed Data
            normalized_df[col] = np.log(normalized_df[col])

            # scale data
            # min max scaler because there are outliers in dataset
            try:
                normalized_df[[col]] = min_max_scaler.fit_transform(normalized_df[[col]])
            except:
                print(normalized_df[col])

        return normalized_df




# ----------------------------------------------------
data_o = Data()
data_o.run()
