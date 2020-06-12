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
    parse data: python -m scripts.data.py --op=parse_data --city_dir=[dir_name_inside_data_dir]

"""
from .k_means_segmentation import get_segments
from .segments_analysis import analyse_segments

import argparse
import os
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random


class Data:

    def run(self):
        args = self._get_command_line_arguments()

        # load data
        if args['op'] == 'parse_data':
            df_city_loaded = self._load_csv_data(args['city_dir'])

            # house feature analysis
            df_houses = self._get_df_houses(df_city_loaded)

            # customer segmentation --------------------
            # self._familiarity_with_data(city_data)
            # original_df, normalized_df = self._prepare_data(city_data)
            # self._familiarity_with_data(original_df)
            # segmented_df = get_segments(original_df, normalized_df)
            # analyse_segments(original_df, normalized_df, segmented_df)

    @staticmethod
    def _load_csv_data(city_dir):
        print('[INFO] Parsing started...')
        DATA_DIR = 'data/'

        # csv files with city data consist of many separate files
        all_files = glob.glob(os.path.join(DATA_DIR + city_dir + "/*.csv"))
        print('files=', len(all_files))
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
        print(dataset.info())

        df_houses = dataset.loc[(dataset['ob-type-1'] == 'Single Family Home') |
                           (dataset['ob-type-4'] == 'Multi Family') |
                           (dataset['ob-type-3'] == 'Townhouse')]
        print(len(df_houses))

        # what's columns and type of data
        # print(dataset.head())

        # familiarity with particular column
        # pd.set_option('display.float_format', lambda x: '%.1f' % x)
        # print(dataset['quiet-rate'].describe())

        # print(dataset['quiet-rate'].unique())
        # print(dataset['lot-size'].value_counts())
        # print(dataset['ob-beds'].isnull().sum(axis=0))
        # dataset.hist(column='ob-beds')
        # plt.show()

        # top city
        # print('Cities =', dataset['dist-city'].value_counts())

        # city_data = city_data.sort_values('dist-city, dist-name')
        # arr= city_data['dist-name'].unique()
        # arr = city_data.drop_duplicates(subset=['dist-city', 'dist-name'], keep="first")
        # arr = arr.sort_values('dist-city')
        # arr = arr.loc[(arr['dist-city'] == 'Burien')]
        #
        # arr = arr[['dist-city', 'dist-name']]
        # print(arr)
        # debug = dataset.loc[(dataset['dist-name']=='Houghton') & (dataset['dist-city'] == 'Kirkland')]
        # print(debug)

    def _get_df_houses(self, df_city_loaded):
        print(df_city_loaded.info())

        df_houses = self._remove_duplicates(df_city_loaded)
        # df_houses = self._remove_apartments(df_houses)
        df_houses = self._remove_place_errors(df_houses)

        df_houses = self._convert_text_values(df_houses)
        df_houses = self._fill_missed_values(df_houses)
        df_houses = self._create_custom_features(df_houses)

        df_houses_short = self._get_short_df(df_houses)
        # expplore_df = df_houses_short.loc[(df_houses_short['dist-city'] == 'Seattle') & (df_houses_short['dist-name'] == 'Westlake')]
        #
        # pd.set_option('display.float_format', lambda x: '%.1f' % x)
        # print(expplore_df['fin-price'].describe())

         # print(df_houses_short['fin-price'].describe())
        # self._define_correlations(df_houses_short)

        return df_houses

    def _prepare_data(self, dataset):

        original_df = self._remove_duplicates(dataset)
        original_df = self._remove_place_errors(original_df)
        original_df = self._get_rows_with_district_params(original_df)
        #p_data = self._fix_place_errors(p_data)
        original_df = self._add_district_params(original_df)
        original_df = self._convert_text_value_to_numbers(original_df)
        # original_df = self._remove_rudiment_columns(original_df)
        original_df = self._fill_nan_values(original_df)

        normalized_df = self._get_normalized_df(original_df)

        return original_df, normalized_df


    @staticmethod
    def _remove_duplicates(dataset):
        return dataset.drop_duplicates(subset="property_link-href", keep="first")

    @staticmethod
    def _remove_apartments(dataset):
        df_without_apartments = dataset.loc[(dataset['ob-type-1'] == 'Single Family Home') |
                                (dataset['ob-type-4'] == 'Multi Family') |
                                (dataset['ob-type-3'] == 'Townhouse')]
        return df_without_apartments

    @staticmethod
    def _remove_place_errors(dataset):
        d_with_removed_empty_cities = dataset.dropna(subset=['dist-city'])

        d_with_removed_empty_districts_in_seattle = d_with_removed_empty_cities[
            ((d_with_removed_empty_cities['dist-city'] == 'Seattle') &
            (pd.notna(d_with_removed_empty_cities['dist-name']))) |
            (d_with_removed_empty_cities['dist-city'] != 'Seattle')
        ]
        return d_with_removed_empty_districts_in_seattle

    @staticmethod
    def _get_rows_with_district_params(dataset):
        # list of districts that we know its params
        rows_with_district_params = dataset.loc[
            (dataset['dist-city'] == 'Seattle') |
            (dataset['dist-city'] == 'Bellevue') |
            (dataset['dist-city'] == 'Burien') |
            (dataset['dist-city'] == 'Black Diamond') |
            (dataset['dist-city'] == 'Bothell') |
            (dataset['dist-city'] == 'Auburn')
            ]
        return rows_with_district_params

    @staticmethod
    def _fix_place_errors(dataset):
        # perhaps it isn't necessary
        # Seattle Boulevard Park -> Burien Boulevard Park
        mask_1 = (dataset['dist-city'] == 'Seattle') & (dataset['dist-name'] == 'Boulevard Park')
        dataset.loc[mask_1, 'dist-city'] = 'Burien'

        mask_2 = (dataset['dist-city'] == 'Burien') & (dataset['dist-name'] == 'Beverly Park')
        print(dataset.loc[mask_2])
        return dataset

    @staticmethod
    def _remove_rudiment_columns(dataset):
        return dataset.drop(columns=['web-scraper-order', 'web-scraper-start-url', 'property_link',
                                     'property_link-href', 'ob-type-1', 'ob-type-2', 'ob-type-3', 'ob-type-4'])

    def _add_district_params(self, dataset):

        districts_data = self._load_csv_data('districts')

        temp = dataset[['dist-city', 'dist-name']].drop_duplicates()
        print(temp)

        print('Districts =',districts_data.groupby(['dist-city', 'dist-name']).apply(list))
        merged_dataset = pd.merge(left=dataset, right=districts_data,
                                  left_on=['dist-city', 'dist-name'], right_on=['dist-city', 'dist-name'])
        return merged_dataset

    @staticmethod
    def _convert_text_values(df_houses):
        # price
        # input format $1,000
        # output: 1000
        df_houses['fin-price'] = pd.to_numeric((df_houses['fin-price'].replace('[\$,]', '', regex=True)),
                                                 errors='coerce')

        # ob-sqft
        # input 1,000
        # output 1000
        df_houses['ob-sqft'] = pd.to_numeric((df_houses['ob-sqft'].replace('[\,]', '', regex=True)),
                                               errors='coerce')

        # lot-size
        # input - 1,000 sqft
        #       - 1 acres
        # output - 1000
        #        - 43560
        # do it in two steps:
        # 1. convert acres
        # 2. convert sqft

        # choose rows that contains acres
        # convert to numeric and to sqft by multiplying 43560
        df_houses.loc[df_houses['lot-size'].str.contains("acres", na=1), 'lot-size'] = \
            pd.to_numeric((df_houses['lot-size'].replace('[\, acres]', '', regex=True)), errors='coerce') * 43560

        # convert to numeric values, rows that contains sqft
        df_houses['lot-size'] = pd.to_numeric((df_houses['lot-size'].replace('[\, sqft]', '', regex=True)),
                                                errors='coerce')

        return df_houses

    @staticmethod
    def _fill_missed_values(df_houses):
        # most single family houses have 1 or 2 stories, and their proportion is about equal
        # so NaN stories replace for 1 or 2 in random maner
        df_houses['ob-stories'] = df_houses['ob-stories'].fillna(
            pd.Series(np.random.choice([1, 2], size=len(df_houses.index))))

        median_columns = ['fin-price', 'ob-beds', 'ob-bath', 'ob-sqft', 'lot-size']
        for col in median_columns:
            df_houses[col] = df_houses[col].fillna(df_houses[col].median())

        return df_houses

    @staticmethod
    def _create_custom_features(df_houses):
        df_houses.loc[df_houses['ob-type-3'].eq('Townhouse'), 'ob-type'] = 1
        df_houses.loc[df_houses['ob-type-4'].eq('Multi Family'), 'ob-type'] = 2
        df_houses.loc[df_houses['ob-type-1'].eq('Single Family Home'), 'ob-type'] = 3

        return df_houses

    @staticmethod
    def _get_short_df(df_houses):

        df_houses_short = df_houses[['dist-city', 'dist-name', 'fin-price', 'ob-type', 'ob-beds', 'ob-bath', 'ob-sqft', 'lot-size']]
        return df_houses_short

    @staticmethod
    def _define_correlations(df_houses_short):
        corr_matrix = df_houses_short.corr()
        print(corr_matrix['fin-price'].sort_values(ascending=False))

    @staticmethod
    def _convert_text_value_to_numbers(original_df):
        # price
        # input format $1,000
        # output: 1000
        original_df['fin-price'] = pd.to_numeric((original_df['fin-price'].replace('[\$,]', '', regex=True)),
                                                 errors='coerce')

        # ob-sqft
        # input 1,000
        # output 1000
        original_df['ob-sqft'] = pd.to_numeric((original_df['ob-sqft'].replace('[\,]', '', regex=True)),
                                                 errors='coerce')

        # lot-size
        # input - 1,000 sqft
        #       - 1 acres
        # output - 1000
        #        - 43560
        # do it in two steps:
        # 1. convert acres
        # 2. convert sqft

        # choose rows that contains acres
        # convert to numeric and to sqft by multiplying 43560
        original_df.loc[original_df['lot-size'].str.contains("acres", na=1), 'lot-size'] = \
            pd.to_numeric((original_df['lot-size'].replace('[\, acres]', '', regex=True)), errors='coerce') * 43560

        # convert to numeric values, rows that contains sqft
        original_df['lot-size'] = pd.to_numeric((original_df['lot-size'].replace('[\, sqft]', '', regex=True)),
                                                                       errors='coerce')

        # for lot data NaN change to 0
        # because there are houses without lots, and they have 0 (NaN) lot-size
        original_df['lot-size'] = original_df['lot-size'].fillna(0)


        # most single family houses have 1 or 2 stories, and their proportion is about equal
        # so NaN stories replace for 1 or 2 in random maner
        original_df['ob-stories'] = original_df['ob-stories'].fillna(pd.Series(np.random.choice([1, 2], size=len(original_df.index))))

        # for NaN ob-type replace to Single Family Home as they the most number
        original_df.loc[
            (original_df['ob-type-1'].isnull()) &
            (original_df['ob-type-2'].isnull()) &
            (original_df['ob-type-3'].isnull()) &
            (original_df['ob-type-4'].isnull()), 'ob-type-1'] = 'Single Family Home'


        # кастомные категоризация для домов. учитывается комбинация тип жилья + его этажность (для Single Family)
        original_df.loc[original_df['ob-type-2'].eq('Condo'), 'ob-type'] = 1
        original_df.loc[original_df['ob-type-3'].eq('Townhouse'), 'ob-type'] = 2
        original_df.loc[original_df['ob-type-4'].eq('Multi Family'), 'ob-type'] = 3
        original_df.loc[(original_df['ob-type-1'].eq('Single Family Home')) & (original_df['ob-stories'] == 1), 'ob-type'] = 4
        original_df.loc[(original_df['ob-type-1'].eq('Single Family Home')) & (original_df['ob-stories'] == 2), 'ob-type'] = 5
        original_df.loc[(original_df['ob-type-1'].eq('Single Family Home')) & (original_df['ob-stories'] > 2), 'ob-type'] = 6

        # commute-rate, crime-rate, outdoor-activities-rate have Letter type categorisation
        # convert it into numeric
        original_df = original_df.replace(regex=r'^D', value=1)
        original_df = original_df.replace(regex=r'^C', value=2)
        original_df = original_df.replace(regex=r'^B', value=3)
        original_df = original_df.replace(regex=r'^A', value=4)

        original_df.loc[original_df['ob-dining-room'].eq('Dining Room'), 'ob-dining-room'] = 2
        original_df.loc[original_df['ob-walk-in-closet'].eq('Walk In Closet'), 'ob-walk-in-closet'] = 1
        original_df.loc[original_df['ob-laundry-room'].eq('Laundry Room'), 'ob-laundry-room'] = 1
        original_df.loc[original_df['ob-basement'].str.contains(pat="Bas", na=1), 'ob-basement'] = 1

        original_df.loc[original_df['quiet-rate'] <= 40, 'quiet-rate'] = 1
        original_df.loc[(original_df['quiet-rate'] > 40) & (original_df['quiet-rate'] <= 70), 'quiet-rate'] = 2
        original_df.loc[original_df['quiet-rate'] > 70, 'quiet-rate'] = 3

        original_df.loc[original_df['distance-downtown'] <= 20, 'distance-downtown'] = 4
        original_df.loc[(original_df['distance-downtown'] > 20) & (original_df['distance-downtown'] <= 40), 'distance-downtown'] = 3
        original_df.loc[(original_df['distance-downtown'] > 40) & (original_df['distance-downtown'] <= 60), 'distance-downtown'] = 2
        original_df.loc[original_df['distance-downtown'] > 60, 'distance-downtown'] = 1

        original_df.loc[original_df['ob-sqft'] <= 1076, 'ob-sqft'] = 1
        original_df.loc[(original_df['ob-sqft'] > 1076) & (original_df['ob-sqft'] <= 1615), 'ob-sqft'] = 2 # S > 100 and < 150 m2
        original_df.loc[(original_df['ob-sqft'] > 1615) & (original_df['ob-sqft'] <= 2153), 'ob-sqft'] = 3 #  S > 150 and < 200 m2
        original_df.loc[original_df['ob-sqft'] > 2153, 'ob-sqft'] = 4 # S > 200m2

        original_df.loc[original_df['lot-size'] == 0, 'lot-size'] = 1
        original_df.loc[(original_df['lot-size'] > 1) & (original_df['lot-size'] <= 3441), 'lot-size'] = 2
        original_df.loc[(original_df['lot-size'] > 3441) & (original_df['lot-size'] <= 6004.75), 'lot-size'] = 3
        original_df.loc[original_df['lot-size'] > 6004.75, 'lot-size'] = 4

        # change schools from 10 grades to 4
        # schools_rate_medium = original_df.groupby(['elem-schools-rate', 'COL2'])[['COL3','COL4']].apply(np.median)
        original_df['schools-rate'] = np.median([
            original_df['elem-schools-rate'],
            original_df['middle-schools-rate'],
            original_df['high-schools-rate'],
        ], axis=0)

        # original_df.hist(column='lot-size')
        # plt.show()
        # print(original_df[['schools-rate', 'elem-schools-rate', 'middle-schools-rate']])
        # print(original_df['schools-rate'].describe())
        # print(original_df['lot-size'].unique())
        # print(original_df['ob-walk-in-closet'].unique())

        return original_df

    @staticmethod
    def _fill_nan_values(original_df):
        # fill nan values
        # using median because fin-price is skewed data
        median_columns = ['fin-price', 'ob-beds', 'ob-bath', 'ob-sqft']
        for col in median_columns:
            original_df[col] = original_df[col].fillna(original_df[col].median())



        # 1 is because the most object is the 1 type
        # original_df['ob-type'] = original_df['ob-type'].fillna(1)
        # 2 is because the mean for ob-stories
        # original_df['ob-stories'] = original_df['ob-stories'].fillna(2)

        # I don't know why by 0 in Excel is NaN id Dataframe

        original_df['private-schools-num'] = original_df['private-schools-num'].fillna(1)

        # 0 - is equal to False for house feature
        original_df['ob-dining-room'] = original_df['ob-dining-room'].fillna(1)


        return original_df

    @staticmethod
    def _get_normalized_df(original_df):
        '''
            pre-processing data for k-mean segmentation:
            - transform skewed data with log tranasformation
            - normalize data
        '''
        necessary_columns = ['fin-price', 'ob-beds', 'ob-bath', 'ob-sqft', 'lot-size', 'ob-type', 'ob-dining-room',
                             'distance-downtown', 'commute-rate', 'crime-rate', 'dog-friendly-rate', 'quiet-rate',
                             'schools-rate']
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

        # take only necessary columns
        normalized_df = normalized_df[necessary_columns]

        return normalized_df

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--op", required=True, help="Operation type")
        # ap.add_argument("--fo", required=True, help="Operation type with data file.")
        ap.add_argument("--city_dir", required=False)
        args = vars(ap.parse_args())
        return args


# ----------------------------------------------------
data_o = Data()
data_o.run()
