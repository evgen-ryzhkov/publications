"""
Driving license reader

Input:
    - Photo of driving license (it could be holding by hand)
Output:
    - Scan of the document: must to have rather pretty view (approximately like it usually have scanned document view)
    - Text data: First, Last Names (Latin letters), Birth date, driving license ID number.

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


class Data:

    def run(self):
        args = self._get_command_line_arguments()

        # load data
        if args['op'] == 'parse_data':
            city_data = self._load_csv_data(args['city_dir'])

            # self._familiarity_with_data(city_data)
            original_df, normalized_df = self._prepare_data(city_data)
            # self._familiarity_with_data(original_df)
            segmented_df = get_segments(original_df, normalized_df)
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

        # what's columns and type of data
        # print(dataset.head())

        # familiarity with particular column
        # pd.set_option('display.float_format', lambda x: '%.1f' % x)
        # print(dataset['crime-rate'].describe())
        print(dataset['crime-rate'].unique())
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

    def _prepare_data(self, dataset):

        original_df = self._remove_duplicates(dataset)
        original_df = self._remove_place_errors(original_df)
        original_df = self._get_rows_with_district_params(original_df)
        # p_data = self._fix_place_errors(p_data)
        original_df = self._add_district_params(original_df)
        original_df = self._convert_text_value_to_numbers(original_df)
        original_df = self._remove_rudiment_columns(original_df)
        original_df = self._fill_nan_values(original_df)

        normalized_df = self._get_normalized_df(original_df)

        return original_df, normalized_df


    @staticmethod
    def _remove_duplicates(dataset):
        return dataset.drop_duplicates(subset="property_link-href", keep="first")

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
        merged_dataset = pd.merge(left=dataset, right=districts_data,
                                  left_on=['dist-city', 'dist-name'], right_on=['dist-city', 'dist-name'])
        return merged_dataset

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

        # ob-type
        # creating custorm column with numeric categories
        # - Single Family Home -> 1
        # - Condo              -> 4
        # - Townhouse          -> 3
        # - Multi Family       -> 2
        original_df.loc[original_df['ob-type-1'].eq('Single Family Home'), 'ob-type'] = 1
        original_df.loc[original_df['ob-type-2'].eq('Condo'), 'ob-type'] = 4
        original_df.loc[original_df['ob-type-3'].eq('Townhouse'), 'ob-type'] = 3
        original_df.loc[original_df['ob-type-4'].eq('Multi Family'), 'ob-type'] = 2

        # commute-rate, crime-rate, outdoor-activities-rate have Letter type categorisation
        # convert it into numeric
        original_df = original_df.replace(regex=r'^D-$', value=0)
        original_df = original_df.replace(regex=r'^D$', value=1)
        original_df = original_df.replace(regex=r'^D\+$', value=2)
        original_df = original_df.replace(regex=r'^C-$', value=3)
        original_df = original_df.replace(regex=r'^C$', value=4)
        original_df = original_df.replace(regex=r'^C\+$', value=5)
        original_df = original_df.replace(regex=r'^B-$', value=6)
        original_df = original_df.replace(regex=r'^B$', value=7)
        original_df = original_df.replace(regex=r'^B\+$', value=8)
        original_df = original_df.replace(regex=r'^A-$', value=9)
        original_df = original_df.replace(regex=r'^A$', value=10)
        original_df = original_df.replace(regex=r'^A\+$', value=11)

        original_df.loc[original_df['ob-dining-room'].eq('Dining Room'), 'ob-dining-room'] = 1
        original_df.loc[original_df['ob-walk-in-closet'].eq('Walk In Closet'), 'ob-walk-in-closet'] = 1
        original_df.loc[original_df['ob-laundry-room'].eq('Laundry Room'), 'ob-laundry-room'] = 1
        original_df.loc[original_df['ob-basement'].str.contains(pat="Bas", na=1), 'ob-basement'] = 1

        # print('-walk-in-closet')
        # print(original_df['ob-walk-in-closet'].describe())
        # print(original_df['ob-walk-in-closet'].unique())
        print('ob-basement')
        print(original_df['ob-basement'].unique())
        # print('ob-laundry-room')
        # print(original_df['ob-laundry-room'].describe())
        # print(original_df['ob-laundry-room'].unique())

        return original_df

    @staticmethod
    def _fill_nan_values(original_df):
        # fill nan values
        # using median because fin-price is skewed data
        median_columns = ['fin-price', 'ob-beds', 'ob-bath', 'ob-sqft']
        for col in median_columns:
            original_df[col] = original_df[col].fillna(original_df[col].median())

        # for lot data NaN change to 0
        # because there are houses without lots, and they have 0 (NaN) lot-size
        # fillna 1 instead of 0 because normalize give an error
        original_df['lot-size'] = original_df['lot-size'].fillna(1)

        # 1 is because the most object is the 1 type
        original_df['ob-type'] = original_df['lot-size'].fillna(1)
        # 2 is because the mean for ob-stories
        original_df['ob-stories'] = original_df['lot-size'].fillna(2)

        # I don't know why by 0 in Excel is NaN id Dataframe
        original_df['middle-schools-rate'] = original_df['lot-size'].fillna(0)
        original_df['high-schools-rate'] = original_df['lot-size'].fillna(0)
        original_df['private-schools-num'] = original_df['lot-size'].fillna(0)

        # 0 - is equal to False for house feature
        original_df['ob-dining-room'] = original_df['ob-dining-room'].fillna(0)


        return original_df

    @staticmethod
    def _get_normalized_df(original_df):
        '''
            pre-processing data for k-mean segmentation:
            - transform skewed data with log tranasformation
            - normalize data
        '''
        necessary_columns = ['fin-price', 'ob-beds', 'ob-bath', 'ob-sqft', 'lot-size', 'ob-type', 'ob-stories',
                             'distance-downtown', 'commute-rate', 'crime-rate', 'dog-friendly-rate', 'quiet-rate',
                             'elem-schools-rate', 'middle-schools-rate', 'high-schools-rate', 'private-schools-num']
        normalized_df = original_df.copy()
        min_max_scaler = MinMaxScaler()

        for col in necessary_columns:
            # Transform Skewed Data
            normalized_df[col] = np.log(normalized_df[col])

            # normalize data
            normalized_df[[col]] = min_max_scaler.fit_transform(normalized_df[[col]])

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
