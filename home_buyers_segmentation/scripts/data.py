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

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler




class Data:

    def run(self):
        args = self._get_command_line_arguments()

        # run parsing data
        if args['op'] == 'parse_data':
            city_data = self._load_csv_data(args['city_dir'])

            # self._familiarity_with_data(city_data)
            prepared_data = self._prepare_data(city_data)
            self._familiarity_with_data(prepared_data)

    def _load_csv_data(self, city_dir):
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
        print('Dataset size', dataset.shape)
        # print(dataset.info())

        # what's columns and type of data
        # print(dataset.head())

        # familiarity with particular column
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        print(dataset['fin-price'].describe())
        dataset.hist(column='fin-price')
        plt.show()

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
        p_data = self._remove_duplicates(dataset)
        p_data = self._remove_place_errors(p_data)
        p_data = self._get_rows_with_district_params(p_data)
        # p_data = self._fix_place_errors(p_data)

        p_data = self._remove_rudiment_columns(p_data)
        p_data = self._add_district_params(p_data)

        p_data = self.prepare_data(p_data)
        # p_data = self._strings_to_numbers(p_data)

        return p_data

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
                                     'property_link-href'])

    def _add_district_params(self, dataset):
        districts_data = self._load_csv_data('districts')
        merged_dataset = pd.merge(left=dataset, right=districts_data,
                                  left_on=['dist-city', 'dist-name'], right_on=['dist-city', 'dist-name'])
        return merged_dataset

    def prepare_data(self, dataset):

        prepared_data = self._prepare_prices(dataset)

        return prepared_data

    @staticmethod
    def _prepare_prices(dataset):
        # convert value to numer format
        dataset['fin-price'] = pd.to_numeric((dataset['fin-price'].replace('[\$,]', '', regex=True)),
                                               errors='coerce')
        # fill nan values
        # using median because fin-price is skewed data
        dataset['fin-price'] = dataset['fin-price'].fillna(dataset['fin-price'].median())

        # Transform Skewed Data
        dataset['fin-price'] = np.log(dataset['fin-price'])

        # normalize data
        min_max_scaler = MinMaxScaler()
        dataset[['fin-price']] = min_max_scaler.fit_transform(dataset[['fin-price']])

        return dataset


    @staticmethod
    def _strings_to_numbers(dataset):
        # prices
        dataset['house-price'] = pd.to_numeric((dataset['house-price'].replace('[\$,]', '', regex=True)), errors='coerce')
        dataset['house-price-sqft'] = pd.to_numeric((dataset['house-price-sqft'].replace('[\$,]', '', regex=True)), errors='coerce')
        dataset['monthly-cost'] = pd.to_numeric((dataset['monthly-cost'].replace('[\$,]', '', regex=True)), errors='coerce')
        dataset['principal-interest'] = pd.to_numeric((dataset['principal-interest'].replace('[\$,/mo]', '', regex=True)), errors='coerce')
        dataset['property-taxes'] = pd.to_numeric((dataset['property-taxes'].replace('[\$,/mo]', '', regex=True)), errors='coerce')
        dataset['home-insurance'] = pd.to_numeric((dataset['home-insurance'].replace('[\$,/mo]', '', regex=True)), errors='coerce')
        dataset['rental-value'] = pd.to_numeric((dataset['rental-value'].replace('[\$,/mo]', '', regex=True)), errors='coerce')

        # simple values
        dataset['house-sqft'] = dataset['house-sqft'].replace(' ', '')

        # dataset['house-sqft'] = (dataset['house-sqft'].replace(' ', '')).astype(float)


        return dataset



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
