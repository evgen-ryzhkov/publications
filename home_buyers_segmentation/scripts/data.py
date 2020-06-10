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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns




class Data:

    def run(self):
        args = self._get_command_line_arguments()

        # load data
        if args['op'] == 'parse_data':
            city_data = self._load_csv_data(args['city_dir'])

            # self._familiarity_with_data(city_data)
            # data pre-processing
            original_df, normalized_df = self._prepare_data(city_data)
            self._familiarity_with_data(original_df)
            self._get_segments(original_df, normalized_df)

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
        # print('Dataset size', dataset.shape)
        print(dataset.info())

        # what's columns and type of data
        # print(dataset.head())

        # familiarity with particular column
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        # print(dataset['lot-size'].describe())
        print(dataset['lot-size'].value_counts())
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
        original_df = self._remove_rudiment_columns(original_df)
        original_df = self._add_district_params(original_df)
        original_df = self._convert_text_value_to_numbers(original_df)
        original_df = self._fill_nan_values(original_df)

        normalized_df = self._get_normalized_df(original_df)

        return original_df, normalized_df

    def _get_segments(self, df_original, df_normalized):

        # hyperparameter tuning
        # get number of segments - elbow method
        # n_clusters = range(1, 10)
        # inertia = {}
        # inertia_values = []
        #
        # for n in n_clusters:
        #     model = KMeans(
        #         n_clusters=n,
        #         init='k-means++',
        #         max_iter=500,
        #         random_state=42)
        #     model.fit(df_normalized)
        #     inertia[n]=model.inertia_
        #     inertia_values.append(model.inertia_)
        #
        # for key, val in inertia.items():
        #     print(str(key) + ' : ' + str(val))
        #
        # plt.plot(n_clusters, inertia_values, 'bx-')
        # plt.xlabel('Values of K')
        # plt.ylabel('Inertia')
        # plt.title('The Elbow Method using Inertia')
        # plt.show()
        # plot shows that 3 is optimal clusters number

        # run model - get clusters
        kmeans_model = KMeans(n_clusters=3, random_state=1)
        kmeans_model.fit(df_normalized)

        # Extract cluster labels
        cluster_labels = kmeans_model.labels_

        # Create a cluster label column in original dataset
        df_new = df_original.assign(Cluster=cluster_labels)
        # df_temp = df_new[['ob-beds', 'fin-price']]
        # print(df_temp.head())

        # analyze segments --------------------------
        # show clasters stats
        print('[INFO] Clusters stat ------------')
        print(df_new.info())
        df_stat_count = df_new.groupby('Cluster').size()
        print(df_stat_count)
        df_stat_count.plot.bar()
        plt.show()



        # snake plot approach
        # Transform df_normal as df and add cluster column
        df_normalized = pd.DataFrame(df_normalized,
                                         index=df_original.index,
                                         columns=df_original.columns)
        df_normalized['Cluster'] = df_new['Cluster']

        # Melt data into long format
        df_melt = pd.melt(df_normalized.reset_index(),
                          id_vars=['Cluster'],
                          value_vars=['ob-beds', 'fin-price', 'ob-bath', 'ob-sqft', 'lot-size'],
                          var_name='Metric',
                          value_name='Value')

        plt.xlabel('Metric')
        plt.ylabel('Value')
        sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
        plt.show()



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

        return original_df

    @staticmethod
    def _get_normalized_df(original_df):
        '''
            pre-processing data for k-mean segmentation:
            - transform skewed data with log tranasformation
            - normalize data
        '''
        necessary_columns = ['fin-price', 'ob-beds', 'ob-bath', 'ob-sqft', 'lot-size']
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
