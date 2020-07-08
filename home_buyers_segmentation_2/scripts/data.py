"""
Home buyers segmentation by property and district features

Input:
    - Seattle sold properties
    - Seattle districts ratings
Output:


Written by Evgeniy Ryzhkov

------------------------------------------------------------

Usage:

    # start overall segmentation
    parse data: python -m scripts.data.py

"""
from .k_means_segmentation import get_number_of_segments, get_segments, validate_cluster_sizes
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
import seaborn as sns


class Data:

    def run(self):

        # load data
        df_city_loaded = self._load_csv_data()

        # general data cleaning
        df_original_cleaned = self._clean_data(df_city_loaded)

        print('[INFO] Object segmentation --------------------------')
        df_object_segments = get_object_segments(df_original_cleaned)
        exit()
        print('[INFO] Price segmentation ---------------------------')
        df_price_segments = get_price_segments(df_original_cleaned)
        print('[INFO] Place segmentation ---------------------------')
        df_place_segments = get_place_segments(df_original_cleaned)

        df_merged = pd.concat([df_object_segments, df_price_segments, df_place_segments], axis=1)
        df_preprocessed = self._preprocess_data(df_merged)

        # to define number of cluster, run this function
        # get_number_of_segments(df_preprocessed)
        num_clusters = 6
        overall_cluster_column_name = 'cluster_overall'
        df_original_overall_segmented = get_segments(df_merged, df_preprocessed, overall_cluster_column_name,
                                                  num_clusters)
        f_validation, df_stat = validate_cluster_sizes(df_original_overall_segmented, overall_cluster_column_name)

        self._profile_clusters(df_original_overall_segmented, df_stat, overall_cluster_column_name, num_clusters)

    @staticmethod
    def _load_csv_data():
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

    @staticmethod
    def _profile_clusters(df_segmented, df_stat, cluster_col_name, n_clusters):

        # define value for calculation distribution for each column
        ob_types = ['Flat', 'House', 'Townhouse']
        ob_storages = ['No storage', 'S Storage', 'M Storage', 'B Storage']
        # ob_energies = ['Good EE', 'Normal EE', 'Poor EE']
        ob_gardens = ['No Garden', 'S Garden', 'M Garden', 'B Garden']
        ob_cars = ['Poor CF', 'Usual CF', 'Good CF', 'VGood CF']
        loc_types = ['City', 'Suburbs']
        place_views = ['No view', 'Good view']
        place_schools = ['Few sch', 'A few of sch', 'A lot of sch']
        place_cafes = ['Few cafe', 'A few of cafe', 'A lot of cafe']
        place_nightlife = ['Poor N_life', 'Good N_life', 'V Good N_life']
        # place_parks = ['None parks', 'A few of parks', 'A lot of parks']
        place_driving = ['>1h driving', '40-60 mins driving', '20-40 mins driving', '<20 mins driving']
        df_profiling_cols = ob_types + ob_storages + ob_gardens + ob_cars + loc_types + place_views +\
                            place_schools + place_cafes + place_nightlife + place_driving

        # initiation of df_profiling
        n_columns = len(df_profiling_cols)
        init_array = np.zeros((n_clusters, n_columns))
        df_profiling = pd.DataFrame(data=init_array, columns=df_profiling_cols)

        # fill df_profiling with real values
        # value - percents each type of features in cluster
        for cluster in range(n_clusters):
            df_cluster = df_segmented.loc[df_segmented[cluster_col_name] == cluster]
            df_cluster_len = len(df_cluster)

            for col in df_profiling_cols:

                # there are different columns in original df
                # for different profiling columns
                # for example: columnn prop_size in df_original
                # transforms into three columns S, M, L in df_profiling
                df_col_name = ''
                if col in ob_types:
                    df_col_name = 'cat_ob_type'
                elif col in ob_storages:
                    df_col_name = 'cat_storage'
                elif col in ob_gardens:
                    df_col_name = 'cat_garden'
                elif col in ob_cars:
                    df_col_name = 'cat_car_friendly'
                elif col in place_views:
                    df_col_name = 'loc_view'
                elif col in loc_types:
                    df_col_name = 'cat_loc'
                elif col in place_schools:
                    df_col_name = 'school_cat'
                elif col in place_cafes:
                    df_col_name = 'cafe_rest_cat'
                elif col in place_nightlife:
                    df_col_name = 'nightlife_cat'
                elif col in place_driving:
                    df_col_name = 'commit_driving_cat'

                percent_val = round((len(df_cluster.loc[df_cluster[df_col_name] == col]) / df_cluster_len) * 100)
                df_profiling.loc[cluster, col] = percent_val

        # mean columns
        df_means = round(
            df_segmented.groupby(cluster_col_name)['cat_living_area', 'ob_bedrooms', 'fin_price'].mean())

        # add distribution values for each cluster
        df_profiling = pd.concat([df_stat, df_profiling, df_means], axis=1, sort=False)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # full screen mode

        # exclude some columns (like means) from heatmap
        mask = np.zeros(df_profiling.shape)
        mask[:, [33, 34, 35]] = True
        cm = sns.light_palette("green")

        sns.heatmap(df_profiling, mask=mask, annot=True, fmt="g", cmap=cm, square=True, xticklabels=True)
        sns.heatmap(df_profiling, alpha=0, cbar=False, annot=True, fmt="g", square=True, xticklabels=True, annot_kws={"color": "black"})

        # sns.heatmap(df_profiling, annot=True, fmt="g", square=True, xticklabels=True)
        plt.show()


# ----------------------------------------------------
data_o = Data()
data_o.run()
