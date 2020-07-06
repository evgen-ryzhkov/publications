'''
    Creating neighborhoods characteristics by its name

    Input:
        - Seattle sold properties
        - Seattle districts ratings
    Output:


    Written by Evgeniy Ryzhkov

    ------------------------------------------------------------

    Usage:

        create empty neighborhoods.csv: python -m scripts.neighborhoods.py --op=create_empty --donor_file_path=data/real_estate/ --csv_file_path=data/neighborhoods/data.csv
        fill with lat/lng:              python -m scripts.neighborhoods.py --op=fill_coord --csv_file_path=data/neighborhoods/data.csv
        fill driving distances:         python -m scripts.neighborhoods.py --op=fill_distances --feature=driving --csv_file_path=data/neighborhoods/data.csv
        fill transit distances:         python -m scripts.neighborhoods.py --op=fill_distances --feature=transit --csv_file_path=data/neighborhoods/data.csv
        fill schools:                   python -m scripts.neighborhoods.py --op=fill_places --feature=school --csv_file_path=data/neighborhoods/data.csv
        fill supermarkets:              python -m scripts.neighborhoods.py --op=fill_places --feature=supermarket --csv_file_path=data/neighborhoods/data.csv
        fill parks:                     python -m scripts.neighborhoods.py --op=fill_places --feature=park --csv_file_path=data/neighborhoods/data.csv
        fill cafe:                      python -m scripts.neighborhoods.py --op=fill_places --feature=cafe --csv_file_path=data/neighborhoods/data.csv
        fill restaurant:                python -m scripts.neighborhoods.py --op=fill_places --feature=restaurant --csv_file_path=data/neighborhoods/data.csv
        fill bar:                       python -m scripts.neighborhoods.py --op=fill_places --feature=bar --csv_file_path=data/neighborhoods/data.csv
        fill night_club:                python -m scripts.neighborhoods.py --op=fill_places --feature=night_club --csv_file_path=data/neighborhoods/data.csv
        fill movie_theater:             python -m scripts.neighborhoods.py --op=fill_places --feature=movie_theater --csv_file_path=data/neighborhoods/data.csv
        add new neighborhoods:          python -m scripts.neighborhoods.py --op=add_neighborhoods --donor_file_path=data/real_estate/ --csv_file_path=data/neighborhoods/data.csv
        check and fix problem rows:     python -m scripts.neighborhoods.py --op=problems_analysis --csv_file_path=data/neighborhoods/data.csv
        transform df:                   python -m scripts.neighborhoods.py --op=transform_df --csv_file_path=data/neighborhoods/data.csv

'''

import os
import numpy as np
import pandas as pd
import glob
import re
import argparse
from settings.secrets import API_KEY
import requests


class Neighborhoods:

    def __init__(self):
        self.API_KEY_S = '@key=' + API_KEY

    def run(self):
        args = self._get_command_line_arguments()

        if args['op'] == 'create_empty':
            df_real_estate = self._load_csv_data(args['donor_file_path'])
            self._create_empty_csv(df_real_estate, args['csv_file_path'])

        if args['op'] == 'add_neighborhoods':
            df_real_estate = self._load_csv_data(args['donor_file_path'])
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._add_neighborhoods(df_neighborhoods, df_real_estate, args['csv_file_path'])

        if args['op'] == 'fill_coord':
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._fill_csv_with_coord(df_neighborhoods, args['csv_file_path'])

        if args['op'] == 'fill_distances':
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._fill_csv_with_distances(df_neighborhoods, args['feature'], args['csv_file_path'])

        if args['op'] == 'fill_places':
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._fill_csv_with_place_info(df_neighborhoods, args['feature'], args['csv_file_path'])

        if args['op'] == 'problems_analysis':
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._check_problems(df_neighborhoods, args['csv_file_path'])

        if args['op'] == 'transform_df':
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._transform_df(df_neighborhoods, args['csv_file_path'])

    def _create_empty_csv(self, df, empty_file_path):
        '''
            using donor file take neighborhood names
            create csv file with unique pairs of city_name and neighborhood name
        '''
        print('[INFO] Neighborhoods csv creating was started...')
        df_neigborhoods = df.copy()
        df_neigborhoods = df_neigborhoods[['city', 'neighborhood']].drop_duplicates()
        print('[INFO] Neighborhoods number=', len(df_neigborhoods))

        # add meaningful features
        df_neigborhoods['country'] = 'Netherlands'  # hard coding for demo
        df_neigborhoods['lat'] = ''
        df_neigborhoods['lng'] = ''
        df_neigborhoods['commit_time_driving'] = ''
        df_neigborhoods['commit_time_transit'] = ''
        df_neigborhoods['school_num'] = ''
        df_neigborhoods['cafe_restaurant_num'] = ''
        # bar, night clubs, movie_theatre
        df_neigborhoods['night_life_num'] = ''
        df_neigborhoods['supermarket_num'] = ''
        df_neigborhoods['park_num'] = ''

        self._save_csf_file(df_neigborhoods, empty_file_path)

    def _add_neighborhoods(self, df_neighborhoods, df_real_estate, data_csv_path):
        print('[INFO] Updating neighborhoods csv creating was started...')
        # TODO code dublicate, need refactoring
        df_new_neigborhoods = df_real_estate.copy()
        df_new_neigborhoods = df_new_neigborhoods[['city', 'neighborhood']].drop_duplicates()
        df_new_neigborhoods['country'] = 'Netherlands'
        df_new_neigborhoods['lat'] = ''
        df_new_neigborhoods['lng'] = ''
        df_new_neigborhoods['commit_time_driving'] = ''
        df_new_neigborhoods['commit_time_transit'] = ''
        df_new_neigborhoods['school_num'] = ''
        df_new_neigborhoods['cafe_restaurant_num'] = ''
        df_new_neigborhoods['night_life_num'] = ''
        df_new_neigborhoods['supermarket_num'] = ''
        df_new_neigborhoods['park_num'] = ''
        # / to do

        # append new neighborhoods at the end
        df_merged = pd.concat([df_neighborhoods, df_new_neigborhoods])
        df_merged = df_merged.drop_duplicates(subset=['city', 'neighborhood'], keep='first')

        self._save_csf_file(df_merged, data_csv_path)

    def _fill_csv_with_coord(self, df_neighborhoods, data_csv_path):
        print('[INFO] Filling coordinates was started...')
        GEOCODING_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json?'
        problem_rows = []

        empty_coord_df = df_neighborhoods.loc[(pd.isna(df_neighborhoods['lat'])) | (pd.isna(df_neighborhoods['lng']))]
        empty_coord_rows_num = len(empty_coord_df)
        processing_count = 1

        for index, row in empty_coord_df.iterrows():
            # we need only rows with empty lat or lng
            # TODO it seems there is no need this condition anymore
            if (pd.isna(row['lat'])) | (pd.isna(row['lat'])):
                print('[INFO] Processing ' + str(processing_count) + ' from ' + str(empty_coord_rows_num), end='\r', flush=True)
                processing_count += 1
                address = str(row['city']) + ',' + str(row['neighborhood']) + ',' + str(row['country'])

                params_geocoding = {
                    'key': API_KEY,
                    'address': address
                }
                try:
                    response = requests.get(GEOCODING_API_URL, params=params_geocoding)
                    response.raise_for_status()
                    response_json = response.json()
                    df_neighborhoods.loc[index, 'lat'] = response_json['results'][0]['geometry']['location']['lat']
                    df_neighborhoods.loc[index, 'lng'] = response_json['results'][0]['geometry']['location']['lng']
                except:
                    # if error then it's maybe because of neighborhood
                    # try just city
                    address = str(row['city']) + ',' + str(row['country'])
                    params_geocoding = {
                        'key': API_KEY,
                        'address': address
                    }
                    try:
                        response = requests.get(GEOCODING_API_URL, params=params_geocoding)
                        response.raise_for_status()
                        response_json = response.json()
                        df_neighborhoods.loc[index, 'lat'] = response_json['results'][0]['geometry']['location']['lat']
                        df_neighborhoods.loc[index, 'lng'] = response_json['results'][0]['geometry']['location']['lng']

                    except:
                        # if there is still error print problem rows number and go further
                        problem_rows.append(index)
                        continue

        self._save_csf_file(df_neighborhoods, data_csv_path)
        if len(problem_rows) > 0:
            print('[ERROR] There were problem rows:', problem_rows)
        else:
            print('[OK] There were not any problem rows.')

    def _fill_csv_with_distances(self, df_neighborhoods, type, data_csv_path):
        print('[INFO] Filling distances was started...')
        DISTANCE_API_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json?'

        # destination is Amsterdam city center
        DESTINATION = 'Amsterdam+Centraal+railway+station,+Stationsplein,+1012+AB+Amsterdam,+Netherlands'

        problem_rows = []

        if type == 'driving':
            col_name = 'commit_time_driving'
            api_type = 'driving'
        elif type == 'transit':
            col_name = 'commit_time_transit'
            api_type = 'transit'

        empty_distances_df = df_neighborhoods.loc[pd.isna(df_neighborhoods[col_name])]
        empty_distances_rows_num = len(empty_distances_df)
        processing_count = 1

        for index, row in empty_distances_df.iterrows():
            print('[INFO] Processing ' + str(processing_count) + ' from ' + str(empty_distances_rows_num), end='\r',
                  flush=True)
            processing_count += 1
            origins = str(row['lat']) + ',' + str(row['lng'])

            params = {
                'language': 'en-EN',
                'mode': api_type,
                'key': API_KEY,
                'destinations': DESTINATION,
                'origins': origins
            }

            try:
                response = requests.get(DISTANCE_API_URL, params=params)
                response.raise_for_status()
                response_json = response.json()
                df_neighborhoods.loc[index, col_name] =\
                    response_json['rows'][0]['elements'][0]['duration']['text']

            except:
                # if there is still error print problem rows number and go further
                problem_rows.append(index)

        self._save_csf_file(df_neighborhoods, data_csv_path)
        if len(problem_rows) > 0:
            print('[ERROR] There were problem rows:', problem_rows)
        else:
            print('[OK] There were not any problem rows.')

    def _fill_csv_with_place_info(self, df_neighborhoods, feature_name, data_csv_path):
        print('[INFO] Filling location info was started...')
        PLACE_API_URL = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?'

        problem_rows = []

        if feature_name == 'school':
            col_name = 'school_num'
            api_type = 'school'
        elif feature_name == 'supermarket':
            col_name = 'supermarket_num'
            api_type = 'supermarket'
        elif feature_name == 'park':
            col_name = 'park_num'
            api_type = 'park'
        elif feature_name == 'cafe':
            col_name = 'cafe_num'
            api_type = 'cafe'
        elif feature_name == 'restaurant':
            col_name = 'restaurant_num'
            api_type = 'restaurant'
        elif feature_name == 'bar':
            col_name = 'bar_num'
            api_type = 'bar'
        elif feature_name == 'night_club':
            col_name = 'night_clubs_num'
            api_type = 'night_club'
        elif feature_name == 'movie_theater':
            col_name = 'movie_theatres_num'
            api_type = 'movie_theater'

        empty_places_df = df_neighborhoods.loc[(pd.isna(df_neighborhoods[col_name]))]
        empty_places_rows_num = len(empty_places_df)
        processing_count = 1

        for index, row in empty_places_df.iterrows():
            print('[INFO] Processing ' + str(processing_count) + ' from ' + str(empty_places_rows_num), end='\r',
                  flush=True)
            processing_count += 1

            origins = str(row['lat']) + ',' + str(row['lng'])
            params_school = {
                'language': 'en-EN',
                'key': API_KEY,
                'type': api_type,
                'location': origins,
                'radius': 1000
            }

            try:
                response = requests.get(PLACE_API_URL, params=params_school)
                response.raise_for_status()
                response_json = response.json()
                df_neighborhoods.loc[index, col_name] = len(response_json['results'])
            except:
                problem_rows.append(index)
                continue

        self._save_csf_file(df_neighborhoods, data_csv_path)
        if len(problem_rows) > 0:
            print('[ERROR] There were problem rows:', problem_rows)
        else:
            print('[OK] There were not any problem rows.')

    def _check_problems(self, df_neighborhoods, data_csv_path):
        # analysis of problem rows
        print(df_neighborhoods.loc[935])
        #  [33, 560, 595, 871, 921, 935, 952, 1038, 1174, 1196, 1382, 1404, 1478, 1484, 1493, 1537, 1597, 1621, 1632, 1638, 1639, 1640, 1661, 1668, 1731, 1733, 1747, 1775, 1780, 1789, 1801, 1821, 1824, 1837, 1840, 1846, 1884, 1896, 1925, 1931, 1963, 1967, 1973, 1979, 1984, 1987, 1989, 1991, 2035, 2040, 2072, 2080, 2100, 2108, 2117,
        # 2131, 2136, 2147, 2148, 2192, 2211, 2216, 2218, 2220, 2223, 2252, 2262, 2264, 2266, 2268, 2272, 2286, 2288, 2292, 2293, 2294]
        # fix problems
        # df_fixed = df_neighborhoods.drop([df_neighborhoods.index[2233]])
        # df_neighborhoods.iloc[921, df_neighborhoods.columns.get_loc('commit_time_transit')] = '2 hour'
        # self._save_csf_file(df_neighborhoods, data_csv_path)

    def _transform_df(self, df_neighborhoods, data_csv_path):
        # rename columns
        df_neighborhoods.rename(columns={
            'cafe_restaurant_num': 'cafe_num',
            'night_life_num': 'restaurant_num',
        }, inplace=True)

        # add new columns
        df_neighborhoods['bar_num'] = ''
        df_neighborhoods['night_clubs_num'] = ''
        df_neighborhoods['movie_theatres_num'] = ''

        self._save_csf_file(df_neighborhoods, data_csv_path)

    @staticmethod
    def _load_csv_data(dir_path):
        # csv files with city data contains of many separate files
        all_files = glob.glob(os.path.join(dir_path + "/*.csv"))
        li = []
        try:
            for filename in all_files:
                df = pd.read_csv(filename, index_col=None, header=0)
                li.append(df)

            frame = pd.concat(li, axis=0, ignore_index=True)
            return frame

        except FileNotFoundError:
            raise ValueError('[ERROR] CSV file not found!')
        except:
            raise ValueError('[ERROR] Something wrong with loading of CSV file!')

    @staticmethod
    def _load_csv_file(file_path):
        try:
            df = pd.read_csv(file_path, index_col=None, header=0)
            return df
        except FileNotFoundError:
            raise ValueError('[ERROR] CSV file not found!')
        except:
            raise ValueError('[ERROR] Something wrong with loading of CSV file!')

    @staticmethod
    def _save_csf_file(df, data_csv_path):
        # update csv file
        try:
            df.to_csv(data_csv_path, encoding='utf-8', index=False)
            print('[OK] Neighborhoods csv was saved.')
        except Exception as e:
            print('[ERROR] Something wrong with csv saving. ' + str(e))

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--op", required=True, help="Operation type")
        ap.add_argument("--donor_file_path", required=False)
        ap.add_argument("--csv_file_path", required=False)
        ap.add_argument("--feature", required=False)
        args = vars(ap.parse_args())
        return args


# ----------------------------------------------------
neighborhoods_o = Neighborhoods()
neighborhoods_o.run()