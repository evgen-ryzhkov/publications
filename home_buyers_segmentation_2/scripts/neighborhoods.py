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
        add new neighborhoods:          python -m scripts.neighborhoods.py --op=add_neighborhoods --donor_file_path=data/real_estate/ --csv_file_path=data/neighborhoods/data.csv
        check and fix problem rows:     python -m scripts.neighborhoods.py --op=problems_analysis --csv_file_path=data/neighborhoods/data.csv

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

        if args['op'] == 'problems_analysis':
            df_neighborhoods = self._load_csv_file(args['csv_file_path'])
            self._check_problems(df_neighborhoods, args['csv_file_path'])


    @staticmethod
    def _create_empty_csv(df, empty_file_path):
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

        # save csv file
        try:
            df_neigborhoods.to_csv(empty_file_path, encoding='utf-8', index=False)
            print('[OK] Neighborhoods csv was created.')
        except Exception as e:
            print('[ERROR] Something wrong with csv creating. ' + str(e))

    @staticmethod
    def _add_neighborhoods(df_neighborhoods, df_real_estate, data_csv_path):
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

        # update csv file
        try:
            df_merged.to_csv(data_csv_path, encoding='utf-8', index=False)
            print('[OK] Neighborhoods csv was saved.')
        except Exception as e:
            print('[ERROR] Something wrong with csv saving. ' + str(e))

    @staticmethod
    def _fill_csv_with_coord(df_neighborhoods, data_csv_path):
        print('[INFO] Filling coordinates was started...')
        GEOCODING_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json?'
        problem_rows = []

        empty_coord_df = df_neighborhoods.loc[(pd.isna(df_neighborhoods['lat'])) | (pd.isna(df_neighborhoods['lng']))]
        empty_coord_rows_num = len(empty_coord_df)
        processing_count = 1

        for index, row in empty_coord_df.iterrows():
            # we need only rows with empty lat or lng
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

        # update csv file
        try:
            df_neighborhoods.to_csv(data_csv_path, encoding='utf-8', index=False)
            print('[OK] Neighborhoods csv was saved.')
            if len(problem_rows) > 0:
                print('[ERROR] There were problem rows:', problem_rows)
            else:
                print('[OK] There were not any problem rows.')
        except Exception as e:
            print('[ERROR] Something wrong with csv saving. ' + str(e))

    @staticmethod
    def _check_problems(df_neighborhoods, data_csv_path):
        # analysis of problem rows
        #print(df_neighborhoods.loc[2233])

        # fix problems
        df_fixed = df_neighborhoods.drop([df_neighborhoods.index[2233]])

        # update csv file
        try:
            df_fixed.to_csv(data_csv_path, encoding='utf-8', index=False)
            print('[OK] Neighborhoods csv was saved.')
        except Exception as e:
            print('[ERROR] Something wrong with csv saving. ' + str(e))

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
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--op", required=True, help="Operation type")
        ap.add_argument("--donor_file_path", required=False)
        ap.add_argument("--csv_file_path", required=False)
        args = vars(ap.parse_args())
        return args


# ----------------------------------------------------
neighborhoods_o = Neighborhoods()
neighborhoods_o.run()