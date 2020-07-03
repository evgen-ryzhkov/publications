'''
    Creating neighborhoods characteristics by its name

    Input:
        - Seattle sold properties
        - Seattle districts ratings
    Output:


    Written by Evgeniy Ryzhkov

    ------------------------------------------------------------

    Usage:

        # parse data
        create empty neighborhoods.csv: python -m scripts.neighborhoods.py --op=create_empty --donor_file_path=data/real_estate/ --empty_file_path=data/neighborhoods/empty.csv

'''

import os
import numpy as np
import pandas as pd
import glob
import re
import argparse


class Neighborhoods:

    def run(self):
        args = self._get_command_line_arguments()

        if args['op'] == 'create_empty':
            df_real_estate = self._load_csv_data(args['donor_file_path'])
            self._create_empty_csv(df_real_estate, args['empty_file_path'])

    @staticmethod
    def _create_empty_csv(df, empty_file_path):
        '''
            using donor file take neighborhood names
            create csv file with unique pairs of city_name and neighborhood name
        '''
        print('[INFO] Neighborhoods csv creating started...')
        df_neigborhoods = df.copy()
        df_neigborhoods = df_neigborhoods[['city', 'neighborhood']].drop_duplicates()
        print('[INFO] Neighborhoods number=', len(df_neigborhoods))

        # add meaningful features
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
            print('[INFO] Neighborhoods csv was created.')
        except Exception as e:
            print('[ERROR] Something wrong with csv creating. ' + str(e))

    @staticmethod
    def _load_csv_data(dir_path):
        print(dir_path)
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
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--op", required=True, help="Operation type")
        ap.add_argument("--donor_file_path", required=False)
        ap.add_argument("--empty_file_path", required=False)
        args = vars(ap.parse_args())
        return args


# ----------------------------------------------------
neighborhoods_o = Neighborhoods()
neighborhoods_o.run()