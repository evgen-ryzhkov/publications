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
    parse data: python -m scripts.data.py --op=parse_data --fo=update --region=[reiong name from dictionary]

"""

import argparse


class Data:

    def run(self):
        args = self._get_command_line_arguments()

        # run parsing data
        if args['op'] == 'parse_data':
            self._parse_data(args['region'])

    # parsing data from zillow
    def _parse_data(self, region):
        print('[INFO] Parsing started...')

        # regiong urls dictionary
        REGION_URLS = {
            'seattle': 'https://www.zillow.com/homes/for_sale/house_type/?searchQueryState=%7B%22pagination%22%3A%7B%7D%2C%22usersSearchTerm%22%3A%22Seattle%2C%20WA%22%2C%22mapBounds%22%3A%7B%22west%22%3A-124.63987601213681%2C%22east%22%3A-119.66856253557431%2C%22south%22%3A46.58769550209506%2C%22north%22%3A48.79469225492623%7D%2C%22isMapVisible%22%3Atrue%2C%22mapZoom%22%3A9%2C%22filterState%22%3A%7B%22con%22%3A%7B%22value%22%3Afalse%7D%2C%22sort%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22land%22%3A%7B%22value%22%3Afalse%7D%2C%22manu%22%3A%7B%22value%22%3Afalse%7D%7D%2C%22isListVisible%22%3Atrue%2C%22customRegionId%22%3A%22a6a2d5ece6X1-CRocj8je9hqnwe_ygbrk%22%7D'
        }
        zillow_url = REGION_URLS[region]



    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--op", required=True, help="Operation type")
        ap.add_argument("--fo", required=True, help="Operation type with data file.")
        ap.add_argument("--region", required=False, help="Region name for parsiong")
        args = vars(ap.parse_args())
        return args


# ----------------------------------------------------
data_o = Data()
data_o.run()
