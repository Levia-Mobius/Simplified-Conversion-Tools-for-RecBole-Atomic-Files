import bz2
import csv
import json
import operator
import os
import re
import time
import gzip
from datetime import datetime
from dateutil import parser
import time
import ast

import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm


from .base_dataset import BaseDataset
from .cosmetics import CosmeticsDataset




class MLUser(BaseDataset):
    def __init__(self, input_path, output_path):
        super(MLUser, self).__init__(input_path, output_path)
        '''
        For MovieLens100K and MovieLens1M
            MovieLens100K: https://grouplens.org/datasets/movielens/100k/
            MovieLens1M: https://grouplens.org/datasets/movielens/1m/
        * user information is available for these 2 datasets
        * I have no idea how to retrieve Movie descriptions via the API for these 2 datasets
        ==========
        Downloaded from MovieLens:
            - 100K inter: u.data; item: u.item; user: u.user
            - 1M inter: ratings.dat; item: movies.dat; user: users.dat    
        ==========
        input_path & dataset_name = ml-100k ; ml-1m 
        output_path = output_data/ml-100k ; output_data/ml-1m
        '''
        if input_path not in ('ml-100k', 'ml-1m'):
            raise ValueError(
                f"Invalid dataset: {input_path}. Allowed options are 'ml-100k' and 'ml-1m'."
            )
            
        
        self.dataset_name = input_path
        inter_name = 'u.data' if self.dataset_name == 'ml-100k' else 'ratings.dat'
        item_name = 'u.item' if self.dataset_name == 'ml-100k' else 'movies.dat'
        user_name = 'u.user' if self.dataset_name == 'ml-100k' else 'users.dat'

        self.inter_file = os.path.join(self.input_path, inter_name)
        self.item_file = os.path.join(self.input_path, item_name)
        self.user_file = os.path.join(self.input_path, user_name)

        self.item_sep = '|' if self.dataset_name == 'ml-100k' else '::'
        self.user_sep = '|' if self.dataset_name == 'ml-100k' else '::'
        self.inter_sep = '\t' if self.dataset_name == 'ml-100k' else '::'

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()


        # selected feature fields
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             2: 'rating:float',
                             3: 'timestamp:float'}
        
        self.item_fields = {0: 'item_id:token',
                            1: 'movie_title:token',
                            2: 'date:float',
                            3: 'class:token_seq'}
        
        self.user_fields = {0: 'user_id:token',
                            1: 'age:token',
                            2: 'gender:token',
                            3: 'occupation:token',
                            4: 'zip_code:token'}
        
    def load_inter_data(self):
       return pd.read_csv(self.inter_file, delimiter=self.inter_sep, header=None, engine='python')

    def load_item_data(self):
        origin_data = pd.read_csv(
            self.item_file, delimiter=self.item_sep, header=None, 
            engine='python', encoding='latin-1'
        )
        processed_data = origin_data.iloc[:, 0:3]
        
        if self.dataset_name == 'ml-100k':
            all_type = ['unkown', 'Action', 'Adventure', 'Animation',
                        'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']
        
            processed_data['class'] = origin_data.iloc[:, 5:].apply(
                lambda row: ', '.join(
                    [all_type[i] for i, v in enumerate(row) if v == 1]
                ),axis=1
            )
            # get the title
            processed_data.iloc[:,1] = processed_data.iloc[:,1].apply(lambda row: (row.rsplit('(', 1))[0].strip())
            # get the unix timstamp
            processed_data.iloc[:,2] = pd.to_datetime(processed_data.iloc[:,2], format='%d-%b-%Y').astype('int64') // 1e9
        else:
            # only for ml-1m
            self.item_fields[2] = 'release_year:token'
            processed_data = self.GeneralProcess(processed_data)       
        return processed_data
        
    def load_user_data(self):
        if self.dataset_name == 'ml-1m':
            self.user_fields[1] = 'gender:token'
            self.user_fields[2] = 'age:token'
        return pd.read_csv(self.user_file, delimiter=self.user_sep, header=None, engine='python')
        

class MLwoUser(BaseDataset):
    def __init__(self, input_path, output_path, meta_info):
        super(MLwoUser, self).__init__(input_path, output_path)
        '''
        For MovieLens10M, MovieLens20M and MovieLens32M
            MovieLens10M: https://grouplens.org/datasets/movielens/10m/
            MovieLens20M: https://grouplens.org/datasets/movielens/20m/
            MovieLens32M: https://grouplens.org/datasets/movielens/32m/
        * user information is not available for these 3 datasets
        
        ==========
        Downloaded from MovieLens:
            - 10M inter: ratings.dat; item: movies.dat
            - 20M & 32M inter: ratings.csv; movies.csv
        Meta movie information fetched from TMDb API (only for 20M & 32M):
            - meta.csv
        ==========
        input_path & dataset_name = ml-10m ; ml-20m ; ml-32m
        output_path = output_data/ml-10m ; output_data/ml-20m ; output_data/ml-32m
        '''
        if input_path not in ('ml-10m', 'ml-20m', 'ml-32m'):
            raise ValueError(
                f"Invalid dataset: {input_path}. Allowed options are 'ml-10m', 'ml-20m' and  'ml-32m'."
            )
        if (input_path == 'ml-10m') and (meta_info):
            raise NotImplementedError("Additional meta information is not supported for the 'ml-10m' dataset.")

        self.meta_info = meta_info
        self.dataset_name = input_path
        
        inter_name = 'ratings.dat' if self.dataset_name == 'ml-10m' else 'ratings.csv'
        item_name = 'movies.dat' if self.dataset_name == 'ml-10m' else 'movies.csv'
        
        self.inter_file = os.path.join(self.input_path, inter_name)
        self.item_file = os.path.join(self.input_path, item_name)
        
        self.sep = '::' if self.dataset_name == 'ml-10m' else ','
        
        # output file
        self.output_inter_file, self.output_item_file, _ = self.get_output_files()
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             2: 'rating:float',
                             3: 'timestamp:float'}

        self.item_fields = {0: 'item_id:token',
                            1: 'movie_title:token_seq',
                            2: 'release_year:token',
                            3: 'class:token_seq'}
    
    def load_inter_data(self):
        return pd.read_csv(self.inter_file, delimiter=self.sep, header=None, engine='python')

    def load_item_data(self):
        origin_data = pd.read_csv(
            self.item_file, delimiter=self.sep, header=None, 
            engine='python', encoding='latin-1' if self.dataset_name == 'ml-10m' else None
        )
        processed_data = origin_data.iloc[:, 0:3]
        processed_data = self.GeneralProcess(processed_data)
        
        if self.meta_info:
            # i.e., in the case of 20M and 32M with meta_file
            self.item_fields[4] = 'tag:token_seq'
            self.item_fields[5] = 'date:float'
            self.item_fields[6] = 'runtime:token'
            self.item_fields[7] = 'description:token_seq'
            meta_file = os.path.join(self.input_path, 'meta.csv')
            
            # include additional information
            meta = pd.read_csv(meta_file)
            meta.iloc[:,0] = meta.iloc[:,0].apply(lambda row: str(row))
            
            processed_data = processed_data.merge(meta, left_on=0, right_on='movieId', how='left')
            processed_data.iloc[1:,5] = processed_data.iloc[1:,5].apply(
                lambda x: ', '.join(ast.literal_eval(x)) if isinstance(x, str) else 'None'
            )
            processed_data.drop(columns='movieId', inplace=True)
              
        return processed_data


class Amazon(BaseDataset):
    def __init__(self, input_path, output_path):
        super(Amazon, self).__init__(input_path, output_path)
        '''
        file name example: (for instance, the Video_Games subset)
        ==========
        Downloaded from Amazon: 
            meta_Video_Games.jsonl (for item); Video_Games.jsonl.gz (review)
        ==========
        input_path & dataset_name: Amazon_Video_Games
        output_path: output_data/Amazon_Video_Games
        ==========
        inter_file = Amazon_Video_Games/Video_Games.jsonl.gz
        item_file = Amazon_Video_Games/meta_Video_Games.jsonl
        output_inter_file = output_data/Amazon_Video_Games/Amazon_Video_Games.inter
        output_item_file = output_data/Amazon_Video_Games/Amazon_Video_Games.item
        '''
        if not input_path.startswith("Amazon_"):
            raise ValueError(
                f"Invalid input_path '{input_path}' or Mismatched dataset processing class. Amazon dataset must start with 'Amazon_'."
            )
        
        self.dataset_name = input_path

        # extract the subset name (e.g., Video_Games)
        sub = self.input_path.split("_", 1)[1]
        
        # input_file
        self.inter_file = os.path.join(self.input_path, f'{sub}.jsonl.gz')
        self.item_file = os.path.join(self.input_path, f'meta_{sub}.jsonl')

        self.sep = ','

        self.output_inter_file, self.output_item_file, _ = self.get_output_files()
        
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             2: 'rating:float',
                             3: 'timestamp:float'}

        self.item_fields = {0: 'item_id:token',
                            1: 'title:token',
                            2: 'description:token_seq',
                            3: 'categories:token_seq',
                            4: 'price:float',
                            5: 'date:float'}


    def load_inter_data(self):
        keys = ['user_id','parent_asin','rating','timestamp']
        records = []
        with gzip.open(self.inter_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print("Skip bad line.")
                    continue
                record = {key: obj.get(key) for key in keys}
                records.append(record)
        finished_data = pd.DataFrame(records)
        return finished_data
                    
                
    def load_item_data(self):
        records = []
        with open(self.item_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    record = {
                        'parent_asin': item.get('parent_asin'),
                        'title': item.get('title'),
                        'description': item.get('description')[0] if item.get('description') else None,
                        'categories': item.get('categories'),
                        'price': item.get('price') if item.get('price') is not None else 0,
                        'date': item.get('details', {}).get('Date First Available')  
                    }
                    records.append(record)
                except json.JSONDecodeError:
                    print("Skip this line")
        finished_data = pd.DataFrame(records)
        finished_data['categories'] = finished_data['categories'].apply(
            lambda x: ", ".join(x[1:]) if isinstance(x, list) and len(x) > 1 else ""
        )
        finished_data['date'] = finished_data["date"].apply(
            lambda x: int(
                time.mktime(parser.parse(x).timetuple())
            ) if isinstance(x, str) and x.strip() else None
        )
        
        return finished_data
        

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')
            
        







        
