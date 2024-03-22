# to be applied after both ingredients and expanded ingredients df's have been created
# separated out into its own script in order to keep both create- scripts decoupled

from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

import argparse
import re
from pathlib import Path

import pandas as pd
from ast import literal_eval

from food_database.utils.utils import *
from food_database.recipes.create import *
from food_database.utils.parallel import *
from food_database.utils.logger import *

import logging
logger = configure_logger(logging.getLogger(__name__))


def filter_words(ingredients_df, expanded_ingredients_df, save_path):

    ingredients_df[['unit', 'quantity']].replace('', pd.NA, inplace=True)
    expanded_ingredients_df.replace('', pd.NA, inplace=True)
 
    quantity_filter = (ingredients_df['unit'].isnull()) & (ingredients_df['quantity'].isnull())
    
    word_filter = parallel_apply_chunks(
        expanded_ingredients_df, 
        filter_patterns,
        meta=pd.Series(dtype='bool'), 
        args=(filters,),
        npartitions=200,
        chunksize=5e6, 
        save_path=save_path
    )

    expanded_ingredients_df = expanded_ingredients_df.loc[~(quantity_filter & word_filter)]
    ingredients_df = ingredients_df.loc[expanded_ingredients_df.index.to_numpy()]

    return ingredients_df, expanded_ingredients_df

def select_last_file(file_path, glob_pattern=r'*[0-9]*.feather'):
    files = file_path.glob(glob_pattern)
    files = [file for file in files if file.is_file() and re.match(r'^[0-9]+', file.name)]
    files = sorted(list(files))
    files = [file for file in files if not re.match(r'[0-9]+_filtered.feather', str(file.name))]
    last_file = files[-1]
    return last_file

def iterate_file(file_path):
    file_number = int(file_path.name.split('_')[0])
    new_file = Path(file_path.parent)/f"{file_number+1}_filtered.feather"
    return new_file

def main():

    save_dir = Path(f'{root}/data/local/recipe/full/')

    ingredients_df_path = select_last_file(save_dir/'ingredients')
    expanded_ingredients_df_path = select_last_file(save_dir/'expanded_ingredients')

    print(ingredients_df_path, expanded_ingredients_df_path)
    
    logger.info("Loading dataframes")
    ingredients_df = pd.read_feather(ingredients_df_path)
    expanded_ingredients_df = pd.read_feather(expanded_ingredients_df_path, dtype_backend='pyarrow')

    logger.info("Filtering dataframes")
    ingredients_df, expanded_ingredients_df = filter_words(ingredients_df, expanded_ingredients_df, save_path=save_dir/'ingredients'/'filter_function.feather')

    logger.info("Saving")
    ingredients_df.to_feather(iterate_file(ingredients_df_path))
    expanded_ingredients_df.to_feather(iterate_file(expanded_ingredients_df_path))

if __name__ == '__main__':
    main()