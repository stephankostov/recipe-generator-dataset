from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

import logging
from pathlib import Path
import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()

from recipe_dataset.final.process import *
from recipe_dataset.utils.full_run_utils import *
from recipe_dataset.utils.logger import *
from recipe_dataset.utils.parallel import *

logger = configure_logger(logging.getLogger(__name__))

# def pipe_merge_duplicate_foods(foods):

#     meta = pd.Series(name='weight_ratio', dtype='double[pyarrow]', index=['food_id'])
    
#     def f(foods):
#         return foods.reset_index().drop('ingredient', axis=1).set_index('recipe').drop('index', axis=1) \
#             .groupby('recipe').apply(merge_duplicates, include_groups=False).compute()
    
#     foods = parallel_df_function(
#         foods, 
#         f, 
#         npartitions=500,
#         keep_index=True
#     )

#     foods = foods.to_frame().reset_index().set_index('recipe')

#     return foods

def pipe_merge_duplicate_foods(foods):
    merged_duplicates = foods.groupby('recipe').progress_apply(merge_duplicates)
    merged_duplicates = merged_duplicates.to_frame().reset_index().set_index('recipe')
    return merged_duplicates
    

def pipe_process_food_ids(foods, food_names, special_tokens):

    return process_foods_df(foods, food_names, special_tokens)

def main():

    save_dir = Path(f'{root}/../data/local/final/full/recipes')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    food_ids = pd.read_feather(select_last_file(f'{root}/../data/local/molecule/full/food_ids'), dtype_backend='pyarrow')
    food_weights = pd.read_feather(select_last_file(f'{root}/../data/local/density/full/weights'), dtype_backend='pyarrow')
    food_names = np.load(f'{root}/../data/local/final/full/food_names/0.npy')

    foods = food_ids.join(food_weights, how='inner')

    print(foods.shape)
    
    logger.info("Processing foods_df")
    foods = load_or_create_dataframe(
        save_dir/'0_food_ids_processed.feather',
        pipe_process_food_ids,
        is_series=False,
        dtype_backend='pyarrow',
        foods=foods, food_names=food_names, special_tokens=special_tokens
    )

    assert foods[foods['food_id']==np.where(food_names=='salt')[0][0]].empty

    print(foods.shape)

    logger.info("Merging food_ids")
    foods = load_or_create_dataframe(
        save_dir/'1_merged.feather',
        pipe_merge_duplicate_foods,
        is_series=False,
        dtype_backend='pyarrow',
        foods=foods
    )

    foods = foods.astype({
        'food_id': 'int64',
        'weight_ratio': 'float64'
    })

    logger.info("Creating food_ids numpy array")
    food_ids = pd.DataFrame(foods.groupby('recipe')['food_id'].aggregate(lambda x: list(x)[:14]).tolist())
    food_ids = food_ids \
        .fillna(np.where(food_names=='<pad>')[0][0]) \
        .astype('int') \
        .to_numpy()
    np.save(save_dir/'recipe_food_ids.npy', food_ids)

    print(food_ids)

    logger.info("Creating food_weights numpy array")
    food_weights = pd.DataFrame(foods.groupby('recipe')['weight_ratio'].aggregate(lambda x: list(x)[:14]).tolist())
    food_weights = food_weights \
        .fillna(0.) \
        .astype('float64') \
        .to_numpy()
    np.save(save_dir/'recipe_food_weights.npy', food_weights)

    print(food_weights)

if __name__ == '__main__':
    main()

