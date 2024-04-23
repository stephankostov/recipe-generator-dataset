from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from recipe_dataset.utils.parallel import *
from recipe_dataset.recipes.create import *
from recipe_dataset.utils.utils import *
from recipe_dataset.utils.join_utils import *
from recipe_dataset.density.food_match import *
from recipe_dataset.density.portion_match import *
from recipe_dataset.density.finalise import unit_list, get_gram_weight
from recipe_dataset.utils.logger import *
from recipe_dataset.utils.full_run_utils import *

from dask.distributed import Client, LocalCluster
import multiprocessing as mp

import pandas as pd
import dask.dataframe as dd

from pathlib import Path
import json

import logging
logger = configure_logger(logging.getLogger(__name__))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def pipe_gram_weight(ingredients_df, food_portion_df, unit_list):

    gram_weights = parallel_apply(
        ingredients_df[['quantity', 'unit_tags', 'unit_type', 'food_id', 'food_portion_id']],
        get_gram_weight,
        meta=pd.Series(name='gram_weight', dtype='double[pyarrow]'),
        npartitions=500,
        args=(food_portion_df, unit_list)
    ).astype('double[pyarrow]')

    return gram_weights

def pipe_gram_weight_postprocess(gram_weights):

    def fill_na_weights(gram_weights):
        return gram_weights.groupby('recipe')['gram_weight'] \
            .transform(lambda x: x.astype('double[pyarrow]').fillna(x.mean())).compute()

    gram_weights = parallel_df_function(
        gram_weights.to_frame('gram_weight'),
        fill_na_weights,
        keep_index=True,
        npartitions=500
    ).astype('double[pyarrow]')

    return gram_weights

def pipe_pct_weight(gram_weights):

    def ddf_func(ddf):
        recipe_weights = ddf.groupby('recipe')['gram_weight'].transform(sum, meta=pd.Series(name='recipe_weight', dtype='double[pyarrow]')).rename('recipe_weight').compute()
        ddf = ddf.join(recipe_weights, on='recipe').compute()
        results = ddf['gram_weight'] / ddf['recipe_weight']
        return results
    
    return parallel_df_function(
        gram_weights,
        ddf_func,
        npartitions=500,
        keep_index=True
    )

def main():

    logger.info("Initialising files")
    save_dir = Path(f'{root}/../data/local/density/full/weights')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    ingredients_df = pd.read_feather(select_last_file(f'{root}/../data/local/recipe/full/ingredients'), columns=['recipe', 'ingredient', 'quantity', 'unit_tags', 'unit_type'], dtype_backend='pyarrow')
    food_portion_df = pd.read_feather(select_last_file(f'{root}/../data/local/density/full/food_portion'), columns=['fdc_id', 'id', 'gram_weight', 'amount', 'unit_tags', 'unit_type', 'portion_amount'], dtype_backend='pyarrow')
    food_ids = pd.read_feather(select_last_file(f'{root}/../data/local/density/full/food_ids'), dtype_backend='pyarrow').iloc[:,0].rename('food_id')
    food_portion_ids = pd.read_feather(select_last_file(f'{root}/../data/local/density/full/food_portion_ids'), dtype_backend='pyarrow').iloc[:,0].rename('food_portion_id')

    ingredients_df['quantity'] = ingredients_df['quantity'].astype('double[pyarrow]')

    logger.info("Commencing stages")

    logger.info(f"# COMMENCING STAGE {0}: (CONVERT INGREDIENT WEIGHTS) #")
    gram_weights = load_or_create_dataframe(
        save_path=save_dir/'0_gram_weights.feather',
        create_function=pipe_gram_weight,
        is_series=True, 
        series_name='gram_weight',
        dtype_backend='pyarrow',
        ingredients_df=ingredients_df.join(food_ids).join(food_portion_ids),
        food_portion_df=food_portion_df, 
        unit_list=unit_list
    )

    logger.info(f"# COMMENCING STAGE {1}: (POSTPROCESS INGREDIENT WEIGHTS) #")
    gram_weights = load_or_create_dataframe(
        save_path=save_dir/'1_gram_weights_post.feather',
        create_function=pipe_gram_weight_postprocess,
        is_series=True, 
        series_name='gram_weight',
        dtype_backend='pyarrow',
        gram_weights=gram_weights
    )

    logger.info(f"# COMMENCING STAGE {2}: (GET WEIGHT RATIOS) #")
    weight_ratios = load_or_create_dataframe(
        save_path=save_dir/'2_weight_ratios.feather',
        create_function=pipe_pct_weight,
        is_series=True,
        series_name='weight_ratios',
        dtype_backend='pyarrow',
        gram_weights=gram_weights,
    )

if __name__ == '__main__':
    main()