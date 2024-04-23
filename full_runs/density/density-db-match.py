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
from recipe_dataset.utils.logger import *
from recipe_dataset.utils.full_run_utils import *

import pandas as pd
from pathlib import Path

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def literal_eval_cb(df, is_series):
    if is_series: df = df.iloc[:,0]
    logger.info("Applying literal eval")
    df = df.progress_apply(literal_eval)
    df = df.to_frame()
    return df

def match(expanded_ingredients_df, unit_types, food_df, save_path, **kargs):

    exploded_food_df = food_df.explode('description_list')['description_list'].to_frame('description').astype('string[pyarrow]')

    search_ids = load_or_create_dataframe(
        save_path=save_path.parent/(save_path.stem+'_search_list.feather'),
        create_function=search_df_pipe,
        is_series=True,
        series_name='search_ids',
        dtype_backend='pyarrow',
        postprocess_cbs=[literal_eval_cb],
        expanded_ingredients_df=expanded_ingredients_df.join(unit_types),
        exploded_food_df=exploded_food_df,
        **kargs
    )

    food_ids = load_or_create_dataframe(
        save_path=save_path.parent/(save_path.stem+'_select.feather'),
        create_function=select_searches_pipe,
        is_series=True,
        series_name='food_id',
        dtype_backend='pyarrow',
        expanded_ingredients_df=expanded_ingredients_df.join(unit_types).join(search_ids),
        food_df=food_df,
        **kargs
    )

    return food_ids

def food_portion_match(ingredients_df, food_ids, food_portion_df, chunk_save_dir):

    food_portions = parallel_apply_chunks(
        ingredients_df.join(food_ids),
        select_food_portion,
        args=(food_portion_df,),
        meta=pd.Series(dtype='int'),
        npartitions=500,
        chunksize=1e6,
        save_path=chunk_save_dir
    )

    return food_portions


def search_df_pipe(expanded_ingredients_df, exploded_food_df, chunksize, npartitions, func_save_path):

    searched_ids = parallel_apply_chunks(
        expanded_ingredients_df, 
        search_food_df, 
        meta=pd.Series(dtype='object'), 
        chunksize=chunksize,
        npartitions=npartitions,
        save_path=func_save_path,
        start_from=0,
        args=(exploded_food_df,)
    )

    return searched_ids

def select_searches_pipe(expanded_ingredients_df, food_df, chunksize, npartitions, func_save_path):

    selected_ids = parallel_apply_chunks(
        expanded_ingredients_df, 
        select_from_search_ids, 
        meta=pd.Series(dtype='Int64'), 
        chunksize=chunksize,
        npartitions=npartitions,
        save_path=func_save_path,
        start_from=0,
        args=(food_df,)
    )

    return selected_ids

def create_na_synonyms_df_stage(expanded_ingredients_df, food_ids):
    
    na_expanded_ingredients_df = expanded_ingredients_df.loc[food_ids.isna()]
    na_synonyms_df = create_na_synonyms_df(na_expanded_ingredients_df)
    return na_synonyms_df

def main():

    logger.info("Initialising files")
    save_dir = Path(f'{root}/../data/local/density/full/food_ids')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/../data/local/recipe/full/expanded_ingredients'), dtype_backend='pyarrow')
    ingredients_df = pd.read_feather(select_last_file(f'{root}/../data/local/recipe/full/ingredients'), columns=['recipe', 'ingredient', 'unit_type'], dtype_backend='pyarrow')
    food_df = pd.read_feather(select_last_file(save_dir/'../food'))

    # filtering ingredients with weights already measured
    expanded_ingredients_df = expanded_ingredients_df[~(ingredients_df['unit_type'] == 'weight')]

    food_df['description'] = food_df['description'].astype('string')

    logger.info("Commencing stages")

    logger.info(f"# COMMENCING STAGE {0}: (MATCH FOOD DF PRIMARY) #")
    food_ids = match(
        expanded_ingredients_df,
        ingredients_df['unit_type'],
        food_df, 
        save_path=save_dir/'0_food_ids_initial.feather',
        chunksize=5e5,
        npartitions=1000)

    logger.info(f"# COMMENCING STAGE {1}: (CREATE NA SYNONYMS DF) #")
    na_synonyms_df = load_or_create_dataframe(
        save_path=save_dir/'1_na_synonyms.feather',
        create_function=create_na_synonyms_df_stage, 
        is_series=False, 
        dtype_backend='pyarrow',
        expanded_ingredients_df=expanded_ingredients_df
    )

    logger.info(f"# COMMENCING STAGE {2}: (MATCH FOOD DF SYNONONYM DF) #")
    na_synonyms_food_ids = match(
        na_synonyms_df,
        ingredients_df['unit_type'],
        food_df, 
        save_path=save_dir/'2_food_ids_synonyms.feather',
        chunksize=1e5,
        npartitions=500)

    food_ids = food_ids.fillna(na_synonyms_food_ids)
    food_ids.to_frame().to_feather(save_dir/'3_food_ids_filled.feather')

    logger.info(f"# COMMENCING STAGE {4}: (MATCH PORTION DF) #")
    food_portion_df = pd.read_feather(select_last_file(f'{root}/../data/local/density/full/food_portion'), columns=['fdc_id', 'id', 'gram_weight', 'amount', 'unit_tags'], dtype_backend='pyarrow')
    food_portion_ids = load_or_create_dataframe(
        save_path=save_dir/'..'/'food_portion_ids'/'0_portion_ids.feather',
        create_function=food_portion_match,
        is_series=True,
        series_name='food_portion_id',
        dtype_backend='pyarrow',
        ingredients_df=ingredients_df, 
        food_ids=food_ids, 
        food_portion_df=food_portion_df,
        chunk_save_dir=save_dir/'..'/'food_portion_ids'/'0_portion_ids.feather'
    )
    
if __name__ == '__main__':
    main()