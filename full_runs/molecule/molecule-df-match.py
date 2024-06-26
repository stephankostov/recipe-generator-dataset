from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from recipe_dataset.utils.parallel import *
from recipe_dataset.utils.utils import *
from recipe_dataset.utils.join_utils import *
from recipe_dataset.molecule.match import *
from recipe_dataset.utils.logger import *
from recipe_dataset.utils.full_run_utils import *

import pandas as pd
from pathlib import Path

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def match(expanded_ingredients_df, foods, save_path):
    
    food_ids = parallel_apply_chunks(
        expanded_ingredients_df,
        match_ingredient,
        meta=pd.Series(dtype='int64'),
        chunksize=5e5,
        npartitions=5000,
        save_path=save_path,
        args=(foods,)
    )

    return food_ids

def create_na_synonyms_df_stage(expanded_ingredients_df, food_ids):

    na_expanded_ingredients_df = expanded_ingredients_df[food_ids.isna()]
    na_synonyms_df = create_na_synonyms_df(na_expanded_ingredients_df)
    return na_synonyms_df

def main():

    save_dir = Path(f'{root}/../data/local/molecule/full/food_ids')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/../data/local/recipe/full/expanded_ingredients'), dtype_backend='pyarrow')
    food_names = pd.read_feather(f'{root}/../data/local/molecule/full/food/2_feature_engineered.feather', columns=['name'])['name']
    food_names.index.rename('id',inplace=True)
    
    logger.info(f"Loaded dataframes with shapes: {[expanded_ingredients_df.shape, food_names.shape]}")

    logger.info(f"# COMMENCING STAGE {0}: (MATCH PRIMARY) #")
    food_ids = load_or_create_dataframe(
        save_dir/'0_primary_match.feather',
        match, 
        expanded_ingredients_df, food_names, save_dir/'0_primary_match.feather'
    )

    # #TODO: load_or_create_dataframe not able to handle series output from compile chunks
    food_ids = food_ids.rename({0: 'food_id'}, axis=1)
    food_ids = food_ids['food_id']

    logger.info(f"# COMMENCING STAGE {1}: (CREATE NA SYNONYMS DF) #")
    na_synonyms_df = load_or_create_dataframe(
        save_dir/'1_na_synonyms.feather',
        create_na_synonyms_df_stage, 
        expanded_ingredients_df, food_ids
    )

    logger.info(f"# COMMENCING STAGE {2}: (MATCH SYNONONYM DF) #")
    na_synonym_food_ids = load_or_create_dataframe(
        save_dir/'2_na_synonym_match.feather',
        match, 
        na_synonyms_df, food_names, save_dir/'2_na_synonym_match.feather'
    )

    #TODO: load_or_create_dataframe not able to handle series output from compile chunks
    na_synonym_food_ids = na_synonym_food_ids.rename({0: 'food_id'}, axis=1)
    na_synonym_food_ids = na_synonym_food_ids['food_id']

    logger.info(f"# COMMENCING STAGE {3}: (FILLING NA RESULTS) #")
    food_ids = food_ids.fillna(na_synonym_food_ids)
    food_ids.to_frame().to_feather(save_dir/'3_na_filled.feather')

    # logger.info("Joining dataframes")
    # join_results = match(expanded_ingredients_df, 
    #                     food_names, 
    #                     Path(f'{root}/../data/local/molecule/full/food_id'))

    # logger.info("Filling NA ingredients")
    # join_results = na_synonym_join(expanded_ingredients_df, 
    #                 join_results, 
    #                 food_names)
    # join_results.to_frame('food_id').to_feather(f'{root}/../data/local/molecule/full/food_id/3_na_filled.feather')

if __name__ == '__main__':
    main()