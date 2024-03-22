from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from food_database.utils.parallel import *
from food_database.recipes.create import *
from food_database.utils.utils import *
from food_database.utils.join_utils import *
from food_database.density.food_match import *
from food_database.utils.logger import *
from food_database.utils.full_run_utils import *

import pandas as pd
from pathlib import Path

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def join(expanded_ingredients_df, food_df, save_path):

    exploded_food_df = food_df.explode('description_list')['description_list'].to_frame('description')

    matched_food_ids = parallel_apply_chunks(
        expanded_ingredients_df, 
        match_ingredient, 
        meta=pd.Series(dtype='int64'), 
        chunksize=5e5,
        npartitions=5000,
        save_path=save_path,
        start_from=0,
        args=(food_df, exploded_food_df)
    )

    return matched_food_ids

def create_na_synonyms_df_stage(expanded_ingredients_df, food_ids):
    
    na_expanded_ingredients_df = expanded_ingredients_df.loc[food_ids.isna()]
    na_synonyms_df = create_na_synonyms_df(na_expanded_ingredients_df)
    return na_synonyms_df

def na_synonym_join(na_synonyms_df, food_df):

    exploded_food_df = food_df.explode('description_list')['description_list'].to_frame('description')

    na_synonym_food_ids = parallel_apply(na_synonyms_df, match_ingredient, args=(food_df, exploded_food_df), meta=pd.Series(dtype='int64'))

    return na_synonym_food_ids

def main():

    logger.info("Initialising files")
    save_dir = Path(f'{root}/data/local/density/full/food_ids')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/expanded_ingredients'), dtype_backend='pyarrow')
    ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/ingredients'), columns=['recipe', 'ingredient', 'unit_type'], dtype_backend='pyarrow')
    food_df = pd.read_feather(select_last_file(save_dir/'../food'))

    expanded_ingredients_df = expanded_ingredients_df.join(ingredients_df)
    logger.info(expanded_ingredients_df.shape)

    logger.info("Commencing stages")

    logger.info(f"# COMMENCING STAGE {0}: (JOIN PRIMARY) #")
    food_ids = load_or_create_dataframe(
        save_dir/'0_primary_join.feather',
        join, 
        expanded_ingredients_df, food_df, save_dir/'0_primary_join.feather'
    )

    logger.info(f"# COMMENCING STAGE {1}: (CREATE NA SYNONYMS DF) #")
    na_synonyms_df = load_or_create_dataframe(
        save_dir/'1_na_synonyms.feather',
        create_na_synonyms_df_stage, 
        expanded_ingredients_df, food_ids
    )

    logger.info(f"# COMMENCING STAGE {2}: (JOIN SYNONONYM DF) #")
    na_synonym_food_ids = load_or_create_dataframe(
        save_dir/'2_na_synonym_join.feather',
        na_synonym_join, 
        na_synonyms_df, food_df
    )

    food_ids = food_ids.fillna(na_synonym_food_ids)
    food_ids.to_feather(save_dir/'2_finalised.feather')
    
if __name__ == '__main__':
    main()