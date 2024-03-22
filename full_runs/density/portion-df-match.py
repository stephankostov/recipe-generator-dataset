from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from food_database.utils.parallel import *
from food_database.recipes.create import *
from food_database.utils.utils import *
from food_database.utils.join_utils import *
from food_database.density.food_match import *
from food_database.density.portion_match import *
from food_database.utils.logger import *
from food_database.utils.full_run_utils import *


import pandas as pd
from pathlib import Path

import logging

logger = configure_logger(logging.getLogger(__name__))

def food_portion_select(ingredients_df, food_ids, portion_df, save_path):

    food_portions = parallel_apply_chunks(
        ingredients_df.join(food_ids),
        select_food_portion,
        args=(portion_df,),
        meta=pd.Series(dtype='int'),
        npartitions=2000,
        chunksize=1e6,
        save_path=save_path
    )

    return food_portions


def main():

    logger.info("Initialising files")
    save_dir = Path(f'{root}/data/local/density/full/food_portion_ids')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/ingredients'))
    portion_df = pd.read_feather(select_last_file(f'{root}/data/local/density/full/food_portion'))
    food_ids = pd.read_feather(select_last_file(f'{root}/data/local/density/full/food_ids'))

    logger.info("Commencing stages")

    logger.info(f"# COMMENCING STAGE {0}: (FOOD PORTION SELECT) #")
    food_ids = load_or_create_dataframe(
        save_dir/'0_portion_select.feather',
        food_portion_select, 
        ingredients_df, food_ids, portion_df, save_dir/'0_primary_join.feather'
    )

if __name__ == '__main__':
    main()