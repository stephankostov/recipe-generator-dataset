from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

import logging
from pathlib import Path
import pandas as pd
import numpy as np

from recipe_dataset.final.process import *
from recipe_dataset.utils.full_run_utils import *
from recipe_dataset.utils.logger import *

logger = configure_logger(logging.getLogger(__name__))

def main():

    save_dir = Path(f'{root}/../data/local/molecule/full/food_ids')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframes")
    food_ids = pd.read_feather(select_last_file(f'{root}/../data/local/molecule/full/food_ids'), dtype_backend='pyarrow')
    
    logger.info("Processing food_ids")
    food_ids = process_food_ids(food_ids, special_tokens)

    logger.info("Compiling recipe food_ids")
    recipe_food_ids = compile_recipe_food_ids(food_ids, special_tokens)
    
    logger.info("Saving")
    np.save(f'{root}/../data/local/final/full/recipe_food_ids/0.npy', recipe_food_ids)

if __name__ == '__main__':
    main()

