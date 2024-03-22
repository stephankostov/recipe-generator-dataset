from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from food_database.utils.parallel import *
from food_database.recipes.create import *
from food_database.utils.utils import *
from food_database.utils.join_utils import *
from food_database.density.portion_match import *
from food_database.utils.logger import *
from food_database.utils.full_run_utils import *

import pandas as pd
from pathlib import Path
import json
import shutil

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def test_instance(test_info, ingredients_df, food_df, food_portion_df):

    ingredient = ingredients_df.loc[tuple(test_info['ingredient_id'])]

    print("INGREDIENT:", ingredient[ingredient.notnull()], '', sep='\n')
    print("FOOD:", food_df.loc[ingredient['food_id']], '', sep='\n')
    print("PORTION OPTIONS:", food_portion_df.loc[ingredient['food_id'],:].to_markdown(), '', sep='\n')
    if test_info['portion_id'][0] not in food_portion_df.loc[ingredient['food_id'],:].index:
        print('\n')
        print("Given portion index does not exist for selected food")
        print('\n')
        print("xxxxxxxxxxxxxxxxxxxxxxxxx", "RESULT: FAILED", "xxxxxxxxxxxxxxxxxxxxxxxxx", sep='\n')
        print('\n\n')
        return 0, pd.DataFrame()

    print("EXPECTING", food_portion_df.loc[ingredient['food_id'],test_info['portion_id'][0]], '', sep='\n')

    selected_portion = select_food_portion(ingredient, food_portion_df)

    if selected_portion in test_info['portion_id']:
        print('\n')
        print("==========================", "RESULT: PASSED", "==========================", sep='\n')
        print('\n\n')
        return 1, selected_portion
    else:
        print("OUTCOME:", food_portion_df.loc[ingredient['food_id'], selected_portion], sep='\n')
        print('\n')
        print("xxxxxxxxxxxxxxxxxxxxxxxxx", "RESULT: FAILED", "xxxxxxxxxxxxxxxxxxxxxxxxx", sep='\n')
        print('\n\n')
        return 0, selected_portion
    
def write_failure_info(df, save_path):
    df.to_html(save_path)

def initialise_output_files(save_dir):
    if save_dir.exists(): shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

def run_tests(tests_info, save_dir, *args):

    test_score = 0
    for i, test_info in enumerate(tests_info):
        print(f"TEST #{i}", sep='\n')
        test_result, output_df = test_instance(test_info, *args)
        if test_result:
            test_score += 1
        else:
            write_failure_info(output_df, save_dir/f"{i}_{test_info['name']}.html")

    print("TESTS COMPLETE")
    print("PASSED TESTS:", test_score)
    print("TOTAL TESTS:", len(tests_info))

def main():

    logger.info("Initialising files")
    save_dir = Path(f'{root}/data/tests/density/portion-df-join/')
    initialise_output_files(save_dir)

    logger.info("Loading data")

    with open(f'{root}/tests/density/density-db-test-config.json') as f:
        tests_info = json.load(f)['portion_df']

    ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/partial/ingredients/'))
    food_ids = pd.read_feather(select_last_file(f'{root}/data/local/density/partial/food_ids/'))
    food_df = pd.read_feather(select_last_file(f'{root}/data/local/density/full/food/'))
    food_portion_df = pd.read_feather(select_last_file(f'{root}/data/local/density/full/food_portion/'))

    logger.info("Running tests")

    run_tests(tests_info, save_dir, ingredients_df.join(food_ids), food_df, food_portion_df)


if __name__ == '__main__':
    main()