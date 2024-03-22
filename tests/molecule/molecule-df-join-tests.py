from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from food_database.utils.parallel import *
from food_database.recipes.create import *
from food_database.utils.utils import *
from food_database.utils.join_utils import *
from food_database.molecule.match import *
from food_database.utils.logger import *
from food_database.utils.full_run_utils import *

import pandas as pd
from pathlib import Path
import json
import shutil

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def test_instance(test_info, expanded_ingredients_df, food_df):

    ingredient = expanded_ingredients_df.loc[tuple(test_info['ingredient_id'])]

    print("INGREDIENT:", ingredient[ingredient.notnull()], '', sep='\n')
    print("EXPECTING:", food_df.loc[test_info['food_id']]['name'].to_markdown(), '', sep='\n')
    print("COMMENT:", test_info['comment'] if 'comment' in test_info else None )

    matched_df = food_df.loc[find_ingredient_food_df_matches(ingredient, food_df['name'])]

    if matched_df.empty:
        print("OUTCOME:", 'Empty DF', sep='\n')
        print("==========================", "RESULT: FAILED", "==========================", sep='\n')
        return 0, pd.DataFrame()

    selected_df = select_from_matches(ingredient, matched_df['name'], True)

    selected_df = selected_df.fillna('')

    if selected_df.iloc[0].name in test_info['food_id']:
        print('\n')
        print("==========================", "RESULT: PASSED", "==========================", sep='\n')
        print('\n\n')
        return 1, selected_df
    else:
        print("OUTCOME:", selected_df['name'].head().to_markdown(), sep='\n')
        print('\n')
        print("xxxxxxxxxxxxxxxxxxxxxxxxx", "RESULT: FAILED", "xxxxxxxxxxxxxxxxxxxxxxxxx", sep='\n')
        print('\n\n')
        return 0, selected_df
    
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
    save_dir = Path(f'{root}/data/tests/food-df-join/')
    initialise_output_files(save_dir)

    logger.info("Loading data")

    with open(f'{root}/tests/molecule/molecule-db-test-config.json') as f:
        tests_info = json.load(f)['join']

    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/expanded_ingredients/'), dtype_backend='pyarrow')
    food_df = pd.read_feather(select_last_file(f'{root}/data/local/molecule/full/food/'))

    logger.info("Running tests")

    run_tests(tests_info, save_dir, expanded_ingredients_df, food_df)


if __name__ == '__main__':
    main()