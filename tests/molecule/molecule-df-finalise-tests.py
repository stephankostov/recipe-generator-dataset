from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

from food_database.utils.parallel import *
from food_database.recipes.create import *
from food_database.utils.utils import *
from food_database.utils.join_utils import *
from food_database.molecule.finalise import *
from food_database.utils.logger import *
from food_database.utils.full_run_utils import *

import pandas as pd
from pathlib import Path
import json

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def test_instance(test_info, expanded_ingredients_df, food_ids, content_df, unique_orig_foods):

    ingredient = expanded_ingredients_df.loc[tuple(test_info['ingredient_id'])]
    ingredient['food_id'] = food_ids['food_id'].loc[ingredient.name]

    matched_unique_orig_foods = unique_orig_foods.loc[ingredient['food_id']]

    print(test_info)

    print("INGREDIENT:", ingredient[ingredient.notnull()], '', sep='\n')
    print("EXPECTING:", matched_unique_orig_foods[(matched_unique_orig_foods['citation'] == test_info['citation']) & (matched_unique_orig_foods['orig_food_id'].isin(test_info['orig_food_id']))].to_markdown(), '', sep='\n')

    selection = full_select_duplicate_foods(ingredient, unique_orig_foods, content_df)[test_info['citation']]
    selection = selection[1:]

    if not selection:
        print("OUTCOME:", 'Empty DF', sep='\n')
        print("==========================", "RESULT: FAILED", "==========================", sep='\n')
        return 0

    if matched_unique_orig_foods.loc[selection]['orig_food_id'] in test_info['orig_food_id']:
        print('\n')
        print("==========================", "RESULT: PASSED", "==========================", sep='\n')
        print('\n\n')
        return 1
    else:
        print("OPTIONS:", matched_unique_orig_foods[(matched_unique_orig_foods['citation'] == test_info['citation'])].to_markdown(), sep='\n')
        print("OUTCOME:", matched_unique_orig_foods.loc[selection], sep='\n')
        print('\n')
        print("xxxxxxxxxxxxxxxxxxxxxxxxx", "RESULT: FAILED", "xxxxxxxxxxxxxxxxxxxxxxxxx", sep='\n')
        print('\n\n')
        return 0

def run_tests(tests_info, *args):

    test_score = 0
    for i, test_info in enumerate(tests_info):
        print(f"TEST #{i}", sep='\n')
        test_result = test_instance(test_info, *args)
        test_score += test_result

    print("TESTS COMPLETE")
    print("PASSED TESTS:", test_score)
    print("TOTAL TESTS:", len(tests_info))


def main():

    logger.info("Loading data")

    with open(f'{root}/tests/molecule/molecule-db-test-config.json') as f:
        tests_info = json.load(f)['duplicate_selection']

    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/expanded_ingredients/'), dtype_backend='pyarrow')
    food_ids = pd.read_feather(select_last_file(f'{root}/data/local/density/full/food_ids/'), columns=['recipe', 'ingredient', 'unit_type'])
    content_df = pd.read_feather(select_last_file(f'{root}/data/local/density/full/content/'))

    content_df = content_df.set_index(content_df.reset_index().index, append=True)
    content_df.index.rename('content_id', inplace=True, level=2)

    content_df['food_id'] = content_df.index.get_level_values(0)
    unique_orig_foods = content_df[['food_id', 'citation', 'orig_food_id', 'orig_food_common_name']].drop_duplicates()[['citation', 'orig_food_common_name', 'orig_food_id']]
    unique_orig_foods['citation'].fillna('', inplace=True)
    content_df.drop('food_id', axis=1, inplace=True)

    logger.info("Running tests")

    run_tests(tests_info, expanded_ingredients_df, food_ids, content_df, unique_orig_foods)


if __name__ == '__main__':
    main()