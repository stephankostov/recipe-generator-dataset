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
import shutil
import json

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def test_instance(test_info, expanded_ingredients_df, ingredients_df, food_df, exploded_food_df):

    ingredient = expanded_ingredients_df.loc[tuple(test_info['ingredient_id'])]
    print("INGREDIENT:", ingredient[ingredient.notnull()], sep='\n')
    ingredient = transform_ingredient(ingredient)
    print("TRANSFORMED INGREDIENT:", ingredient[ingredient.notnull()], sep='\n')
    print("EXPECTING:", food_df.loc[test_info['food_id']].to_markdown(), sep='\n')

    matched_df = food_df.loc[match_food_df_on_ingredient(ingredient, exploded_food_df)]
    if matched_df.empty:
        print("OUTCOME:", 'No matches found', sep='\n')
        print("==========================", "RESULT: FAILED", "==========================", sep='\n')
        return 0, pd.DataFrame()
    
    unit_type = ingredients_df['unit_type'].loc[ingredient.name]
    if unit_type == 'weight': return 1, pd.DataFrame()

    selected_df = select_from_matches(matched_df, ingredient, {f'{unit_type}_exists':False, **sort_order}, True)

    if selected_df.iloc[0].name in test_info['food_id']:
        print('\n')
        print("==========================", "RESULT: PASSED", "==========================", sep='\n')
        print('\n\n')
        return 1, selected_df
    else:
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

    with open(f'{root}/tests/density/density-db-test-config.json') as f:
        tests_info = json.load(f)['food_df']

    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/expanded_ingredients/'), dtype_backend='pyarrow')
    ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/ingredients/'), columns=['recipe', 'ingredient', 'unit_type'])
    food_df = pd.read_feather(select_last_file(f'{root}/data/local/density/full/food/'))

    exploded_food_df = food_df.explode('description_list')['description_list'].to_frame('description')

    logger.info("Running tests")

    run_tests(tests_info, save_dir, expanded_ingredients_df, ingredients_df, food_df, exploded_food_df)


if __name__ == '__main__':
    main()