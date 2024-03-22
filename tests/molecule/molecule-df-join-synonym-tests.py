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

from ast import literal_eval

import logging

logger = configure_logger(logging.getLogger(__name__))

def create_ingredient_synonym_df(test_info, expanded_ingredients_df):

    ingredients = [tuple(test['ingredient_id']) for test in test_info]
    synonym_df = create_na_synonyms_df(expanded_ingredients_df.loc[ingredients])
    
    return synonym_df

def test_instance(test, expanded_ingredients_df, synonyms_df, food_df):

    ingredient = expanded_ingredients_df.loc[tuple(test['ingredient_id'])]
    synonym_ingredient = synonyms_df.loc[tuple(test['ingredient_id'])]

    print("INGREDIENT:", ingredient[ingredient.notnull()], '', sep='\n')
    print("INGREDIENT SYNONYMS:", synonym_ingredient[synonym_ingredient.notnull()], '', sep='\n')
    print("EXPECTING:", food_df.loc[test['food_id']]['name'].to_markdown(), '', sep='\n')
    print("COMMENT:", test['comment'] if 'comment' in test else None )

    match_df = food_df.loc[find_ingredient_food_df_matches(synonym_ingredient, food_df['name'])]

    if match_df.empty:
        print("OUTCOME:", 'Empty DF', sep='\n')
        print("==========================", "RESULT: FAILED", "==========================", sep='\n')
        return 0

    ordered_df = select_from_matches(synonym_ingredient, match_df['name'], True)

    ordered_df = ordered_df.fillna('')

    if ordered_df.iloc[0].name in test['food_id']:
        print('\n')
        print("==========================", "RESULT: PASSED", "==========================", sep='\n')
        print('\n\n')
        return 1
    else:
        print("OUTCOME:", ordered_df['name'].head().to_markdown(), sep='\n')
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
        tests_info = json.load(f)['synonym_join']

    expanded_ingredients_df = pd.read_feather(select_last_file(f'{root}/data/local/recipe/full/expanded_ingredients/'), dtype_backend='pyarrow')
    food_df = pd.read_feather(select_last_file(f'{root}/data/local/molecule/full/food/'))

    ingredient_synonym_df = create_ingredient_synonym_df(tests_info, expanded_ingredients_df)

    logger.info("Running tests")

    run_tests(tests_info, expanded_ingredients_df, ingredient_synonym_df, food_df)


if __name__ == '__main__':
    main()