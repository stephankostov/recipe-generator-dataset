from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

import argparse
import re
from pathlib import Path

import pandas as pd
from ast import literal_eval

from food_database.utils.utils import *
from food_database.recipes.create import *
from food_database.utils.parallel import *
from food_database.utils.logger import *

import logging
logger = configure_logger(logging.getLogger(__name__))

def create_from_recipes_df(recipes_df):

    recipes_df['ingredients'] = recipes_df['ingredients'].apply(literal_eval)
    ingredients_df = recipes_df.explode('ingredients')

    # setting indices for the ingredients in each recipe (multiindex)
    ingredients_df['ingredient'] = ingredients_df.groupby(ingredients_df.index).cumcount()
    ingredients_df = ingredients_df.set_index([ingredients_df.index, 'ingredient'])
    ingredients_df.index = ingredients_df.index.set_names(['recipe', 'ingredient'])

    # selecting columns
    ingredients_df = ingredients_df[['ingredients', 'NER']]
    ingredients_df.rename(columns={'ingredients':'ingredient_string'}, inplace=True)

    return ingredients_df

def parse_ingredient_string_preprocess(ingredients_df, **kargs):

    ingredients_df['ingredient_string'] = parallel_apply(
        ingredients_df['ingredient_string'], 
        ner_preprocess_ingredient_string, 
        meta=pd.Series(dtype='string'), 
        npartitions=500
    )
    
    return ingredients_df

def parse_ingredient_string_parsing(ingredients_df, **kargs):

    ingredients_df['parsed'] = parallel_apply(
        ingredients_df['ingredient_string'], 
        parse_ingredient_string, 
        meta=pd.Series(dtype='object'), 
        npartitions=500
    )
    
    ingredients_df['parsed'] = ingredients_df['parsed'].apply(literal_eval)

    expanded = pd.json_normalize(ingredients_df['parsed'])
    expanded = expanded.astype('string')
    expanded.set_index(ingredients_df.index, inplace=True)
    ingredients_df = pd.concat([ingredients_df, expanded], axis=1)

    ingredients_df.drop(['parsed'], axis=1, inplace=True, errors='ignore')

    return ingredients_df

def parse_ingredient_string_postprocess(ingredients_df, **kargs):

    ingredients_df = ingredients_df[~ingredients_df['name'].isna()]

    ingredients_df['name.ner'] = parallel_apply(
        ingredients_df[['NER', 'name']],
        find_ner_match,
        meta=pd.Series(dtype="string"),
        npartitions=500,
        keep_index=True
    )
    ingredients_df = ingredients_df.rename({'name.ner': 'name.name', 'name': 'name.description', 'ingredient_string': 'ingredient_string'}, axis=1)
    ingredients_df = ingredients_df[['name.name', 'name.description', 'quantity', 'unit', 'comment', 'preparation', 'ingredient_string']]
    ingredients_df['name.name'] = parallel_apply(
        ingredients_df[['name.name', 'name.description']],
        lambda ingredient: ingredient['name.name'] if pd.notnull(ingredient['name.name']) else ingredient['name.description'],
        meta=pd.Series(dtype='string'),
    )

    ingredients_df = ingredients_df.replace(r'^\s*$', pd.NA, regex=True)

    ingredients_df = ingredients_df.astype({
        'name.name': 'string',
        'name.description': 'string',
        'quantity': 'string',
        'unit': 'string',
        'comment': 'string',
        'preparation': 'string',
        'ingredient_string': 'string'
    })

    return ingredients_df


def clean_columns(ingredients_df, **kargs):
    
    for c in ['name.name', 'name.description', 'comment', 'unit']: 
        ingredients_df[c] = parallel_apply(ingredients_df[c], clean_ingredient_string, npartitions=100, meta=pd.Series(dtype='string'))
        logger.info(f"COMPLETED COLUMN {c}")

    ingredients_df['quantity'] = parallel_apply(
        ingredients_df['quantity'],
        clean_quantity,
        meta=pd.Series(dtype="Float64")
    )

    ingredients_df['quantity'] = ingredients_df['quantity'].astype('string')
        
    return ingredients_df

def tag_units_stage(ingredients_df, **kargs):
    
    unit_tags = parallel_apply_chunks(
        ingredients_df['unit'], 
        tag_units, 
        meta=pd.Series(dtype='object'), 
        npartitions=500,
        chunksize=3e6,
        **kargs)
    
    unit_tags = unit_tags.apply(literal_eval)

    ingredients_df[['unit_tags', 'unit_remainders', 'unit_type']] = pd.DataFrame(unit_tags.tolist(), index=unit_tags.index)

    ingredients_df['unit_type'] = ingredients_df['unit_type'].astype('category')
    
    return ingredients_df


def main():

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--from_stage', type=int, default=0)
    args = p.parse_args()

    stages = [
        create_from_recipes_df,
        parse_ingredient_string_preprocess,
        parse_ingredient_string_parsing,
        parse_ingredient_string_postprocess,
        clean_columns,
        tag_units_stage
    ]

    logger.info("Initialising files")
    save_dir = Path(f'{root}/data/local/recipe/full/ingredients')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframe")
    if args.from_stage:
        stage = args.from_stage-1
        stage_name = stages[stage].__name__
        ingredients_df = pd.read_feather(save_dir/f'{stage}_{stage_name}.feather')
    else: 
        ingredients_df = pd.read_csv(f'{root}/data/datasets/recipe/recipes_nlg/full_dataset.csv')

    for i, stage in enumerate(stages):
        if i < args.from_stage:
            continue
        logger.info(f"# COMMENCING STAGE {i} ({stage.__name__}) #")
        save_path = save_dir/f'{i}_{stage.__name__}'
        ingredients_df = stage(ingredients_df, save_path=str(save_path)+'_function.feather')
        ingredients_df.to_feather(str(save_path)+'.feather')

    return ingredients_df

if __name__ == '__main__':
    main()
