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

def initialize(ingredients_df):

    expanded_ingredients_df = ingredients_df[['name.name', 'name.description', 'ingredient_string']]
    expanded_ingredients_df = expanded_ingredients_df.astype('string')

    return expanded_ingredients_df

def split_nouns(expanded_ingredients_df, **kwargs):

    meta = pd.DataFrame({
        'name.name': pd.Series(dtype='string'),
        'name.description': pd.Series(dtype='string'),
        'ingredient_string': pd.Series(dtype='string'),
        'name.name.nouns': pd.Series(dtype='object'),
        'name.name.others': pd.Series(dtype='object'),
        'name.description.nouns': pd.Series(dtype='object'),
        'name.description.others': pd.Series(dtype='object')
    })
    expanded_ingredients_df = parallel_apply_chunks(
        expanded_ingredients_df, 
        split_ingredient_fields_by_noun, 
        meta=meta, 
        npartitions=300,
        chunksize=3e6,
        **kwargs
    )

    return expanded_ingredients_df

def split_nouns_postprocess(expanded_ingredients_df, save_path):
        
    expanded_ingredients_df.drop([col for col in expanded_ingredients_df.columns if not any(s in col for s in ['nouns', 'others'])], axis=1, inplace=True)

    return expanded_ingredients_df
    

def expand_columns(expanded_ingredients_df, save_path):

    if not isinstance(expanded_ingredients_df.dtypes.iloc[0], list): 
        expanded_ingredients_df = expanded_ingredients_df.map(literal_eval)

    # split expand lists into individual columns
    original_cols = expanded_ingredients_df.columns
    for expand_col in original_cols:
        expanded = pd.DataFrame(expanded_ingredients_df[expand_col].tolist(), index=expanded_ingredients_df.index)
        expanded.columns = [expand_col + '.' + str(c) for c in expanded.columns]
        expanded_ingredients_df = expanded_ingredients_df.join(expanded)
        logger.info(f'COMPLETED COLUMN: {expand_col}')

    expanded_ingredients_df.drop(columns=original_cols, inplace=True)
    expanded_ingredients_df = expanded_ingredients_df.astype('string[pyarrow]')
    
    return expanded_ingredients_df

def clean_description(expanded_ingredients_df, save_path):

    meta = pd.DataFrame(columns=['name.name.nouns.0', 'name.name.nouns.1', 'name.name.nouns.2', 'name.name.nouns.3', 'name.name.nouns.4', 'name.name.nouns.5', 'name.name.others.0', 'name.name.others.1', 'name.name.others.2', 'name.name.others.3', 'name.name.others.4', 'name.name.others.5', 'name.description.nouns.0', 'name.description.nouns.1', 'name.description.nouns.2', 'name.description.nouns.3', 'name.description.nouns.4', 'name.description.nouns.5', 'name.description.others.0', 'name.description.others.1', 'name.description.others.2', 'name.description.others.3', 'name.description.others.4', 'name.description.others.5'], dtype='string')
    expanded_ingredients_df = parallel_apply(expanded_ingredients_df, remove_name_from_description, meta=meta, npartitions=2000)

    return expanded_ingredients_df

def clean_columns(expanded_ingredients_df, save_path):

    checkpoint_dir = save_path/'cleaned_columns'
    checkpoint_dir.mkdir(exist_ok=True)

    for col in expanded_ingredients_df.columns:
        save_path = checkpoint_dir/f'{col}.feather'
        if save_path.exists(): 
            expanded_ingredients_df[col] = pd.read_feather(save_path).iloc[0,:]
            continue
        logger.info(f'CLEANING COLUMN: {col}')
        expanded_ingredients_df[col] = parallel_apply(expanded_ingredients_df[col], clean_ingredient_string, meta=pd.Series(dtype='string[pyarrow]'), npartitions=100)
        expanded_ingredients_df[col].to_frame().to_feather(save_path)

    expanded_ingredients_df = expanded_ingredients_df.replace(r'^\s*$', pd.NA, regex=True)

    return expanded_ingredients_df

def rearrange_columns(expanded_ingredients_df, save_path):

    split_types = ['nouns', 'others']
    original_cols = list(dict.fromkeys([re.sub(r'\.(nouns|others).*', '', c) for c in expanded_ingredients_df.columns]))
    reordered_cols = []
    for col in original_cols:
        for word_type in split_types:
            col_expanded = [c for c in expanded_ingredients_df.columns if col in c and word_type in c]
            if word_type =='nouns': col_expanded.reverse()
            reordered_cols.extend(col_expanded)
        
    expanded_ingredients_df = expanded_ingredients_df[reordered_cols]
    return expanded_ingredients_df

def main():

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--from_stage', type=int, default=0)
    script_args = p.parse_args()

    save_dir = Path(f'{root}/data/local/recipe/full/expanded_ingredients')

    stages = [
        initialize,
        split_nouns,
        split_nouns_postprocess,
        expand_columns,
        clean_description,
        clean_columns,
        rearrange_columns,
    ]

    logger.info("Initialising files")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataframe")
    if script_args.from_stage:
        stage = script_args.from_stage-1
        stage_name = list(stages.keys())[stage].__name__
        expanded_ingredients_df = pd.read_feather(save_dir/f'{stage}_{stage_name}.feather', dtype_backend='pyarrow')
    else: 
        expanded_ingredients_df = pd.read_feather(f'{save_dir}/../ingredients/3_parse_ingredient_string_postprocess.feather') # stage before column cleaning

    for i, stage in enumerate(stages):
        if i < script_args.from_stage: continue
        logger.info(f"# COMMENCING STAGE {i} ({stage.__name__}) #")
        save_path = save_dir/f'{i}_{stage.__name__}'
        expanded_ingredients_df = stage(expanded_ingredients_df, save_path=str(save_path)+'_function.feather')
        expanded_ingredients_df.to_feather(str(save_path)+'.feather')

    return expanded_ingredients_df

if __name__ == '__main__':
    main()