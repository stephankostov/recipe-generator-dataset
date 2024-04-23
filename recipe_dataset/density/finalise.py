# AUTOGENERATED! DO NOT EDIT! File to edit: ../../notebooks/06-density-db-finalise.ipynb.

# %% auto 0
__all__ = ['root', 'get_gram_weight']

# %% ../../notebooks/06-density-db-finalise.ipynb 4
from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

# %% ../../notebooks/06-density-db-finalise.ipynb 5
import pandas as pd
import numpy as np

import json

# %% ../../notebooks/06-density-db-finalise.ipynb 11
with open(f'{root}/config/unit_conversions.json') as f:
    unit_list = json.load(f)

# %% ../../notebooks/06-density-db-finalise.ipynb 18
def get_gram_weight(ingredient, food_portion_df, unit_list):

    weight = 0.0

    if ingredient['unit_type'] != 'weight' and (pd.isnull(ingredient['food_id']) or pd.isnull(ingredient['food_portion_id'])): return pd.NA

    if ingredient['unit_type'] == 'weight':
        
        ingredient_weight_unit = [unit for unit in ingredient['unit_tags'] if unit in unit_list['weight'].keys()][0]
        weight = unit_list['weight'][ingredient_weight_unit]['conversion'] * ingredient['quantity']

    else:

        portion = food_portion_df.loc[ingredient['food_id'], ingredient['food_portion_id']]

        if ingredient['unit_type'] == 'volume':

            ingredient_volume_unit = [unit for unit in ingredient['unit_tags'] if unit in unit_list['volume'].keys()][0]

            if portion['unit_type'] == 'volume':

                portion_volume_unit = [unit for unit in portion['unit_tags'] if unit in unit_list['volume'].keys()][0]
                # simple density calculation if exists
                if not pd.notnull(portion['portion_amount']):
                    density = portion['gram_weight'] / (portion['amount'] * unit_list['volume'][portion_volume_unit]['conversion'])
                else:
                    density = portion['gram_weight'] / (portion['amount'] * portion['portion_amount'] * unit_list['volume'][portion_volume_unit]['conversion']) # #todo can just make porion_amount == 1 or factor this in the amount when creating dataframe
                weight = unit_list['volume'][ingredient_volume_unit]['conversion'] * density * ingredient['quantity']

            else:

                # volume measurement not given -> must be portion (set to NA for now)
                weight = pd.NA

        else: # ingredient whole/portion measurements

            weight = portion['gram_weight'] * ingredient['quantity']


    return weight    
