# AUTOGENERATED! DO NOT EDIT! File to edit: ../../notebooks/training/loading-data.ipynb.

# %% auto 0
__all__ = ['root', 'food_compounds_df', 'special_token_idxs', 'MaskedRecipeDataset']

# %% ../../notebooks/training/loading-data.ipynb 3
from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

# %% ../../notebooks/training/loading-data.ipynb 4
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter
import random
from itertools import islice

# %% ../../notebooks/training/loading-data.ipynb 6
food_compounds_df = pd.read_feather(f'{root}/../data/local/molecule/full/food_compounds/0.feather')

# %% ../../notebooks/training/loading-data.ipynb 7
special_token_idxs = {
    'pad': food_compounds_df.shape[0], 'mask': food_compounds_df.shape[0]+1
}

# %% ../../notebooks/training/loading-data.ipynb 26
class MaskedRecipeDataset(Dataset):

    def __init__(self, recipes):
        self.recipes = recipes
    
    def __len__(self):
        return len(self.recipes) # should be able to have more than one here
    
    def create_recipe_mask(self, recipe):
        recipe_size = (recipe != special_token_idxs['pad']).sum()
        mask = torch.tensor(False)
        while mask.sum() == torch.tensor(0):
            rand = torch.rand(recipe_size)
            mask = (rand < 0.15).to(bool)
        recipe[:recipe_size][mask] = special_token_idxs['mask']
        return recipe
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.to_list()
        recipe = self.recipes[idx]
        recipe = torch.tensor(recipe, dtype=torch.int)
        return (
            self.create_recipe_mask(recipe.clone()).type(torch.int), 
            recipe.type(torch.int)
        )
