# AUTOGENERATED! DO NOT EDIT! File to edit: ../../notebooks/utils/join-utils.ipynb.

# %% auto 0
__all__ = ['root', 'excluded_words', 'get_synset', 'get_food_hypernyms', 'flatten_list', 'clean_word', 'clean_alt_words',
           'find_alt_words']

# %% ../../notebooks/utils/join-utils.ipynb 2
from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

# %% ../../notebooks/utils/join-utils.ipynb 3
from nltk.corpus import wordnet
import warnings
from .utils import *

# %% ../../notebooks/utils/join-utils.ipynb 5
def get_synset(ingredient):

    synsets = wordnet.synsets(ingredient)
    if not synsets: return None

    filtered = [w for w in synsets if 'food' in w.lexname()]
    if filtered: synsets = filtered
    filtered = [w for w in synsets if ingredient in w.name()]
    if filtered: synsets = filtered

    return synsets[0]

# %% ../../notebooks/utils/join-utils.ipynb 8
excluded_words = [
    'cut'
]

# %% ../../notebooks/utils/join-utils.ipynb 10
def get_food_hypernyms(synset):

    with warnings.catch_warnings(): # closure throws warning if it exceeds depth limit
        warnings.simplefilter("ignore")
        hypernyms = list(synset.closure(lambda x: x.hypernyms(), depth=5))

    hypernyms = [ word.name().split('.')[0] for word in hypernyms ]
    
    try:
        hypernyms = hypernyms[:(hypernyms.index('food'))] 
    except ValueError:
        pass
    
    hypernyms = hypernyms[:5]
    
    return hypernyms

def flatten_list(l):
    return [x for xs in l for x in xs]

def clean_word(word):
    tokens = mt.tokenize(word)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return detokenize(tokens)

def clean_alt_words(alt_words):
    alt_words = [reversed(w.split('_')) for w in alt_words]
    alt_words = flatten_list(alt_words)
    alt_words = [clean_word(w) for w in alt_words]
    alt_words = [w for w in alt_words if w not in excluded_words]
    alt_words = list(filter(None, alt_words))
    return alt_words

def find_alt_words(word):

    if not isinstance(word, str) or word == '': return 

    synset = get_synset(word)
    if not synset: return
       
    synonyms = list(set(synset.lemma_names()) - {word})[:5]
    hypernyms = get_food_hypernyms(synset)

    alt_words = [ *synonyms, *hypernyms ]
    
    alt_words = clean_alt_words(alt_words)

    alt_words = alt_words[:5]

    return alt_words
