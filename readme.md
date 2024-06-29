Repository for the [recipe-generator](https://github.com/stephankostov/recipe-generator) dataset creation.

# Key Features

- Dataset Wrangling in Pandas
    - Homogenising quantity units
    - Structuring appropriately
    - Handling NA values
- Fuzzy Matching datasets by food name
    - Creation of generalised framework for Fuzzy Matching Datasets including testing for evaluation of iterations
    - Recipe -> Density DB
    - Recipe -> Molecule DB
- Feature Engineering 
    - Parsing ingredient features from unstructured ingredient strings with Named Entity Recognition model
    - Feature selection of most relevant molecule compounds based on various criteria
- Parallelisation of all data processing with distributed data library [Dask](https://www.dask.org/) 
- Recipe ingredient tokenisation through similarity of their word embedded vectors

# Data Sources

- Recipes: [RecipeNLG](https://recipenlg.cs.put.poznan.pl/)
- Molecular Compositions: [FooDB](https://foodb.ca/)
- Density: [USDA FoodData Central](https://fdc.nal.usda.gov/)
- Molar Masses: [PubChem Compound](https://pubchem.ncbi.nlm.nih.gov/#query=)

# Development Process

1. Initial development (using small dataframe samples) [./notebooks](./notebooks)
2. Core functions automatically exported to [./food_database](./food_database) module using [nbdev](https://nbdev.fast.ai/getting_started.html)
2. Testing framework run [./tests](./full_tests)
3. Full dataframe pipelines [./full_runs](./full_runs)

# TODO

- Processing Pipeline Visualisation: The preprocessing/wrangling of the data ended up getting quite complex - is there a tool to map out this in a flow-diagram?
- Feature Engineering: Investigate optimal number of compounds to include.