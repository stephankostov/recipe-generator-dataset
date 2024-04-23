# Autogenerated by nbdev

d = { 'settings': { 'branch': '',
                'doc_baseurl': '/recipe_dataset',
                'doc_host': 'https://.github.io',
                'git_url': 'https://github.com//recipe_dataset',
                'lib_path': 'recipe_dataset'},
  'syms': { 'recipe_dataset.density.finalise': { 'recipe_dataset.density.finalise.get_gram_weight': ( '06-density-db-finalise.html#get_gram_weight',
                                                                                                      'recipe_dataset/density/finalise.py')},
            'recipe_dataset.density.food_match': { 'recipe_dataset.density.food_match.calculate_search_stats': ( '04-density-db-food-match.html#calculate_search_stats',
                                                                                                                 'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.create_na_synonyms_df': ( '04-density-db-food-match.html#create_na_synonyms_df',
                                                                                                                'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.fuzzy_search': ( '04-density-db-food-match.html#fuzzy_search',
                                                                                                       'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.get_column_synonyms': ( '04-density-db-food-match.html#get_column_synonyms',
                                                                                                              'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.match_ingredient': ( '04-density-db-food-match.html#match_ingredient',
                                                                                                           'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.search_food_df': ( '04-density-db-food-match.html#search_food_df',
                                                                                                         'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.select_cols': ( '04-density-db-food-match.html#select_cols',
                                                                                                      'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.select_from_search_ids': ( '04-density-db-food-match.html#select_from_search_ids',
                                                                                                                 'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.select_from_searches': ( '04-density-db-food-match.html#select_from_searches',
                                                                                                               'recipe_dataset/density/food_match.py'),
                                                   'recipe_dataset.density.food_match.transform_ingredient': ( '04-density-db-food-match.html#transform_ingredient',
                                                                                                               'recipe_dataset/density/food_match.py')},
            'recipe_dataset.density.portion_match': { 'recipe_dataset.density.portion_match.select_food_portion': ( '05-density-db-portion-match.html#select_food_portion',
                                                                                                                    'recipe_dataset/density/portion_match.py')},
            'recipe_dataset.final.process': { 'recipe_dataset.final.process.merge_duplicates': ( '10-final-db-process.html#merge_duplicates',
                                                                                                 'recipe_dataset/final/process.py'),
                                              'recipe_dataset.final.process.process_food_ids': ( '10-final-db-process.html#process_food_ids',
                                                                                                 'recipe_dataset/final/process.py'),
                                              'recipe_dataset.final.process.shift_food_ids': ( '10-final-db-process.html#shift_food_ids',
                                                                                               'recipe_dataset/final/process.py')},
            'recipe_dataset.join_utils': { 'recipe_dataset.join_utils.clean_alt_words': ( '99-join-utils.html#clean_alt_words',
                                                                                          'recipe_dataset/join_utils.py'),
                                           'recipe_dataset.join_utils.clean_word': ( '99-join-utils.html#clean_word',
                                                                                     'recipe_dataset/join_utils.py'),
                                           'recipe_dataset.join_utils.find_alt_words': ( '99-join-utils.html#find_alt_words',
                                                                                         'recipe_dataset/join_utils.py'),
                                           'recipe_dataset.join_utils.flatten_list': ( '99-join-utils.html#flatten_list',
                                                                                       'recipe_dataset/join_utils.py'),
                                           'recipe_dataset.join_utils.get_food_hypernyms': ( '99-join-utils.html#get_food_hypernyms',
                                                                                             'recipe_dataset/join_utils.py'),
                                           'recipe_dataset.join_utils.get_synset': ( '99-join-utils.html#get_synset',
                                                                                     'recipe_dataset/join_utils.py')},
            'recipe_dataset.logger': { 'recipe_dataset.logger.configure_logger': ( '99-logger.html#configure_logger',
                                                                                   'recipe_dataset/logger.py')},
            'recipe_dataset.molecule.finalise': {},
            'recipe_dataset.molecule.match': { 'recipe_dataset.molecule.match.calculate_match_stats': ( '09-molecule-db-match.html#calculate_match_stats',
                                                                                                        'recipe_dataset/molecule/match.py'),
                                               'recipe_dataset.molecule.match.create_na_synonyms_df': ( '09-molecule-db-match.html#create_na_synonyms_df',
                                                                                                        'recipe_dataset/molecule/match.py'),
                                               'recipe_dataset.molecule.match.default_transform': ( '09-molecule-db-match.html#default_transform',
                                                                                                    'recipe_dataset/molecule/match.py'),
                                               'recipe_dataset.molecule.match.find_ingredient_food_df_matches': ( '09-molecule-db-match.html#find_ingredient_food_df_matches',
                                                                                                                  'recipe_dataset/molecule/match.py'),
                                               'recipe_dataset.molecule.match.fuzzy_search_words': ( '09-molecule-db-match.html#fuzzy_search_words',
                                                                                                     'recipe_dataset/molecule/match.py'),
                                               'recipe_dataset.molecule.match.match_ingredient': ( '09-molecule-db-match.html#match_ingredient',
                                                                                                   'recipe_dataset/molecule/match.py'),
                                               'recipe_dataset.molecule.match.select_from_matches': ( '09-molecule-db-match.html#select_from_matches',
                                                                                                      'recipe_dataset/molecule/match.py')},
            'recipe_dataset.parallel': { 'recipe_dataset.parallel.chunk_df': ('99-parallel.html#chunk_df', 'recipe_dataset/parallel.py'),
                                         'recipe_dataset.parallel.compile_chunks': ( '99-parallel.html#compile_chunks',
                                                                                     'recipe_dataset/parallel.py'),
                                         'recipe_dataset.parallel.convert_size': ( '99-parallel.html#convert_size',
                                                                                   'recipe_dataset/parallel.py'),
                                         'recipe_dataset.parallel.initialize_chunk_dir': ( '99-parallel.html#initialize_chunk_dir',
                                                                                           'recipe_dataset/parallel.py'),
                                         'recipe_dataset.parallel.parallel_apply': ( '99-parallel.html#parallel_apply',
                                                                                     'recipe_dataset/parallel.py'),
                                         'recipe_dataset.parallel.parallel_apply_chunks': ( '99-parallel.html#parallel_apply_chunks',
                                                                                            'recipe_dataset/parallel.py')},
            'recipe_dataset.recipes.create': { 'recipe_dataset.recipes.create.clean_quantity': ( '01-recipes-db-process.html#clean_quantity',
                                                                                                 'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.filter_patterns': ( '01-recipes-db-process.html#filter_patterns',
                                                                                                  'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.find_ner_match': ( '01-recipes-db-process.html#find_ner_match',
                                                                                                 'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.get_match_idxs': ( '01-recipes-db-process.html#get_match_idxs',
                                                                                                 'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.is_number': ( '01-recipes-db-process.html#is_number',
                                                                                            'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.ner_preprocess_ingredient_string': ( '01-recipes-db-process.html#ner_preprocess_ingredient_string',
                                                                                                                   'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.parse_ingredient_string': ( '01-recipes-db-process.html#parse_ingredient_string',
                                                                                                          'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.pipeline_parse_ingredient_string_parsing': ( '01-recipes-db-process.html#pipeline_parse_ingredient_string_parsing',
                                                                                                                           'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.preprocess_cup_units': ( '01-recipes-db-process.html#preprocess_cup_units',
                                                                                                       'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.preprocess_remove_ors': ( '01-recipes-db-process.html#preprocess_remove_ors',
                                                                                                        'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.remove_name_from_description': ( '01-recipes-db-process.html#remove_name_from_description',
                                                                                                               'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.split_ingredient_fields_by_noun': ( '01-recipes-db-process.html#split_ingredient_fields_by_noun',
                                                                                                                  'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.split_nouns': ( '01-recipes-db-process.html#split_nouns',
                                                                                              'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.tag_ingredient_string': ( '01-recipes-db-process.html#tag_ingredient_string',
                                                                                                        'recipe_dataset/recipes/create.py'),
                                               'recipe_dataset.recipes.create.tokenize_with_spans': ( '01-recipes-db-process.html#tokenize_with_spans',
                                                                                                      'recipe_dataset/recipes/create.py')},
            'recipe_dataset.train.load_data': { 'recipe_dataset.train.load_data.MaskedRecipeDataset': ( 'training/loading-data.html#maskedrecipedataset',
                                                                                                        'recipe_dataset/train/load_data.py'),
                                                'recipe_dataset.train.load_data.MaskedRecipeDataset.__getitem__': ( 'training/loading-data.html#maskedrecipedataset.__getitem__',
                                                                                                                    'recipe_dataset/train/load_data.py'),
                                                'recipe_dataset.train.load_data.MaskedRecipeDataset.__init__': ( 'training/loading-data.html#maskedrecipedataset.__init__',
                                                                                                                 'recipe_dataset/train/load_data.py'),
                                                'recipe_dataset.train.load_data.MaskedRecipeDataset.__len__': ( 'training/loading-data.html#maskedrecipedataset.__len__',
                                                                                                                'recipe_dataset/train/load_data.py'),
                                                'recipe_dataset.train.load_data.MaskedRecipeDataset.create_recipe_mask': ( 'training/loading-data.html#maskedrecipedataset.create_recipe_mask',
                                                                                                                           'recipe_dataset/train/load_data.py')},
            'recipe_dataset.utils': { 'recipe_dataset.utils.clean_ingredient_string': ( '99-utils.html#clean_ingredient_string',
                                                                                        'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.clear_variable_cache': ( '99-utils.html#clear_variable_cache',
                                                                                     'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.contains_whole_word': ( '99-utils.html#contains_whole_word',
                                                                                    'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.convert_fractions_to_decimal': ( '99-utils.html#convert_fractions_to_decimal',
                                                                                             'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.count_list_matches': ( '99-utils.html#count_list_matches',
                                                                                   'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.detokenize': ('99-utils.html#detokenize', 'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.get_unit_type': ('99-utils.html#get_unit_type', 'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.join_repeated_numeric_tags': ( '99-utils.html#join_repeated_numeric_tags',
                                                                                           'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.show_var_sizes': ('99-utils.html#show_var_sizes', 'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.sizeof_fmt': ('99-utils.html#sizeof_fmt', 'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.tag_numerics_with_float_value': ( '99-utils.html#tag_numerics_with_float_value',
                                                                                              'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.tag_units': ('99-utils.html#tag_units', 'recipe_dataset/utils.py'),
                                      'recipe_dataset.utils.train_unit_tagger': ( '99-utils.html#train_unit_tagger',
                                                                                  'recipe_dataset/utils.py')},
            'recipe_dataset.utils.full_run_utils': { 'recipe_dataset.utils.full_run_utils.load_or_create_dataframe': ( 'utils/full-run-utils.html#load_or_create_dataframe',
                                                                                                                       'recipe_dataset/utils/full_run_utils.py'),
                                                     'recipe_dataset.utils.full_run_utils.select_last_file': ( 'utils/full-run-utils.html#select_last_file',
                                                                                                               'recipe_dataset/utils/full_run_utils.py')},
            'recipe_dataset.utils.join_utils': { 'recipe_dataset.utils.join_utils.clean_alt_words': ( 'utils/join-utils.html#clean_alt_words',
                                                                                                      'recipe_dataset/utils/join_utils.py'),
                                                 'recipe_dataset.utils.join_utils.clean_word': ( 'utils/join-utils.html#clean_word',
                                                                                                 'recipe_dataset/utils/join_utils.py'),
                                                 'recipe_dataset.utils.join_utils.find_alt_words': ( 'utils/join-utils.html#find_alt_words',
                                                                                                     'recipe_dataset/utils/join_utils.py'),
                                                 'recipe_dataset.utils.join_utils.flatten_list': ( 'utils/join-utils.html#flatten_list',
                                                                                                   'recipe_dataset/utils/join_utils.py'),
                                                 'recipe_dataset.utils.join_utils.get_food_hypernyms': ( 'utils/join-utils.html#get_food_hypernyms',
                                                                                                         'recipe_dataset/utils/join_utils.py'),
                                                 'recipe_dataset.utils.join_utils.get_synset': ( 'utils/join-utils.html#get_synset',
                                                                                                 'recipe_dataset/utils/join_utils.py')},
            'recipe_dataset.utils.logger': { 'recipe_dataset.utils.logger.configure_logger': ( 'utils/logger.html#configure_logger',
                                                                                               'recipe_dataset/utils/logger.py')},
            'recipe_dataset.utils.parallel': { 'recipe_dataset.utils.parallel.chunk_df': ( 'utils/parallel.html#chunk_df',
                                                                                           'recipe_dataset/utils/parallel.py'),
                                               'recipe_dataset.utils.parallel.compile_chunks': ( 'utils/parallel.html#compile_chunks',
                                                                                                 'recipe_dataset/utils/parallel.py'),
                                               'recipe_dataset.utils.parallel.convert_size': ( 'utils/parallel.html#convert_size',
                                                                                               'recipe_dataset/utils/parallel.py'),
                                               'recipe_dataset.utils.parallel.initialize_chunk_dir': ( 'utils/parallel.html#initialize_chunk_dir',
                                                                                                       'recipe_dataset/utils/parallel.py'),
                                               'recipe_dataset.utils.parallel.parallel_apply': ( 'utils/parallel.html#parallel_apply',
                                                                                                 'recipe_dataset/utils/parallel.py'),
                                               'recipe_dataset.utils.parallel.parallel_apply_chunks': ( 'utils/parallel.html#parallel_apply_chunks',
                                                                                                        'recipe_dataset/utils/parallel.py'),
                                               'recipe_dataset.utils.parallel.parallel_df_function': ( 'utils/parallel.html#parallel_df_function',
                                                                                                       'recipe_dataset/utils/parallel.py')},
            'recipe_dataset.utils.utils': { 'recipe_dataset.utils.utils.clean_ingredient_string': ( 'utils/utils.html#clean_ingredient_string',
                                                                                                    'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.clear_variable_cache': ( 'utils/utils.html#clear_variable_cache',
                                                                                                 'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.contains_whole_word': ( 'utils/utils.html#contains_whole_word',
                                                                                                'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.convert_fractions_to_decimal': ( 'utils/utils.html#convert_fractions_to_decimal',
                                                                                                         'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.count_list_matches': ( 'utils/utils.html#count_list_matches',
                                                                                               'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.detokenize': ( 'utils/utils.html#detokenize',
                                                                                       'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.get_unit_type': ( 'utils/utils.html#get_unit_type',
                                                                                          'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.join_repeated_numeric_tags': ( 'utils/utils.html#join_repeated_numeric_tags',
                                                                                                       'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.show_var_sizes': ( 'utils/utils.html#show_var_sizes',
                                                                                           'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.sizeof_fmt': ( 'utils/utils.html#sizeof_fmt',
                                                                                       'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.tag_numerics_with_float_value': ( 'utils/utils.html#tag_numerics_with_float_value',
                                                                                                          'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.tag_units': ( 'utils/utils.html#tag_units',
                                                                                      'recipe_dataset/utils/utils.py'),
                                            'recipe_dataset.utils.utils.train_unit_tagger': ( 'utils/utils.html#train_unit_tagger',
                                                                                              'recipe_dataset/utils/utils.py')}}}
