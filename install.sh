conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas numpy nltk spacy tqdm word2number ingredients-parser-nlp pyarrow dask dask[distributed] seaborn thefuzz sentence_transformers
pip install git+https://github.com/stephankostov/parse-ingredients.git
python -m spacy download en_core_web_sm
