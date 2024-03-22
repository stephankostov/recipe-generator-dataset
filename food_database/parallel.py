# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/99-parallel.ipynb.

# %% auto 0
__all__ = ['root', 'logger', 'convert_size', 'parallel_apply', 'chunk_df', 'initialize_chunk_dir', 'parallel_apply_chunks',
           'compile_chunks']

# %% ../notebooks/99-parallel.ipynb 1
from pyprojroot import here
root = here()
import sys
sys.path.append(str(root))

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import multiprocessing as mp
import pandas as pd
import logging
import warnings
import datetime
from pathlib import Path
from .logger import *

import math

import logging
logger = configure_logger(logging.getLogger(__name__))

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def parallel_apply(df, func, meta, npartitions=int(0.9 * mp.cpu_count()), keep_index=False, **kargs):
    
    # resetting MultiIndex as not supported in Dask (applied back later)
    og_index = df.index
    df.reset_index(inplace=True, drop=(not keep_index))
    ddf = dd.from_pandas(df, npartitions=npartitions)

    memory_usage = ddf.memory_usage(deep=True, index=False).compute()
    if isinstance(memory_usage, pd.Series): memory_usage = memory_usage.sum()

    logger.info(f"Commencing parallel apply")
    logger.info(f"DF shape: {df.shape} | DF size: {convert_size(memory_usage)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with LocalCluster(n_workers=int(0.9 * mp.cpu_count()),
            processes=True,
            threads_per_worker=1,
            memory_limit='1.5GB',
        ) as cluster, Client(cluster) as client:
            results = ddf.apply(func, meta=meta, axis=1, **kargs).compute() if isinstance(df, pd.DataFrame) else ddf.apply(func, meta=meta, **kargs).compute()

    df.index = og_index
    results.index = og_index

    return results

def chunk_df(df, n=1e6):
    logger.info(f'Splitting dataframe into {df.shape[0]/n} chunks of size {n}')
    return [df[i:i+int(n)] for i in range(0,df.shape[0],int(n))]

def initialize_chunk_dir(save_path):
    save_path = Path(save_path)
    file_name = save_path.name
    if '.' in file_name: file_name = file_name.split('.')[0]
    chunk_dir = Path(save_path.parent)/f'{file_name}_chunks'
    chunk_dir.mkdir(exist_ok=True)
    return chunk_dir

def parallel_apply_chunks(df, func, meta, chunksize, save_path, npartitions=int(0.9 * mp.cpu_count()), keep_index=False, start_from=0, **kargs):

    chunk_dir = initialize_chunk_dir(save_path)
    chunks = chunk_df(df, chunksize)

    for i, chunk in enumerate(chunks):

        if i < start_from: continue
        if (chunk_dir/f"{i}.feather").exists(): continue

        logger.info(f'------------------')
        logger.info(f'COMMENCING CHUNK {i}')
        logger.info(f'------------------')

        result = parallel_apply(chunk, func, meta, npartitions, keep_index, **kargs)
        if isinstance(result, pd.Series): result = result.to_frame()
        result.to_feather(chunk_dir/f'{i}.feather')

    compiled_results = compile_chunks(save_path)

    return compiled_results

def compile_chunks(save_path):

    logger.info("Compiling chunks")

    save_path = Path(save_path)
    chunk_dir = initialize_chunk_dir(save_path)

    compiled_chunks = pd.DataFrame()
    for i in range(len(list(chunk_dir.iterdir()))):
        chunk = pd.read_feather(chunk_dir/f"{i}.feather")
        compiled_chunks = pd.concat([compiled_chunks, chunk], axis=0)

    compiled_chunks.to_feather(save_path)

    if compiled_chunks.shape[1] == 1: # convert to series if series
        compiled_chunks = compiled_chunks.iloc[:, 0]

    return compiled_chunks

