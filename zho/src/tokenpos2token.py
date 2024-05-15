import re
import multiprocessing
from pathlib import Path
from typing import Any, Union
import glob


def worker(tokenpos_filename:Union[str,Path], token_path:Union[str,Path]):
    tokenpos_filename = Path(tokenpos_filename)
    token_filename = Path(token_path).joinpath(tokenpos_filename.stem+'.token')

    with open(tokenpos_filename, mode='r', encoding='utf-8') as file:
        text = file.read()
        output = re.sub(':[A-Z]+', '', text)
    with open(token_filename, mode='w', encoding='utf-8') as file:
        file.write(output)
    return


def main(tokenpos_path:Union[str,Path], token_path:Union[str,Path], n_worker:int):
    filenames = glob.glob(str(Path(tokenpos_path).joinpath(f'*.{"tokenpos"}').absolute()))
    pool_args = [(fn, token_path) for fn in filenames]
    with multiprocessing.Pool(processes=n_worker) as pool:
        res = pool.starmap(worker, pool_args)
    return