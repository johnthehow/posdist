import subprocess
import os
import re
from pathlib import Path
import multiprocessing
import time
import inspect

class timer(object):
    def __enter__(self):
        self.time_start = time.time()
    def __exit__(self,exc_type, exc_val,exc_tb):
        self.time_end = time.time()
        print(f'duration: {self.time_end - self.time_start}')

def worker(cmd):
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] {cmd}')
    subprocess.run(cmd, shell=True, capture_output=True, check=True)
    return


def main(todel_file, token_path, n_worker):

    with open(todel_file, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(sep='\t') for line in lines]
    os.chdir(token_path)
    token_filenames = os.listdir()
    token_filename0 = token_filenames[0]
    digit_range = re.search('_\\d+\\.',token_filename0).span()
    digit_range = (digit_range[0]+1,digit_range[1]-1)
    cmds = []
    for fn,ln in lines:
        full_filename = token_filename0[:digit_range[0]] + fn + token_filename0[digit_range[1]:]
        cmd = f'sed -i "{ln}d" {full_filename}'
        cmds.append(cmd)

    pool_args = [(cmd,) for cmd in cmds]
    with multiprocessing.Pool(n_worker) as pool:
        pool.starmap(worker, pool_args)
    return

if __name__ == '__main__':
    todel_file = r"C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\standalone\todel.txt"
    token_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_FULL\TOKEN'
    n_worker = 4
    with timer() as tm:
        main(todel_file, token_path, n_worker)