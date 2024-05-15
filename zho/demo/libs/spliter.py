'''
[id] 20240427035238
[desc] 1) 调用LINUX split命令对语料库进行切分
[dep] 1) C:/cygwin64/bin/split.exe(>=8.x)
[usage] 1) standalone
'''

import subprocess
from pathlib import Path
import os
import inspect
import argparse
from typing import Union, Any
import shutil

def main(corpus_filename: Union[str,Path], lines_per_split:int, index_length:int, split_savepath:Union[str,Path]):
    corpus_filename = Path(corpus_filename)
    split_savepath = Path(split_savepath)
    cmd_res = subprocess.run(f'C:/cygwin64/bin/split.exe -l {lines_per_split} --numeric-suffix=1 -a {index_length} "{str(corpus_filename.absolute())}" "{str(corpus_filename.parent)}/{corpus_filename.stem}_" --additional-suffix=.split', shell=True)
    move_cmd = f'move "{corpus_filename.parent}\\*.split" "{split_savepath}">Nul'
    move_res = subprocess.run(move_cmd, shell=True)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('lines_per_split')
    parser.add_argument('index_length', help='number of digits in split index')
    args = parser.parse_args()
    main(args.filename, int(args.lines_per_split), int(args.index_length))