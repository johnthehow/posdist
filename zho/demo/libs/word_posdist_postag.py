import multiprocessing
import os
from pathlib import Path
import re
import inspect
import glob
import subprocess
import time

class timer(object):
    def __enter__(self):
        self.time_start = time.time()
    def __exit__(self,exc_type, exc_val,exc_tb):
        self.time_end = time.time()
        print(f'duration: {self.time_end - self.time_start}')

def create_nested_dict(keys_list):
    if not keys_list:
        return []
    result = {}
    for key in keys_list[0]:
        result[key] = create_nested_dict(keys_list[1:])
    return result


def worker(tokenpos_filename, posdist_pos_path, wordposlist, sentlens):
    tokenpos_filename = Path(tokenpos_filename)
    posdist_pos_path = Path(posdist_pos_path)
    posdist_filename = posdist_pos_path.joinpath(tokenpos_filename.name+'.posdist')
    cache_result_dict = create_nested_dict([wordposlist, sentlens])
    with open(tokenpos_filename, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if len(line.split()) >= min(sentlens) and len(line.split()) <= max(sentlens):
                for wordpos in wordposlist:
                    word, pos = wordpos.split(sep=':')
                    cache_result_dict[word][len(line.split())] += [pos+1 for pos, word_in_sent in enumerate(line.split()) if word_in_sent==word]
    with open(posdist_filename, mode='w', encoding='utf-8') as file:
        for word in wordposlist:
            for sentlen in sentlens:
                file.write(f'{" ".join([str(i) for i in cache_result_dict[word][sentlen]])}\n')
    return

def dos2unix_worker(posdist_filename):
    posdist_filename = Path(posdist_filename)
    os.chdir(posdist_filename.parent)
    cmd = f'dos2unix {posdist_filename} 1>nul 2>&1'
    subprocess.run(cmd, shell=True)
    return

def main(tokenpos_path, posdist_pos_path, wordlist_filename, n_worker, min_sentlen, max_sentlen):
    tokenpos_path = Path(tokenpos_path)
    posdist_pos_path = Path(posdist_pos_path)
    wordlist_filename = Path(wordlist_filename)


    # 为子进程准备参数
    token_filenames = glob.glob(str(Path(tokenpos_path).joinpath(f'*.{"token"}').absolute()))
    with open(wordlist_filename, mode='r', encoding='utf-8') as file:
        wordposlist = file.readlines()
        wordposlist = [w.strip() for w in wordposlist]
    sentlens = [i for i in range(min_sentlen, max_sentlen+1)]

    # 每个子进程统计一个文件的位频分布
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] counting posdist')
    with timer() as tm:
        pool_args = [(fn, posdist_pos_path, wordposlist, sentlens) for fn in token_filenames]
        with multiprocessing.Pool(n_worker) as pool:
            pool.starmap(worker, pool_args)

    # 拼接多个posdist文件为一个posdist文件
    # 为文件拼接准备行头
    header_filename = posdist_pos_path.joinpath('0000.posdist')
    with open(header_filename, mode='w', encoding='utf-8') as file:
        for word in wordposlist:
            for sentlen in sentlens:
                file.write(f'[{word}]\t[{sentlen:03d}]\t\n')

    os.chdir(posdist_pos_path) 
    # 转换文件行尾为\n, 否则其后的paste命令无法正确执行
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] converting CRLF to LF ')
    with timer() as tm:
        posdist_filenames = glob.glob(str(Path(posdist_pos_path).joinpath(f'*.{"posdist"}').absolute()))
        pool_args = [(fn,) for fn in posdist_filenames]
        with multiprocessing.Pool(n_worker) as pool:
            pool.starmap(dos2unix_worker, pool_args)
    # 拼接多个posdist文件为一个posdist文件
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] merging posdist files')
    cmd = f'paste -d " " *.posdist> merged.txt'
    with timer() as tm:
        subprocess.run(cmd, shell=True)
    cmd = f'sed "s/\\s\\+/ /g" merged.txt>cleaned_merged.posdist'
    subprocess.run(cmd, shell=True)
    # # 删除拼接产生的中间阶段文件
    os.remove(header_filename)
    os.remove('merged.txt')
    return
    

if __name__ == '__main__':
    main(tokenpos_path, posdist_pos_path, wordlist_filename, n_worker, min_sentlen, max_sentlen)