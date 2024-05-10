'''
[id] 20240430152643
[desc]
    1) 统计空格分词的语料库(多个子文件)的句长分布, 生成折线图
    2) 支持多进程, 因而当语料库很大时, 应当将语料库拆成多个小文件
[usage]
    1) import.main
    2) ifname=main
    3) ifname=main >>> python this.py 4 D:\test.png
'''

from typing import Union, Any, Callable, Sequence, Mapping
from pathlib import Path
from collections import Counter
import inspect
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import glob


def count(filename:str):
    lens = []
    with open(filename, mode='r', encoding='utf-8') as file:
        while True:
            line = file.readline().strip()
            line_tokenized = line.split(sep=' ')
            lens.append(len(line_tokenized))
            if not line:
                break
    return lens

def multiproc(func, iterables, n_workers):
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(func, iterables)
        results = [i for result in results for i in result]
    return results

def main(filepath:Union[str,Path], n_worker:int, img_filename:Union[str,Path]):
    filenames = glob.glob(str(Path(filepath).joinpath(f'*.{"token"}').absolute()))
    iterables = [(i,) for i in filenames]
    counts = multiproc(count, iterables, n_worker)
    pmf = Counter(counts)
    pmf = dict(sorted(pmf.items(), key=lambda x: x[0]))
    complete_pmf = {k:0 for k in range(min(counts),max(counts)+1)} 
    complete_pmf.update(pmf)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    xs = list(complete_pmf.keys())
    ys = list(complete_pmf.values())
    fig = plt.figure(figsize=[6.4,4.8])
    ax = fig.subplots()
    ax.set_title('语言:汉语')
    ax.set_xlabel('句长')
    ax.set_ylabel('频次')
    ax.plot(xs,ys)
    ax.set_xticks(list(range(1,len(xs),2)))
    ax.set_xticklabels(list(range(1,len(xs),2)),rotation=90)
    plt.savefig(img_filename)
    plt.close()
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_worker')
    parser.add_argument('img_filename')
    args = parser.parse_args()
    filenames = select_files()
    main(filenames, int(args.n_worker), args.img_filename)

    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}] done')

