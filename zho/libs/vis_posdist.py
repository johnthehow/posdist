'''
1. 选择hdf5文件
2. 新建hdf5容器
3. 容器链接分散的hdf5文件
4. 给定单词和句长, 查询所有HDF5文件, 返回结果到一个列表
'''

import multiprocessing
import h5py
from collections import Counter
from pathlib import Path
import os
import inspect
import matplotlib.pyplot as plt
import argparse
import glob
from collections import namedtuple
import time

class timer(object):
    def __enter__(self):
        self.time_start = time.time()
    def __exit__(self,exc_type, exc_val,exc_tb):
        self.time_end = time.time()
        print(f'duration: {self.time_end - self.time_start}')

def main(posdist_filename, wordlist_filename, image_savepath, min_sentlen, max_sentlen, interval, density):
    '''
    最多绘制10个句长, 否则marker和线条样式不够用
    '''
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] loading posdist data')
    with timer() as tm:
        with open(posdist_filename, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            lines_split = [line.split() for line in lines]
    with open(wordlist_filename, mode='r', encoding='utf-8') as file:
        wordlist = file.readlines()
        wordlist = [line.strip() for line in wordlist]

    for word in wordlist:
        print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}] plotting word: {word}')
        image_savepath = Path(image_savepath)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig = plt.figure()
        ax = fig.subplots()
        ax.set_xlabel('句中线性位置')
        if density == True:
            ax.set_ylabel('概率')
        else:
            ax.set_ylabel('频数')
        ax.set_xticks(range(1,max_sentlen+1))
        ax.set_xticklabels(range(1,max_sentlen+1))

        markerstyle = ['+','x','D','s','o','^','v','4','*','p']
        style_cnt = 0
        for sentlen in range(min_sentlen,max_sentlen+1,interval):
            for line in lines_split:
                if line[0][1:-1] == word and int(line[1][1:-1]) == sentlen:
                    word_posdist_data = line[2:]
                    word_posdist_data = [int(i) for i in word_posdist_data]
                    pmf = dict(sorted(Counter(word_posdist_data).items(),key=lambda i:i[0]))
                    complete_pmf = {k:0 for k in range(1,sentlen+1)} 
                    complete_pmf.update(pmf)
                    if density == True:
                        xs = list(complete_pmf.keys())
                        ys = list(map(lambda x:x/sum(complete_pmf.values()),complete_pmf.values()))
                    else:
                        xs = list(pmf.keys())
                        ys = list(pmf.values())
                    ax.plot(xs, ys, label=str(sentlen), marker=markerstyle[style_cnt],markersize=4,fillstyle='none',linewidth=1,linestyle=(0,(10,style_cnt)))
            style_cnt += 1
        ax.set_title(f'功能词: {word}, 频数: {sum(complete_pmf.values())}')
        ax.legend(title='句长')
        plt.savefig(image_savepath.joinpath(f'{word}.png'))
        plt.close()
    return

    
if __name__ == '__main__':

    posdist_filename = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_MINI\POSDIST\cleaned_merged.posdist'
    wordlist_filename = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\WORDLIST\jp_10.wordlist'
    image_savepath = r'D:\test\img'
    main(posdist_filename, wordlist_filename, image_savepath, 14, 24, 2, True)