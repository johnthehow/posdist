'''
[id] 20240516022842
[desc] 得到一个词性在一个句长中的线性位置分布的样本原始数据
[usage]
    1) ifname=main式独立运行
    2) import
[comment]
    1) 默认使用多进程并行
'''

import os
from pathlib import Path
import inspect
import matplotlib.pyplot as plot
import glob
import multiprocessing


def worker(tokenpos_filename, target_pos, sentlen):
    with open(tokenpos_filename, mode='r', encoding='utf-8') as file:
        file_pos_poss = [] # 一个文件中目标词类的位置们
        while True:
            line = file.readline().strip()
            line_split = line.split()
            line_pos_poss = [] # 一句话中目标词类的位置们
            if len(line_split) == sentlen:
                for idx,tokenpos in enumerate(line_split):
                    token, pos = tokenpos.split(sep=':')
                    if pos == target_pos:
                        line_pos_poss.append(idx+1)
            file_pos_poss += line_pos_poss
            if not line:
                break
        file_pos_poss = [str(i) for i in file_pos_poss] # 为了后期写入文件, 将每个元素都转换成字符串
    return file_pos_poss


def main(tokenpos_filepath, target_pos, sentlen, output_filename, n_worker):
    filenames = glob.glob(str(Path(tokenpos_filepath).joinpath(f'*.{"tokenpos"}').absolute()))
    pool_args = [(fn, target_pos, sentlen) for fn in filenames]
    with multiprocessing.Pool(n_worker) as pool:
        results = pool.starmap(worker, pool_args)
    with open(output_filename, mode='w', encoding='utf-8') as file:
        file.write(' '.join([' '.join(res) for res in results]))


if __name__ == '__main__':
    tokenpos_filepath = 'E:/同步空间/作业与校务_镜像/博士/研究/POSDIST_BERT/中间阶段成果/东亚语言位频分布/ZH/UNPC/UNPC_FULL/TOKENPOS'
    target_pos = 'PRON'
    sentlen = 20
    output_filename = fr'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\utils\{target_pos}_{sentlen}.posposdist'
    n_worker = 4
    main(tokenpos_filepath, target_pos, sentlen, output_filename, n_worker)
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] done')