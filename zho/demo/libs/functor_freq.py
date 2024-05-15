'''
[id] 20240507120019
[desc]
    1) 根据spacy的词性标注结果, 提取文本中最高频的功能词
[usage]
    1) ifname
    2) import.main
[dep]
    1) spacy
    2) spacy加载之模型zh_core_web_sm, ja_core_news_sm, ko_core_news_sm
'''

from pathlib import Path
import os
import inspect
import glob
import multiprocessing
from collections import Counter

def worker(tokenpos_filename):
    tokenpos_filename = Path(tokenpos_filename)
    checklist_pos = ['ADP','SCONJ','X','AUX','CCONJ','PART','PRON','DET']
    res_dict = {k:[] for k in checklist_pos}
    with open(tokenpos_filename, mode='r', encoding='utf-8') as file:
        text = file.read()
        split = text.split()
        for pos in checklist_pos:
            res_dict[pos] = [i.replace(f':{pos}','') for i in split if i.endswith(f':{pos}')]
    return res_dict


def main(tokenpos_path, n_workers, functor_filename, min_freq):
    checklist_pos = ['ADP','SCONJ','X','AUX','CCONJ','PART','PRON','DET']
    tokenpos_path = Path(tokenpos_path)
    functor_filename = Path(functor_filename)
    filenames = glob.glob(str(Path(tokenpos_path).joinpath(f'*.{"tokenpos"}').absolute()))
    pool_args = [(fn,) for fn in filenames]
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(worker, pool_args)

    result_dict = {k:[] for k in checklist_pos}
    for d in results:
        for k in checklist_pos:
            result_dict[k].extend(d[k])
    for k in checklist_pos:
        result_dict[k] = dict(sorted(Counter(result_dict[k]).items(), key=lambda x:x[1], reverse=True))
    with open(functor_filename, mode='w', encoding='utf-8') as file:
        for k,v in result_dict.items():
            file.write(f'[{k}]\n')
            for kk,vv in v.items():
                if int(vv)>=min_freq:
                    file.write(f'{kk}:{vv}')
                    file.write('\n')
            file.write('\n')
    return

if __name__ == '__main__':
    tokenpos_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_FULL\TOKENPOS'
    n_workers = 4
    functor_filename = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_FULL\FUNCTOR\test_functors.txt'
    main(tokenpos_path, n_workers, functor_filename)
