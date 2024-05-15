'''
[id] 20240516025821
[desc] 绘制一个样本(一个词(或词类)且一个句长)的线性位置分布
[usage]
    1) ifname=main
    2) import
'''

from typing import List
import matplotlib.pyplot as plt
from collections import Counter


def main(sample:List[int], sentlen:int, density=True)->None:
    plt.rcParams["font.sans-serif"]=["SimHei"]
    fig = plt.figure()
    ax = fig.subplots()
    pmf = dict(sorted(Counter(sample).items(),key=lambda i:i[0]))
    complete_pmf = {k:0 for k in range(1,sentlen+1)} 
    complete_pmf.update(pmf)
    if density == True:
        xs = list(complete_pmf.keys())
        ys = list(map(lambda x:x/sum(complete_pmf.values()),complete_pmf.values()))
    else:
        xs = list(pmf.keys())
        ys = list(pmf.values())
    ax.plot(xs, ys, marker='v',markersize=4,fillstyle='none',linewidth=1)
    ax.set_title(f'语言:汉语 词类:PRON 频数:{len(sample)}')
    ax.set_ylabel(f'概率')
    ax.set_xlabel(f'句中线性位置')
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    plt.show()
    return

if __name__ == '__main__':
    sample_filename = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\utils\PRON_20.posposdist'
    with open(sample_filename, mode='r', encoding='utf-8') as file:
        sample = file.readline().strip()
        sample = sample.split()
        sample = [int(i) for i in sample]
    main(sample, 20)