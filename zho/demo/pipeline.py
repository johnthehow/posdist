'''
[id] 20240421013920
[desc] 东亚语言分布研究主函数
[usage] 
    1) 直接不带外部参数运行, 不能作为模块导入
    2) 运行所需参数从config.py导入
    3) 脚本中的每个step都可以单独运行, 如果之前step的脚本运行过一遍
[depend]
    1) import依赖: ./lib下的所有脚本
    2) import依赖: config.py
    3) import依赖: pypinyin, pinyin, pykakasi
    4) 操作系统依赖: LINUX/cygwin: paste, dos2unix, sed, sort, uniq
[performance]
    1) Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
    2) 256G RAM
    3) 3.5GB文本
    4) 用时约1小时
[scheme]
    **) 分割
    **) 清理
    **) 合并排序去重再切分
    **) 深度去重
    **) 分词+词性标注
    **) 分词
    **) 统计行数,词数
    **) 统计词频
    **) 统计功能词频
    **) 统计句长分布
    **) 统计位频分布
    **) 可视化位频分布
    **) 罗马字可视化文件名
'''
from config import *
from libs import spliter, cleaner, sortuniq, tokenize_postag, tokenpos2token, lineword_stat, word_freq, functor_freq, word_sentlen_dist, word_posdist, vis_posdist, romanize
from pathlib import Path
import glob
import subprocess
import os
import multiprocessing
import inspect
import time
import warnings
warnings.filterwarnings('ignore') 



class timer(object):
    def __enter__(self):
        self.time_start = time.time()
    def __exit__(self,exc_type, exc_val,exc_tb):
        self.time_end = time.time()
        print(f'duration: {self.time_end - self.time_start}')


if __name__ == '__main__':

    for i in ['SPLIT','CLEAN','EXCEPT','STATS','SORTUNIQ','DESIM','TOKENPOS','TOKEN', 'LINEWORD', 'WORDFREQ', 'FUNCTOR', 'SENTLEN', 'POSDIST','IMG']:
        os.makedirs(CORPUS_ROOT_PATH.joinpath(i), exist_ok=True)

    # step1: 把大语料库拆成小语料库
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: split')
    with timer() as tm:
        step1 = spliter.main(CORPUS_FILENAME, SPLITER_MAIN_LINES_PER_SPLIT, SPLITER_MAIN_INDEX_LENGTH, CORPUS_SPLIT_PATH)


    # step2: 清理语料库
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: clean')
    with timer() as tm:
        filenames = glob.glob(str(Path(CORPUS_SPLIT_PATH).joinpath(f'*.{"split"}').absolute()))
        pool_args = [(fn, CLEANER_MAIN_LANG, CLEANER_MAIN_MIN_SENTLEN, CLEANER_MAIN_MAX_SENLEN,  CLEANER_MAIN_MIN_CHINESE_PCT, 0) for fn in filenames]
        with multiprocessing.Pool(processes=CLEANER_N_WORKER) as pool:
            results = pool.starmap(cleaner.main, pool_args)
        move_cmd = f'move "{str(CORPUS_SPLIT_PATH.joinpath("*.clean").absolute())}" "{CORPUS_CLEAN_PATH}">Nul'
        subprocess.run(move_cmd, shell=True)
        move_cmd = f'move "{str(CORPUS_SPLIT_PATH.joinpath("*.except").absolute())}" "{CORPUS_EXCEPT_PATH}">Nul'
        subprocess.run(move_cmd, shell=True)
        move_cmd = f'move "{str(CORPUS_SPLIT_PATH.joinpath("*.stats").absolute())}" "{CORPUS_STATS_PATH}">Nul'
        subprocess.run(move_cmd, shell=True)

    # step: 排序去重语料库
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: sort & deduplicate')
    with timer() as tm:
        sortuniq.main(CORPUS_CLEAN_PATH, CORPUS_SORTUNIQ_PATH, CORPUS_DESIM_PATH, SORTUNIQ_MAIN_N_WORKER, SORTUNIQ_MAIN_SIM_RATIO)

    # step 3: 分词+词性标注
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: tokenize & postag')
    with timer() as tm:
        filenames = glob.glob(str(Path(CORPUS_DESIM_PATH).joinpath(f'*.{"desim"}').absolute()))
        pool_args = [(fn,TOKEN_POSTAG_MAIN_LANG) for fn in filenames]
        with multiprocessing.Pool(processes=TOKEN_POSTAG_N_WORKER) as pool:
            results = pool.starmap(tokenize_postag.main, pool_args)
        move_cmd = f'move "{str(CORPUS_DESIM_PATH.joinpath("*.tokenpos").absolute())}" "{CORPUS_TOKENPOS_PATH}">Nul'
        subprocess.run(move_cmd, shell=True)

    # step 4:分词+词性标注>分词
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: tokenize & postag > tokenize')
    with timer() as tm:
        tokenpos2token.main(CORPUS_TOKENPOS_PATH, CORPUS_TOKEN_PATH, TOKENPOS2TOKEN_MAIN_N_WORKER)

    # step 5: 统计句长分布
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: sentlen dist')
    with timer() as tm:
      word_sentlen_dist.main(CORPUS_TOKEN_PATH, WORD_SENTLEN_DIST_MAIN_N_WORKER, WORD_SENTLEN_DIST_MAIN_IMG_FILENAME)

    # step 6: 生成词频表
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: word freq')
    with timer() as tm:
        word_freq.main(CORPUS_TOKEN_PATH, WORD_FREQ_MAIN_WORDFREQ_FILENAME)

    # step 7: 生成功能词词频表
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: functor explorer')
    with timer() as tm:
        functor_freq.main(CORPUS_TOKENPOS_PATH,FUNCTOR_FREQ_MAIN_N_WORKER, FUNCTOR_FREQ_MAIN_FUNCTOR_FILENAME, FUNCTOR_FREQ_MAIN_MIN_FREQ)

    # step 8: 统计位频分布
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: posdist')
    with timer() as tm:
        word_posdist.main(CORPUS_TOKEN_PATH, CORPUS_POSDIST_PATH, WORDLIST_FILENAME, WORD_POSDIST_MAIN_N_WORKER, WORD_POSDIST_MAIN_MIN_SENTLEN, WORD_POSDIST_MAIN_MAX_SENTLEN)

    # step 9: 可视化位频分布
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: visualization')
    with timer() as tm:
        vis_posdist.main(VIS_POSDIST_MAIN_POSDIST_FILENAME, WORDLIST_FILENAME, CORPUS_IMG_PATH, VIS_POSDIST_MAIN_MIN_SENTLEN, VIS_POSDIST_MAIN_MAX_SENTLEN,VIS_POSDIST_MAIN_INTERVAL, VIS_POSDIST_MAIN_DENSITY)


    # step 10: 可视化文件名加罗马字
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] major step: romanize')
    with timer() as tm:
        romanize.main(ROMANIZE_MAIN_LANG, CORPUS_IMG_PATH, ROMANIZE_MAIN_IMG_EXTENSION)




