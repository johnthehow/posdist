'''
[id] 20240421020115
[feature] 中文位频分布研究配置文件
'''
from pathlib import Path

# project
CORPUS_LANG = 'zh'
N_WORKER = 60

# CORPUS_FILENAME = Path(r"C:\Users\Administrator\Desktop\BaiduSyncdisk\swap\TASKS\POSDIST\zho\io\JP\JPC\JPC_UNIQ.raw")
CORPUS_FILENAME = Path(r"C:\Users\Administrator\Desktop\BaiduSyncdisk\swap\TASKS\POSDIST\zho\io\ZH\UNPC\UNPC.raw")
# CORPUS_FILENAME = Path(r"C:\Users\Administrator\Desktop\BaiduSyncdisk\swap\TASKS\POSDIST\zho\io\KR\KOR_LEIPZIG.raw")
# WORDLIST_FILENAME = Path(r"C:\Users\Administrator\Desktop\BaiduSyncdisk\swap\TASKS\POSDIST\zho\io\JP\WORDLIST\jp_100.wordlist")
WORDLIST_FILENAME = Path(r"C:\Users\Administrator\Desktop\BaiduSyncdisk\swap\TASKS\POSDIST\zho\io\ZH\WORDLIST\zh.wordlist")
# WORDLIST_FILENAME = Path(r"C:\Users\Administrator\Desktop\BaiduSyncdisk\swap\TASKS\POSDIST\zho\io\KR\WORDLIST\kr.wordlist")
CORPUS_ROOT_PATH = CORPUS_FILENAME.parent
CORPUS_SPLIT_PATH = CORPUS_ROOT_PATH.joinpath('SPLIT')
CORPUS_CLEAN_PATH = CORPUS_ROOT_PATH.joinpath('CLEAN')
CORPUS_EXCEPT_PATH = CORPUS_ROOT_PATH.joinpath('EXCEPT')
CORPUS_STATS_PATH = CORPUS_ROOT_PATH.joinpath('STATS')
CORPUS_TOKEN_PATH = CORPUS_ROOT_PATH.joinpath('TOKEN')
CORPUS_TOKENPOS_PATH = CORPUS_ROOT_PATH.joinpath('TOKENPOS')
CORPUS_POSDIST_PATH = CORPUS_ROOT_PATH.joinpath('POSDIST')
CORPUS_IMG_PATH = CORPUS_ROOT_PATH.joinpath('IMG')
CORPUS_SENTLEN_PATH = CORPUS_ROOT_PATH.joinpath('SENTLEN')
CORPUS_WORDFREQ_PATH = CORPUS_ROOT_PATH.joinpath('WORDFREQ')
CORPUS_FUNCTOR_PATH = CORPUS_ROOT_PATH.joinpath('FUNCTOR')
CORPUS_SORTUNIQ_PATH = CORPUS_ROOT_PATH.joinpath('SORTUNIQ')
CORPUS_DESIM_PATH = CORPUS_ROOT_PATH.joinpath('DESIM')
CORPUS_LINEWORD_PATH = CORPUS_ROOT_PATH.joinpath('LINEWORD')


# SPLITTER.py
SPLITER_MAIN_LINES_PER_SPLIT = 10000
SPLITER_MAIN_INDEX_LENGTH = 4

# CLEANER.PY
CLEANER_MAIN_LANG = CORPUS_LANG
CLEANER_MAIN_MIN_SENTLEN = 10
CLEANER_MAIN_MAX_SENLEN = 100
CLEANER_MAIN_MIN_CHINESE_PCT = 0.8
CLEANER_N_WORKER = N_WORKER

# SORTUNIQ.PY
SORTUNIQ_MAIN_N_WORKER = N_WORKER
SORTUNIQ_MAIN_SIM_RATIO = 0.9

# TOKEN_POSTAG.PY
TOKEN_POSTAG_N_WORKER = N_WORKER
TOKEN_POSTAG_MAIN_LANG = CORPUS_LANG

# TOKENPOS2TOKEN.PY
TOKENPOS2TOKEN_MAIN_N_WORKER = N_WORKER

# WORD_POSDIST.PY
WORD_POSDIST_MAIN_N_WORKER = N_WORKER
WORD_POSDIST_MAIN_MIN_SENTLEN = 10
WORD_POSDIST_MAIN_MAX_SENTLEN = 100


# ROMANIZE.PY
ROMANIZE_MAIN_LANG = CORPUS_LANG
ROMANIZE_MAIN_IMG_EXTENSION = 'png'

# WORD_SENTLEN_DIST.PY
WORD_SENTLEN_DIST_MAIN_N_WORKER = N_WORKER
WORD_SENTLEN_DIST_MAIN_IMG_FILENAME = CORPUS_SENTLEN_PATH.joinpath('sentlen.png')


# WORD_FREQ.PY
WORD_FREQ_MAIN_TOKEN_PATH = CORPUS_TOKEN_PATH
WORD_FREQ_MAIN_WORDFREQ_FILENAME = CORPUS_WORDFREQ_PATH.joinpath('wordfreq.txt')


# FUNCTOR_FREQ.PY
FUNCTOR_FREQ_MAIN_N_WORKER = N_WORKER
FUNCTOR_FREQ_MAIN_FUNCTOR_FILENAME = CORPUS_FUNCTOR_PATH.joinpath('functor.txt')
FUNCTOR_FREQ_MAIN_MIN_FREQ = 1000

# VIS_POSDIST.PY
VIS_POSDIST_MAIN_POSDIST_FILENAME = CORPUS_POSDIST_PATH.joinpath(f'cleaned_merged.posdist')
VIS_POSDIST_MAIN_MIN_SENTLEN = 14
VIS_POSDIST_MAIN_MAX_SENTLEN = 24
VIS_POSDIST_MAIN_INTERVAL = 2
VIS_POSDIST_MAIN_DENSITY = True