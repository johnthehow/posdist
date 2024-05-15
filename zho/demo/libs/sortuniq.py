import os
import subprocess
import re
from pathlib import Path
import inspect
import multiprocessing
import difflib
import glob
import time

class timer(object):
    def __enter__(self):
        self.time_start = time.time()
    def __exit__(self,exc_type, exc_val,exc_tb):
        self.time_end = time.time()
        print(f'duration: {self.time_end - self.time_start}')

def dos2unix(filename):
    filename = Path(filename)
    cmd = f'dos2unix "{filename}">nul 2>&1'
    subprocess.run(cmd, shell=True)
    return


def worker(input_filename, sim_ratio:float):
    input_filename = Path(input_filename)
    output_filename = input_filename.with_suffix(input_filename.suffix + '.desim')
    with open(input_filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    unique_lines = [lines[0]]
    for line in lines[1:]:
        if difflib.SequenceMatcher(None, line, unique_lines[-1]).ratio() < sim_ratio:
            unique_lines.append(line)

    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    return

def main(clean_path, sortuniq_path, desim_path, n_worker, sim_ratio):
    os.chdir(clean_path)
    filenames = os.listdir()
    clean_filenames = [fn for fn in filenames if fn.endswith('clean')]

    # 为文件名和切分作参数准备
    filename_stem = re.findall('.+?(?=\\d+\\.)',clean_filenames[0])[0][:-1] # JPC_100K
    wc_return = subprocess.run(f'wc -l {clean_filenames[0]}', shell=True, capture_output=True).stdout.decode('utf-8').split()[0] # 返回第一个clean文件的行数
    lines_per_split = 10**(len(wc_return))
    index_length = len(re.search('\\d+\\.',clean_filenames[0])[0][:-1])

    # 合并所有clean文件为一个名为.merge的文件
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] merging files')
    merged_clean_filename = filename_stem + '.clean'
    cmd = f'copy /b *.clean {merged_clean_filename}>nul'
    subprocess.run(cmd, shell=True)

    # 删除合并后的文件中的所有空行
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] removing empty lines')
    cmd = f'sed -i "/^$/d" {merged_clean_filename}'
    subprocess.run(cmd, shell=True)

    # 排序合并后的文件
    with timer() as tm:
        print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] sorting cleaned corpus')
        sorted_merged_clean_filename = merged_clean_filename + '.sort'
        cmd = f'"C:\\Program Files\\Git\\usr\\bin\\sort.exe" --parallel={n_worker} {merged_clean_filename}>{sorted_merged_clean_filename}'
        subprocess.run(cmd, shell=True)
    
    # 去重
    with timer() as tm:
        print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] deduplicating cleaned and sorted corpus')
        uniq_sorted_merged_clean_filename = sorted_merged_clean_filename + '.uniq'
        cmd = f'uniq {sorted_merged_clean_filename}>{uniq_sorted_merged_clean_filename}'
        subprocess.run(cmd, shell=True)

    # 切分文件
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] spliting cleaned and sorted corpus for DOS2UNIX')
    cmd = f'"C:\\Program Files\\Git\\usr\\bin\\split.exe" -l {lines_per_split} --numeric-suffix=1 -a {index_length} {uniq_sorted_merged_clean_filename} {filename_stem}_ --additional-suffix=.clean.sort.uniq.resplit'
    subprocess.run(cmd, shell=True)

    # # 转换所有CRLF为LF
    # with timer() as tm:
    #     print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] converting cleaned corpus from CRLF to LF')
    #     filenames = glob.glob(str(Path(clean_path).joinpath(f'*.{"resplit"}').absolute()))
    #     pool_args = [(fn,) for fn in filenames]
    #     with multiprocessing.Pool(n_worker) as pool:
    #         pool.starmap(dos2unix, pool_args)

    # 删除中间阶段文件
    os.remove(merged_clean_filename)
    os.remove(sorted_merged_clean_filename)
    os.remove(uniq_sorted_merged_clean_filename)

    # 移动后缀为.resplit的文件到SORTUNIQ目录
    sortuniq_path = Path(sortuniq_path)
    cmd = f'move *.resplit "{sortuniq_path}">nul'
    subprocess.run(cmd, shell=True)

    # 进一步去重
    os.chdir(sortuniq_path)
    print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] decuplicating by similarity')
    filenames = glob.glob(str(Path(sortuniq_path).joinpath(f'*.{"resplit"}').absolute()))
    pool_args = [(fn, sim_ratio) for fn in filenames]
    with multiprocessing.Pool(n_worker) as pool:
        pool.starmap(worker, pool_args)
    cmd = f'move *.desim "{desim_path}">nul'
    subprocess.run(cmd, shell=True)
    return


if __name__ == '__main__':
    clean_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_MINI\CLEAN'
    sortuniq_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_MINI\SORTUNIQ'
    desim_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_MINI\DESIM'
    sim_ratio = 0.9
    n_worker = 4
    main(clean_path, sortuniq_path, desim_path, n_worker, sim_ratio)
