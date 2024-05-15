'''
[id] 20240509005129
[desc]
    1) 从分过词的语料库中, 寻找某A词出现在句长B的句子中C位置的句子
[usage]
    1) ifname=main
'''


from pathlib import Path
import glob
import multiprocessing
import os
import subprocess


def worker(token_filename, word, pos, sentlen):
    token_filename = Path(token_filename)
    output_filename = token_filename.with_suffix(token_filename.suffix + '.select')
    with open(token_filename, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines_split = [line.split() for line in lines]
    with open(output_filename, mode='w', encoding='utf-8') as output:
        line_cnt = 0 
        for sent in lines_split:
            line_cnt += 1
            if len(sent) == sentlen:
                if sent[pos-1] == word:
                    output.write(f'{" ".join(sent)}[{token_filename}] [{line_cnt}]\n')
    return


def main(corpus_token_path, report_savepath, n_worker, word, pos, sentlen):
    token_filenames = glob.glob(str(Path(corpus_token_path).joinpath(f'*.{"token"}').absolute()))
    pool_args = [(fn, word, pos, sentlen) for fn in token_filenames]
    with multiprocessing.Pool(n_worker) as pool:
        results = pool.starmap(worker, pool_args)
    os.chdir(corpus_token_path)
    report_filename = f'report_word_{word}_sentlen_{sentlen}_pos_{pos}.select'
    cmd = f'copy /b *.select {report_filename}'
    subprocess.run(cmd, shell=True, check=True)
    cmd = f'move {report_filename} {report_savepath} '
    subprocess.run(cmd, shell=True, check=True)
    select_filenames = glob.glob(str(Path(corpus_token_path).joinpath(f'*.{"select"}').absolute()))
    for i in select_filenames:
        os.remove(i)
    return


if __name__ == '__main__':
    corpus_token_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\JP\JPC\JPC_MINI\TOKEN'
    report_savepath = os.getcwd()
    n_worker = 4
    word = 'の'
    pos = 11
    sentlen = 22
    main(corpus_token_path, report_savepath, n_worker, word, pos, sentlen)