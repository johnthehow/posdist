'''
[id] 20240424144306
[desc] 1) 清理UNPC熟料库专用程序
[usage]
    1) standalone (ifname=main)
    2) acceleratable (feeder_processor)

'''

import re
import os
from typing import Union
from pathlib import Path
from bs4 import BeautifulSoup
import inspect
from zhconv import convert
import string
import argparse
import subprocess


STATS = {
'SENT_TOTAL':0,
'SENT_OUTPUT':0,
'SENT_WITH_LINENO':0,
'SENT_WITH_URL':0,
'SENT_WITH_HTML':0,
'SENT_WITH_NONPRINTABLE':0,
'SENT_TOO_SHORT':0,
'SENT_TOO_LONG':0,
'SENT_ZERO_LENGTH':0,
'SENT_NON_NATIVE':0,
'SENT_WITH_FULLWIDTH_DIGIT':0
}

def mod_remove_lineno(line):
    result = re.sub('^.+\t', '', line)
    if result != line:
        STATS['SENT_WITH_LINENO'] += 1
    return result # 应用

def mod_remove_all_space(line):
    return re.sub('\s+', '', line) # 应用

def mod_convert_fullwidth_to_halfwidth(line):
    ''' [feature] 转换全角字符为半角字符
        [arg] 1)name: 2)meaning: 3)shape: 4)range: 5)source: 6)eg: 
        [return] 1)meaning: 2)shape:
        [depend] 1)call_dep: 2)arg_dep: 
        [issue] 1): 2): 
        [comment] 1): 生成自LLAMA3-70B 2): 
        [id] 
        [scheme] 1)index: 
    '''
    result = ''.join(chr(ord(c) - 0xFEE0) if 0xFF00 <= ord(c) <= 0xFF9F else c for c in line)
    return result # 应用

def mod_remove_illegal_punct(line):
    KEEP_PUNCT = ['.','-','%','¥','$','€','£',' ']
    result = re.sub(f'[^\w|{"|".join(KEEP_PUNCT)}]','',line)
    return result # 应用

def mod_remove_leading_punct(line):
    result = re.sub('^[^\w]+', '', line)
    return result # 应用

def mod_remove_ending_period(line):
    result = re.sub('\.$', "", line)
    return result # 应用

def mod_remove_leading_itemno(line):
    result = re.sub('^[a-zA-Z0-9一二三四五六七八九十\.]+\.', '', line)
    result = re.sub('^\([a-zA-Z0-9一二三四五六七八九十\.]+\)', '', result)
    result = re.sub('^[a-zA-Z0-9一二三四五六七八九十\.]+ ', '', result)
    return result # 应用

def mod_remove_repeating_punct(line):
    result = re.sub('([^\\w| |\n])\\1+', '\\1', line)
    return result # 应用

def mod_remove_non_floatpoint_period(line):
    result = re.sub("(?<=\\D)\\.","",line)
    result = re.sub("\\.(?=\\D)", "", result)
    return result # 应用

def mod_remove_user_symbol(line):
    user_remove_list = ['㈠','㈡','㈢','㈣','㈤','㈥','㈦','㈧','㈨','㈩']
    result = re.sub('|'.join(user_remove_list), '', line)
    return result # 应用

def mod_convert_hant_to_hans(line): # 语言特有
    result = convert(line, 'zh-cn')
    return result # 应用


def if_has_url(line):
    if re.findall('((http|https)\\:\\/\\/)?[a-zA-Z0-9\\.\\/\\?\\:@\\-_=#]+\\.([a-zA-Z]){2,6}([a-zA-Z0-9\\.\\&\\/\\?\\:@\\-_=#])*', line) != []:
        STATS['SENT_WITH_URL'] += 1
        return True
    else:
        return False # 应用

def if_has_non_printable(line):
    if not line.isprintable():
        STATS['SENT_WITH_NONPRINTABLE'] += 1
        return True
    else:
        return False # 应用

def if_has_html(line):
    if bool(BeautifulSoup(line, "html.parser").find()):
        STATS['SENT_WITH_HTML'] += 1
        return True
    else:
        return False  # 应用

def if_has_fullwidth_digits(line):
    fullwidth_digits = ['０' ,'１' ,'２' ,'３' ,'４' ,'５' ,'６' ,'７' ,'８' ,'９']
    if any(i in line for i in fullwidth_digits):
        STATS['SENT_WITH_FULLWIDTH_DIGIT'] += 1
        return True
    else:
        return False

def if_short(line, min_sentlen:int):
    if len(line) < min_sentlen:
        STATS['SENT_TOO_SHORT'] += 1
        return True
    else:
        return False # 应用

def if_long(line, max_sentlen:int):
    if len(line) >= max_sentlen:
        STATS['SENT_TOO_LONG'] += 1
        return True
    else:
        return False  # 应用

def if_not_native(line, lang, pct):  # 应用
    if lang == 'zh':
        native_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dff]',line)
    elif lang == 'jp':
        native_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]',line)
    elif lang == 'kr':
        native_chars = re.findall(r'[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\u302E-\u303F]',line)
    else:
        print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] Unsupported Language {lang}')
    if len(line) == 0: return False
    if len(native_chars)/len(line) < pct:
        STATS['SENT_NON_NATIVE'] += 1
        return True
    else:
        return False

def main(corpus_filename:Union[Path, str],
        lang:str,
        min_sentlen:int,
        max_sentlen:int,
        min_native_pct:float,
        verbose:bool=True) -> None:
    corpus_filename = Path(corpus_filename)
    output_filename = corpus_filename.parent.joinpath(corpus_filename.name+'.clean')
    stat_filename = corpus_filename.parent.joinpath(corpus_filename.name+'.stats')
    exception_filename = corpus_filename.parent.joinpath(corpus_filename.name+'.except')
    corpus_io = open(corpus_filename, encoding='utf-8', mode='r')
    output_io = open(output_filename, encoding='utf-8', mode='w')
    stats_io = open(stat_filename, encoding='utf-8', mode='w')
    exception_io = open(exception_filename, encoding='utf-8', mode='w')

    # 统计语料库总行数
    corpus_total_lines = 0
    while True:
        line = corpus_io.readline()
        if not line:
            break
        corpus_total_lines += 1
    corpus_io.seek(0)
    
    if verbose == True:
        print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}] total lines: {corpus_total_lines:,}')
    
    # 逐行对语料库进行清理, 清理完后再检查, 检查通过则输出到清理后的文件
    line_cnt = 0
    while True:
        line_cnt += 1
        STATS['SENT_TOTAL'] += 1
        if verbose == True:
            print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}] processed {line_cnt:,} lines', end='\x1b\r')
        # 清理阶段
        line = corpus_io.readline().strip() # 读取一行, 去除换行符
        line = mod_remove_lineno(line) # 删除句首leipzig式句子编号
        line = mod_remove_leading_itemno(line) # 去除行首句子编号
        line = mod_convert_fullwidth_to_halfwidth(line) # 将全角字符转换为半角字符
        if lang == 'zh' or lang == 'jp':
            line = mod_remove_all_space(line) # 删除所有空格(取消分词)
        line = mod_remove_leading_itemno(line) # 删除行首的逻辑编号
        line = mod_remove_repeating_punct(line) # 删除重复标点
        line = mod_remove_non_floatpoint_period(line) # 删除非浮点数中间的.
        line = mod_remove_illegal_punct(line) # 删除非例外的标点符号
        line = mod_remove_leading_punct(line) # 删除行首标点符号
        line = mod_remove_ending_period(line) # 删除行末句号
        line = mod_remove_user_symbol(line) # 删除用户定义的符号 # 如6923行
        if lang == 'zh':
            line = mod_convert_hant_to_hans(line) # 繁体转简体
        elif lang =='jp':
            pass
        elif lang == 'kr':
            pass
        else:
            print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}@{os.getpid()}] Unsupported Language {lang}')

        # 检查阶段: 如过行中有URL, html, 非打印字符则不输出此行到清理后的文件
        check = 0
        if if_has_non_printable(line): check += 1; exception_io.write(f'{"[NONPRINT]":<10} [{line_cnt:010d}] {line}\n')
        if if_has_html(line): check += 1; exception_io.write(f'{"[HTML]":<10} [{line_cnt:010d}] {line}\n')
        if if_has_url(line): check += 1; exception_io.write(f'{"[URL]":<10} [{line_cnt:010d}] {line}\n')
        if if_short(line, min_sentlen): check += 1; exception_io.write(f'{"[SHORT]":<10} [{line_cnt:010d}] {line}\n')
        if if_long(line, max_sentlen): check += 1; exception_io.write(f'{"[LONG]":<10} [{line_cnt:010d}] {line}\n')
        if if_not_native(line, lang, min_native_pct): check += 1; exception_io.write(f'{"[NONNATIVE]":<10} [{line_cnt:010d}] {line}\n')
        if check == 0:
            output_io.write(f'{line}\n')
            STATS['SENT_OUTPUT'] += 1
        if not line:
            if line_cnt >= corpus_total_lines: # 有些行会被清理成空行, 会导致清理提前退出, 因而加上这一句
                # print(f'[{Path(__file__).name}>{inspect.stack()[0][3]}] Process [{os.getpid()}] Done with file {corpus_filename.stem}!')
                break
            else:
                STATS['SENT_ZERO_LENGTH'] += 1
                exception_io.write(f'{"[ZERO]":<10} [{line_cnt:010d}] {line}\n')
                continue

    
    # 显示并写入统计信息
    if verbose == True:
        for k,v in STATS.items():
            print(k,v)
    stats_io.write(f'MIN_SENTLEN: {min_sentlen}\n')
    stats_io.write(f'MAX_SENTLEN: {max_sentlen}\n')
    stats_io.write(f'MIN_NATIVE_PCT: {min_native_pct}\n')
    for k,v in STATS.items():
        stats_io.write(f'{k}: {v}\n')
    
    # 收尾: 关闭文件
    corpus_io.close()
    output_io.close()
    stats_io.close()
    exception_io.close()
    return

if __name__ == '__main__':
    main(r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\zho\io\KR\SPLIT\KOR_LEIPZIG_0001.split', 'kr', 10, 100, 0.8)