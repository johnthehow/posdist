'''
[id] 20240430170032
[desc]
    1) 将汉语和日语的文件名前加上罗马化前缀
[depend]
    1) pinyin
    2) pykakasi
[usage]
    1) ifname=main
'''

from typing import Union, Any, Sequence, Mapping
from pathlib import Path
import subprocess
import os
import glob


def add_prefix(filenames:Sequence[Path], prefixes:Sequence[str]):
    for fn,pf in zip(filenames,prefixes):
        new = f'{fn.parent.joinpath(f"{pf}{fn.name}")}'
        os.rename(fn, new)
    return

def add_pinyin_prefix(filenames):
    import pinyin
    bare_names = [str(i.stem) for i in filenames]
    prefixes = [pinyin.pinyin.get(i,format='strip')+'_' for i in bare_names]
    add_prefix(filenames, prefixes)
    return

def add_romaji_prefix(filenames):
    import pykakasi
    kks = pykakasi.kakasi()
    bare_names = [str(i.stem) for i in filenames]
    prefixes = []
    for fname in bare_names:
        converted_list = kks.convert(fname)
        converted_string = ''.join([i['passport'] for i in converted_list])+'_'
        prefixes.append(converted_string)
    add_prefix(filenames, prefixes)
    return prefixes

def main(lang, file_path, img_extension):
    filenames = glob.glob(str(Path(file_path).joinpath(f'*.{img_extension}').absolute()))
    filenames = [Path(i) for i in filenames]
    if lang == 'zh':
        add_pinyin_prefix(filenames)
    elif lang == 'jp':
        add_romaji_prefix(filenames)
    elif lang == 'kr':
        pass
    return
