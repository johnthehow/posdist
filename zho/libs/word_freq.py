import subprocess
from pathlib import Path
import os
from collections import Counter

def main(token_path, result_filename):
    token_path = Path(token_path)
    cmd = f'copy "{token_path.joinpath("*.token")}" "{token_path.joinpath("merged.token")}">Nul'
    subprocess.run(cmd, shell=True)
    with open(token_path.joinpath("merged.token"), mode='r',encoding='utf-8') as file:
        text = file.read().strip()[:-1]
        split = text.split()
    counter = Counter(split)
    counter = dict(sorted(counter.items(), key=lambda i:i[1], reverse=True))
    with open(result_filename, mode='w',encoding='utf-8') as file:
        for k,v in counter.items():
            file.write(f'{k}:{v}\n')
    os.remove(token_path.joinpath("merged.token"))
    return