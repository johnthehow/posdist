from pathlib import Path
import os


def main(posdist_filename:str, word:str, sentlen:int)->list:
    ''' [feature] 根据词和句长, 返回posdist数据文件中对应的位频统计原始数据(词在句中的线性位置样本)
    [arg] 1)name: posdist_filename 2)meaning: posdist文件名 3)shape: 4)range: 5)source: 6)eg: 
    [arg] 1)name: 2)meaning: 3)shape: 4)range: 5)source: 6)eg: 
    [return] 1)meaning: 统计原始数据 2)shape:
    [depend] 1)call_dep: 2)arg_dep: 
    [issue] 1): 2): 
    [comment] 1): 2): 
    [id] 20240515115447
    [scheme] 1)index: 
    '''
    with open(posdist_filename, mode='r', encoding='utf-8') as file:
        result = []
        for line in file:
            line = line.strip()
            line_split = line.split()
            if line_split[0][1:-1] == word and int(line_split[1][1:-1]) == sentlen:
                result = line_split[2:]
        result = [int(i) for i in result]
        print(len(result))
    return result


if __name__ == '__main__':
    main(r'E:\同步空间\作业与校务_镜像\博士\研究\POSDIST_BERT\中间阶段成果\东亚语言位频分布\ZH\UNPC\UNPC_FULL\POSDIST\cleaned_merged.posdist', '但', 20)