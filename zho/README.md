# 汉语日语位频分布研究

## Usage
1. edit config.py
2. python pipeline.py

## Dependancy
1. cygwin
1. git-bash
1. paste
1. sed
1. split (v8.3+)
1. dos2unix
1. pypinyin
2. pykakasi 

## Comment
1. 输入语料库必须是一行一句型
1. 输入词表必须是一行一词型
1. 在config.py中编辑配置之后才能运行pipeline.py
1. pipeline.py集中调用libs中的脚本
1. libs中的脚本在手动输入参数的情况下能够独立运行
1. utils中的脚本独立运行, 用于语料库去重, 查错等任务
1. pipeline.py中可通过注释的方法选择单独运行某些任务, 前提是之前的任务都执行过
1. 每个libs中的脚本都会产生新的输入, 作为下一个步骤的输入