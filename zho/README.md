# 汉语日语位频分布研究

## Usage
1. edit config.py
2. python pipeline.py

## Dependancy
1. cygwin
1. git-bash
1. paste
1. sed
1. sort
1. uniq
1. split (v8.3+)
1. dos2unix
1. pypinyin
1. pykakasi 
1. spacy
1. spacy.zh_core_web_sm
1. spacy.ja_core_news_sm


## Comment
1. 实现从单体未处理语料库到实验结果的一键式服务
1. 支持多进程并行
1. 输入语料库必须是一行一句型
1. 输入词表必须是一行一词型
1. 在config.py中编辑配置之后才能运行pipeline.py
1. 在config.py中指定语料库(未拆分的一行一句型语料库单体文件)
1. 在config.py中指定输入词表
1. 在config.py中指定多进程数
1. pipeline.py集中调用libs中的脚本
1. libs中的脚本在手动输入参数的情况下能够独立运行
1. utils中的脚本独立运行, 用于语料库去重, 查错等任务
1. pipeline.py中可通过注释的方法选择单独运行某些任务, 前提是之前的任务都执行过
1. 每个libs中的脚本都会产生新的输入, 作为下一个步骤的输入