# 汉语位频分布研究

## 概况
* 链式项目结构, 一个脚本的输出作为另一个脚本的输入
* 逐脚本运行
* 使用多进程分发器(thehow.snips.multiproc.feeder_processor), 在多核心CPU上能极大提高运行效率
* main.py和config.py没有实际作用(并不是星型项目结构)

## 实验步骤
* 确定中文语料库 UNPC / Leipzig
* 最佳为双语平行语料库, 因为分句合理 (UNPC)
* 确定待研究目标词 wordlist.txt
* 部署多进程分发器 feeder_processor.py
* 切分语料库 lib_spliter.py
* 多进程清理中文文本 lib_cleaner_un.py / lib_cleaner_leipzig.py
* 多进程中文文本分词 lib_tokenize.py
* 多进程统计词位频分布 lib_word_posdist.py / lib_char_posdist.py
* 合并多进程位频分布结果 lib_merge_hdf5.py
* 可视化位频分布结果 lib_vis_posdist.py / lib_vis_posdist_extlink.py