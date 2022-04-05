# Linear-position-distribution Analysis
This repository contains code for 'Emergent linguistic regularities learned by neural network language models via linear position distribution'.
It includes code for 功能1, 功能2(对应文中位置), ...

## Requirements
For pre-processing Leipzig news corpora:
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

Statistics and visualizations of Linear-position-distribution:
* [Numpy](http://www.numpy.org/)
* [SciPy](https://scipy.org/)
* [MatplotLib](https://matplotlib.org/)

Extracting attention weights for words in sentences:
* [PyTorch](https://pytorch.org/)
* [Huggingface Transformers](https://huggingface.co/)



## Pre-processing Leipzig news corpora
We provide a script `corpus_prep.py` for Pre-processing Leipzig news corpora. Leipzig used in the paper can be downloaded from <https://wortschatz.uni-leipzig.de/en/download/>. 

Corpus pre-preprocessing is run with:
`python corpus_prep.py <language> <path-to-your-data> <output-path>`

## Statistics and visualizations of linear-position-distribution

## Extracting attention weights for words in sentences

