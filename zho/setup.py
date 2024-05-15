from setuptools import setup

setup(
    name = 'posdistasia',
    version = '0.1',
    description = 'posdist study program for asian languages',
    author = 'John Thehow',
    author_email = 'johnthehow@qq.com',
    packages = ['src'],
    install_requires = ['pypinyin','pinyin','pykakasi','bs4']
    )