@echo off

REM Install scikit-learn
!pip install -U scikit-learn

REM Install nltk and download stopwords
!pip install -U nltk
!python -m nltk.downloader stopwords

REM Install numpy
!pip install -U numpy

REM Install pandas
!pip install -U pandas

REM Install seaborn
!pip install seaborn

REM Install flask
!pip install flask

REM Install scikit-learn-intelex (for Intel processors)
!pip install scikit-learn-intelex
