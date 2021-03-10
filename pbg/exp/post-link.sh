# python -m spacy download pt_core_news_sm
python -m spacy download pt_core_news_lg
echo  "import nltk; nltk.download('stopwords')" >> tmp.py
python tmp.py
rm tmp.py
