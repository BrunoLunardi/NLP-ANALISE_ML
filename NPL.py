# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:21:50 2021

@author: bruno
"""

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

df = pd.read_excel("teste_smarkio_lbs.xls", sheet_name="NLP")

#dividir a base de dados em letras das musicas e o artista a qual as letras pertencem
musicas = df.iloc[:, 0].values
artistas =  df.iloc[:, 1].values
# print(artistas)
# print(musicas)

#dividir a base de dados em teste e treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(musicas, artistas, test_size=0.25, random_state=0)

#remove as stopswords 
stopwordsnltk = nltk.corpus.stopwords.words('english')
# print(stopwordsnltk)

