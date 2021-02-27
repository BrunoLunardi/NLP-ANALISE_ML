# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:21:50 2021

@author: bruno
"""

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

#nltk.download()

df = pd.read_excel("teste_smarkio_lbs.xls", sheet_name="NLP", header=1)

#print(df.head())

#dividir a base de dados em letras das musicas e o artista a qual as letras pertencem
musicas = df.iloc[:, 0].values
artistas =  df.iloc[:, 1].values
# print(artistas)
# print(musicas)

#dividir a base de dados em teste e treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(musicas, artistas, test_size=0.25, random_state=0)

#remove as stopswords 
stopwordsnltk = nltk.corpus.stopwords.words('english')

################## início radical das palavras -> stemming e remove stopwords ##################
def aplicastemmer(musica):
    """
      Função que deixa somente radicais das palavras (PorterStemmer) e retira stopwords, ou seja, palavras
      que são irrelevantes para o processamento do algortimo
    """
    
    #definição do stemmer para deixar radicais das palavras
    stemmer = nltk.PorterStemmer()
    frasesstemming = []
    for palavras in musicas:
        #para cada palavra que não está na lista de stopwords, remove o radical deste e adiciona na lista
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasesstemming.append((comstemming))
        
    return frasesstemming

teste5 = []
teste5 = aplicastemmer(previsores_treinamento)
print(teste5)
################## fim radical das palavras -> stemming ##################

################## inicio buscar todas as palavras distintas da base de dados ##################

################## fim buscar todas as palavras distintas da base de dados ##################
