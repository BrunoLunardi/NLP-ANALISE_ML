# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:21:50 2021

@author: bruno
"""

import pandas as pd
import numpy as np
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

#variavel com as stopswords em inglês
stopwordsnltk = nltk.corpus.stopwords.words('english')

################## tratando as palavras -> stemmer e remove stopwords ##################
def funcStemmerStopWords(letras_musicas):
    """
      Função que deixa somente radicais das palavras (PorterStemmer) e retira stopwords, ou seja, palavras
      que são irrelevantes para o processamento do algortimo
    """
    
    #definição do stemmer para deixar radicais das palavras
    stemmer = nltk.PorterStemmer()
    for idx, letras in np.ndenumerate(letras_musicas):
        #split as letras das musicas pelo "caractere espaço"
        #para cada palavra que não está na lista de stopwords, remove o radical deste e adiciona na lista
        comstemming = [str(stemmer.stem(p)) for p in letras.split() if p not in stopwordsnltk]
        #coloca a letra da musica (sem stopwords e com stemmer) no array de musicas
        letras_musicas[idx] = comstemming
        
    return letras_musicas

################## inicio buscar todas as palavras distintas da base de dados ##################
###### isto serve para montar a tabela de probabilidade do Naive Bayes

def funcListaPalavras(letras_musicas):
    """
      Função que retorna uma lista com todas as palavras contidas nas letras das músicas
    """
    #define a lista que receberá todas as palavras contidas nas letras das músicas
    listaPalavras = []
    #for para cada letra de música
    for idx, letras in np.ndenumerate(letras_musicas):
        #for para pegar cada palavra contida em cada letra de música
        for palavra in letras:
            listaPalavras.append(palavra)

    #retorna uma lista com todas as palavras das letras de música
    return listaPalavras

def funcFreqPalavras(palavras):
    """
      Função para retornar uma lista com as palavrsa e quantas vezes elas foram utilizadas
      return (palavra, frequencia) 
    """
    palavras = nltk.FreqDist(palavras)
    return palavras

def funcListaPalavrasUnicas(frequencia):
    """
      Função que retorna um dicionário de todas as palavras distintas
    """    
    freq = frequencia.keys()
    return freq



#chama função para remover stop words e deixar somente radical da palavra
previsores_treinamento = funcStemmerStopWords(previsores_treinamento)

#obtem a lista de todas as palavras do conjunto de dados
listaTodasPalavrasTreinamento = funcListaPalavras(previsores_treinamento)
#obtém a lista das frequencias das palavras
freqTreinamento = funcFreqPalavras(listaTodasPalavrasTreinamento)

palavraUnicaTreinamento = funcListaPalavrasUnicas(freqTreinamento)


