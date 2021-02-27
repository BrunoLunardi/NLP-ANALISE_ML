# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:21:50 2021

@author: Bruno Guilherme Lunardi
"""

import pandas as pd
import numpy as np
import nltk
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import train_test_split

#nltk.download()

df = pd.read_excel("teste_smarkio_lbs.xls", sheet_name="NLP", header=1)

#dividir a base de dados em letras das musicas e o artista a qual as letras pertencem
musicas = df.iloc[:, 0].values
artistas =  df.iloc[:, 1].values

#dividir a base de dados em teste e treinamento
music_trein, music_teste, art_trein, art_teste = train_test_split(musicas, artistas, test_size=0.30, random_state=0)

#variavel com as stopswords em inglês
stopwordsnltk = nltk.corpus.stopwords.words('english')

################
#INÍCIO FUNÇÕES
################

################## tratando as palavras -> stemmer e remove stopwords ##################
def funcStemmerStopWords(letras_musicas):
    """
      Função que deixa somente radicais das palavras (PorterStemmer) e retira stopwords, ou seja, palavras
      que são irrelevantes para o processamento do algortimo
      return letras_musicas(stemming) -> numpy.narray
    """
    
    #definição do stemmer para deixar radicais das palavras
    stemmer = nltk.PorterStemmer()
    
    for idx, letras in np.ndenumerate(letras_musicas):
        
        #split as letras das musicas pelo "caractere espaço"
        #para cada palavra que não está na lista de stopwords, remove o radical deste e adiciona na lista
        pStemming = [str(stemmer.stem(p)) for p in letras.split() if p not in stopwordsnltk]
        
        #coloca a letra da musica (sem stopwords e com stemmer) no array de musicas
        letras_musicas[idx] = pStemming
        
    return letras_musicas

################## inicio buscar todas as palavras distintas da base de dados ##################
###### isto serve para montar a tabela de probabilidade do Naive Bayes

def funcListaPalavras(letras_musicas):
    """
      Função que retorna uma lista com todas as palavras contidas nas letras das músicas
      return listaPalavras(palavras) -> list
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
      return palavras(palavra, frequencia)  -> dict
    """
    palavras = nltk.FreqDist(palavras)
    return palavras

def funcListaPalavrasUnicas(frequencia):
    """
      Função que retorna um dicionário de todas as palavras distintas
      return freq(keys())  -> dict_keys
    """    
    freq = frequencia.keys()
    return freq

def funcVerificaPalavras(documento):
    """
      Função que recebe uma frase e, de acordo com a base de dados de palavras de treinamento, 
      preenche com valores booleano.
          As palavras da base de dados que existerem na frase receberão valores true, o resto será false
          Isto auxilia no processo de geração da tabela de probabilidade Naive Bayes
      return (caracteristicas)  -> dict_keys
    """       
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavraUnicaTreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

################## fim buscar todas as palavras distintas da base de dados ##################

def funcArrayParaLista(narrayMusicas, narrayArtistas):
    """
        Função que converte numpy.narray das musicas e artistas para uma lista de tuplas com estes dois valores
        retorno listaMusArt(musicas, artistas) -> dict
    """    
    
    listaMusArt = []
    
    listaMusicas = narrayMusicas.tolist()
    listaArtistas = narrayArtistas.tolist()

    for i in range(len(listaMusicas)):
        tuplaAux = (listaMusicas[i], listaArtistas[i])
        listaMusArt.append(tuplaAux)    
    
    return listaMusArt

def funcMatrizConfusao(musicaClass, classificador):
    esperado = []
    classificado = []
    for (letra, artista) in dadosTreinamento:
        resultado = classificador.classify(letra)
        classificado.append(resultado)
        esperado.append(artista)    
        
    matriz = ConfusionMatrix(esperado, classificado)
    print(matriz)

################
#FIM FUNÇÕES
################


################
#INÍCIO TRATAMENTO DADOS DE TREINO
################
#chama função para remover stop words e deixar somente radical da palavra
music_trein = funcStemmerStopWords(music_trein)
#obtem a lista de todas as palavras do conjunto de dados
listaTodasPalavrasTreinamento = funcListaPalavras(music_trein)

#obtém a lista das frequencias das palavras
freqTreinamento = funcFreqPalavras(listaTodasPalavrasTreinamento)
#recebe a lista de todas as palavras distintas da base de dados
palavraUnicaTreinamento = funcListaPalavrasUnicas(freqTreinamento)
#converte array de musicas e artistas para uma lista, unindo estes dois valores
listaMusArtTreinamento = funcArrayParaLista(music_trein, art_trein)

################
#FIM TRATAMENTO DADOS DE TREINO
################

################
#INÍCIO CLASSIFICADOR
################

#aplica os valores da tabela de Naive Bayes
dadosTreinamento = nltk.classify.apply_features(funcVerificaPalavras, listaMusArtTreinamento)
# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(dadosTreinamento)
#exibe as classes da base de dados (artistas)
#print(classificador.labels())

################
#FIM CLASSIFICADOR
################


################
#INÍCIO TESTE CLASSIFICADOR
################
listaMusArtTeste = funcArrayParaLista(music_teste, art_teste)
#coluna
    #se você setar iTuple para o valor 0 (zero), então será exibido a letra da música
    #se você setar iTuple para o valor 1 (um), então será exibido a artista

#define a linha da music_teste que será utilizada para ser classificado 
    #(você pode selecionar qualquer música da base de teste através do índice [linha_classificar])
linha_classificar = 0
musicaParaClassificar = listaMusArtTeste[linha_classificar][0]
#como a base de dados de teste já contém o artista, logo podemos pegar ele como artista esperado que o classificador
    #deve retornar
artistaEsperado = listaMusArtTeste[linha_classificar][1]

#faz a conexão das palavras da letra da música com a lista de palavras únicas da base de dados de treino
    #esta conexão de palavras, na qual se existir a palavra da letra da música na base de dados o valor será True
    #é utilizado na tabela de probabilidades do Naive Bayes
musicaParaClassificar = funcVerificaPalavras(musicaParaClassificar)

#aplica classificador para a letra da música definida selecionada através da linha_classificar
resultClass = classificador.classify(musicaParaClassificar)

print("Era esperado que o classificador retornasse a artista: ", artistaEsperado)
print("O classificador retornou o seguinte resultado para a música escolhida: ", resultClass)

#matriz de confusão
funcMatrizConfusao(listaMusArtTeste, classificador)

################
#FIM TESTE CLASSIFICADOR
################

