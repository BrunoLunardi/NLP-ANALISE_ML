# NLP-ANALISE_ML

Para execução deste classificador é necessário ter instalado no computador o Python 3, Spyder e as bibliotecas Pandas, Numpy, sklearn e nltk.

Para instalar estas bibliotecas no computador:

# Instalação Ferramentas e Bibliotecas

Para instalar o Anaconda (que contém o spyder) acesso o link https://docs.anaconda.com/anaconda/install/

Para instalar cada uma das bibliotecas citadas digite cada um dos códigos abaixo:

## Pandas
pip install pandas

## Numpy
pip install numpy

## Sklearn
pip install -U scikit-learn

## Nltk
pipi install nltk

após instalar a biblioteca NLTk abra um arquivo novo python e digite os seguintes códigos:

import nltk

nltk.download()

Após digitar os dois códigos acima, a janela do "NLTK Downloader" será aberta. Atualize/Instale todos os pacotes selecionado a opção "All" e depois clicando em "Download"

# Git

Para clonar este projeto, basta abrir um terminal de comandos, através dele "navegar" até uma pasta de sua preferência e digitar o seguinte comando:

git clone https://github.com/BrunoLunardi/NLP-ANALISE_ML.git

# Execução do código

Com o spyder aberto e o arquivo NPL.py devidamente importado, basta apertar a tecla "F5" ou clicar em "Run file (F5)". Será exibido no terminal um exemplo de uma música que está sendo classificada. Esta musica está na base de treinamento, o código divide a base de dados em treino e teste.

Caso você deseje alterar a música que está sendo utilizada para classificar, basta alterar o valor da variável "linha_classificar", que está na linha 183. Esta variável é um índice para qual linha o classificador irá predizer de qual artista é a música da base de dados de teste.

O código está devidamente comentado, porém se surgir dúvidas é só enviar uma mensagem que eu auxilío a executar o mesmo
