# GAN-Number

GAN made in python with Keras ro gererate digits. MNIST dataset used to train generator and discriminator models.

<div align="center">
  <img src="https://github.com/Daniel-Alvarenga/GAN-Number/assets/128755697/10fe7c8d-f0c1-4f3f-b1a4-950739c760c7"/>
</div>

Este projeto, GAN-Number, é uma implementação em Python de uma Rede Adversarial Generativa (GAN) projetada para gerar imagens de dígitos escritos à mão. Utiliza a biblioteca Keras e o conjunto de dados MNIST para treinar um modelo GAN capaz de gerar imagens de dígitos realistas. A estrutura do projeto está organizada em três diretórios principais: `image`, `models` e `src`.

# Estrutura do Projeto

A estrutura do projeto está organizada da seguinte forma:

- **image**: Este diretório é destinado ao armazenamento de imagens geradas. A GAN irá gerar imagens de dígitos e salvá-las neste diretório.

- **models**: Este diretório é usado para armazenar modelos treinados. O modelo gerador da GAN será salvo neste diretório.

- **src**: Este diretório contém o código-fonte da implementação da GAN. O arquivo-chave neste diretório é `train.py`, que treina o modelo da GAN.

# Treino

## Importações

O script `train.py` começa importando as bibliotecas necessárias, incluindo NumPy, Matplotlib e Keras. Também importa o conjunto de dados MNIST e define a arquitetura do modelo GAN.

## Pré-processamento de Dados

O script carrega o conjunto de dados MNIST, pré-processa os dados e os prepara para o treinamento. As imagens dos dígitos são normalizadas e remodeladas para se adequarem ao modelo da GAN.

## Arquitetura da GAN

O modelo GAN consiste em duas partes: o gerador e o discriminador.

### Gerador

- O gerador é definido como um modelo Keras sequencial.
- Ele possui várias camadas densas com ativações LeakyReLU e normalização em lote.
- A camada de saída usa a função de ativação tanh e o modelo é compilado com perda de entropia cruzada binária e otimizador Adam.

### Discriminador

- O discriminador também é definido como um modelo Keras sequencial.
- Ele possui várias camadas densas com ativações LeakyReLU.
- A camada de saída usa a função de ativação sigmoid e o modelo é compilado com perda de entropia cruzada binária e otimizador Adam.

## Treinamento da GAN

O script treina o modelo GAN por um número especificado de épocas e tamanho de lote. Ele utiliza um loop de treinamento para atualizar o gerador e o discriminador de forma iterativa. O gerador gera imagens falsas a partir de ruído aleatório e o discriminador é treinado para distinguir entre imagens reais e falsas.

## Plotagem de Imagens Geradas

O script fornece uma função para gerar e plotar imagens criadas pela GAN. Essas imagens geradas são salvas no diretório `image`.

## Salvando o Modelo

Após o treinamento, o modelo gerador é salvo no diretório `models` para uso futuro.

## Executando o Código

Para treinar a GAN e gerar imagens de dígitos, você pode executar a função `train_gan` com o número desejado de épocas e tamanho de lote. As imagens geradas serão salvas no diretório `image` e o modelo gerador treinado será salvo no diretório `models`.

## Uso

Como se pode observar, na pasta models, já há um modelo treinado, ele foi treinado com as configirações do arquivo src/train.py, e seu treinamento leveou 9 horas devido à falta de uma placa de vídeo dedicada. Você pode usá-lo para herar novas imagens a partir do load_models, disponível na lib keras, ou para treinar a GAN com novos hiperparâmentros, e gerar imagens de dígitos, você pode executar o seguinte comando em seu terminal após ter Python e ter clonado o repositório:

```bash
python src/train.py
````

Isso iniciará o processo de treinamento e salvará as imagens geradas e o modelo gerador treinado para uso futuro.

Observe que você pode precisar ter o Keras, NumPy e Matplotlib instalados para executar o código com sucesso, portanto recomendo utilizar um ambiente virtual para instalar as bibliotecas:

Para instalar as dependências necessárias listadas no arquivo `requirements.txt`, você pode usar esse arquivo que contém as bibliotecas necessárias com suas versões para instalá-las usando o seguinte comando:

```bash
pip install -r requirements.txt
```

Isso garantirá que você tenha as versões corretas das bibliotecas necessárias instaladas em seu ambiente de desenvolvimento antes de executar o código da GAN. Certifique-se de estar no diretório raiz do projeto ao executar o comando.

Divirta-se experimentando com sua GAN para gerar imagens de dígitos escritos à mão!
