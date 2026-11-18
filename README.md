# 🌾 Classificação Automatizada de Doenças em Folhas de Arroz com ConvNeXt-Tiny

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Architecture](https://img.shields.io/badge/Model-ConvNeXt--Tiny-00A86B?style=for-the-badge)
![Environment](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leandrolaune/Classifica-o-de-Folhas-de-Milho-com-CNN/blob/main/CONVNEXTTINY_Classifica%C3%A7%C3%A3o_de_doen%C3%A7as_em_planta%C3%A7%C3%B5es_de_arroz.ipynb)

</div>

---

## 📋 Sobre o Projeto

Este repositório contém a implementação de um modelo de Visão Computacional baseado na arquitetura **ConvNeXt-Tiny** utilizando **Transfer Learning** em PyTorch para a identificação e classificação automatizada de fitopatologias em plantações de arroz (*Oryza sativa*).

O diagnóstico precoce e preciso de doenças foliares é essencial para mitigar perdas na produtividade agrícola e otimizar o uso de defensivos. Neste trabalho, aplicamos técnicas modernas de aprendizado profundo, balanceamento de classes e regularização para classificar imagens de folhas de arroz em **6 categorias distintas**, atingindo **99,50% de acurácia no conjunto de validação**.

---

## 🦠 Classes Identificadas

O modelo foi treinado para distinguir 5 condições patológicas e folhas sadias:

| Índice | Classe (Dataset) | Nome Comum / Patologia | Agente / Característica |
| :---: | :--- | :--- | :--- |
| `0` | `brusone` | Brusone do Arroz (*Rice Blast*) | Fungo *Magnaporthe oryzae* |
| `1` | `escaldura` | Escaldura da Folha (*Leaf Scald*) | Fungo *Microdochium oryzae* |
| `2` | `mancha_parda` | Mancha Parda (*Brown Spot*) | Fungo *Bipolaris oryzae* |
| `3` | `queima_bacteriana` | Queima Bacteriana (*Bacterial Leaf Blight*) | Bactéria *Xanthomonas oryzae* |
| `4` | `queima_da_bainha` | Queima da Bainha (*Sheath Blight*) | Fungo *Rhizoctonia solani* |
| `5` | `saudavel` | Folha Saudável (*Healthy*) | Tecido foliar sem infecção aparente |

---

## 📊 Conjunto de Dados (Dataset)

Os dados são obtidos diretamente via `kagglehub` (`felipeguimaraess/doencas-no-arroz`) e encontram-se previamente particionados para garantir a ausência de vazamento de dados (*data leakage*):

* **Treino (`treino`):** 6.550 imagens
* **Validação (`validacao`):** 1.405 imagens
* **Teste (`teste`):** 1.405 imagens
* **Total:** 9.360 imagens

### Pré-processamento e Balanceamento
* **Data Augmentation (apenas no treino):** `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `RandomRotation(15°)` e `ColorJitter` (brilho, contraste e saturação), seguidos de normalização padrão ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
* **Validação e Teste:** `Resize(256)` + `CenterCrop(224)` e normalização, sem distorções aleatórias.
* **Balanceamento de Classes:** Utilização de `WeightedRandomSampler` com pesos inversamente proporcionais à frequência de cada classe no conjunto de treino, assegurando amostragem equilibrada durante as épocas.

---

## 🧠 Arquitetura e Metodologia de Treinamento

O pipeline foi construído sob a estratégia de **Feature Extraction / Fine-Tuning da cabeça de classificação**:

* **Backbone:** `ConvNeXt-Tiny` com pesos pré-treinados (`ConvNeXt_Tiny_Weights.DEFAULT`).
* **Congelamento de Camadas:** Todos os parâmetros convolucionais do extrator de características foram congelados (`requires_grad = False`).
* **Camada Classificadora:** Substituição da camada linear final (`model.classifier[2]`) para saída em 6 classes.
  * *Parâmetros totais:* `27.824.742`
  * *Parâmetros treináveis:* `4.614`
* **Hiperparâmetros:**
  * **Função de Perda:** `CrossEntropyLoss`
  * **Otimizador:** `Adam` (`lr = 0.001`)
  * **Batch Size:** `32`
  * **Épocas Máximas:** `100`
  * **Regularização:** `EarlyStopping` monitorando a perda de validação (`patience = 5`, `min_delta = 0.001`).

---

## 📈 Resultados Obtidos

O treinamento foi executado em GPU NVIDIA T4 (Google Colab) e convergiu de forma estável, sendo interrompido automaticamente pelo **Early Stopping na 32ª época**:

| Métrica | Treino (Época 32) | Validação (Época 32) | Melhor Validação (Época 27) |
| :--- | :---: | :---: | :---: |
| **Loss (Perda)** | `0.0626` | `0.0242` | `0.0220` |
| **Acurácia** | `97.86%` | `99.50%` | `99.50%` |

---

## 🚀 Como Executar o Projeto

### 1. Via Google Colab (Recomendado)
Clique no botão **Open in Colab** no topo deste README para abrir o notebook diretamente no navegador, ativar o ambiente de execução com GPU (`Ambiente de execução > Alterar o tipo de ambiente de execução > T4 GPU`) e executar as células sequencialmente.

### 2. Execução Local

Clone este repositório e instale as dependências necessárias:

```bash
git clone [https://github.com/leandrolaune/Classifica-o-de-Folhas-de-Milho-com-CNN.git](https://github.com/leandrolaune/Classifica-o-de-Folhas-de-Milho-com-CNN.git)
cd Classifica-o-de-Folhas-de-Milho-com-CNN
