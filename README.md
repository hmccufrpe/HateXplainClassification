# HateXplainClassification

Projeto desenvolvido para a disciplina de **Mineração de Texto**, com foco na aplicação de técnicas de Processamento de Linguagem Natural (NLP) para classificação de discursos de ódio em textos.

## Objetivo

Explorar e comparar modelos de aprendizado de máquina baseados em Transformers para identificar e classificar conteúdos de discurso de ódio, utilizando o dataset HateXplain.

## Tecnologias utilizadas

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Modelos utilizados

O projeto explora três arquiteturas baseadas em Transformers:

- **BERT:** modelo pré-treinado para compreensão contextual de textos.
- **DistilBERT:** versão otimizada e mais leve do BERT.
- **RoBERTa:** arquitetura baseada no BERT com melhorias no processo de pré-treinamento.

## Estrutura do projeto

```text
HateXplainClassification/
├── bert/           # Experimentos com BERT
├── data/           # Conjunto de dados
├── distilbert/     # Experimentos com DistilBERT
├── nb/             # Notebooks de análise
├── roberta/        # Experimentos com RoBERTa
├── util/           # Funções auxiliares
└── requirements.txt
```

## Instalação

Clone o repositório:

```bash
git clone https://github.com/hmccufrpe/HateXplainClassification.git

cd HateXplainClassification
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Dataset

O projeto utiliza o **HateXplain**, um conjunto de dados desenvolvido para pesquisas relacionadas à identificação e análise de discurso de ódio em textos.

O dataset contempla categorias como:
- Hate Speech (discurso de ódio);
- Offensive (conteúdo ofensivo);
- Normal (conteúdo neutro).

## Colaboradores

Projeto desenvolvido colaborativamente pelos integrantes do repositório:

- Cauã Bittencourt
- Higor Cordeiro
- João Bispo
- Vinícius Souza

## Referência

Mathew, B., Saha, P., Yimam, S. M., et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.*

[Artigo original](https://ojs.aaai.org/index.php/AAAI/article/view/17745)
