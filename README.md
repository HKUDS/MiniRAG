# MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)


The Code Repository: **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**
<br />

[Tianyu Fan](https://tianyufan0504.github.io/), [Jingyuan Wang](), [Xubin Ren](https://ren-xubin.github.io/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />
</div>

<a href='https://arxiv.org/abs/2501.06713'><img src='https://img.shields.io/badge/arXiv-2501.06713-b31b1b'>


## 🌍 README Translations

[中文说明](./README_CN.md) | [日本語](./README_JA.md)

## TLDR
MiniRAG is an extremely simple retrieval-augmented generation framework that enables small models to achieve good RAG performance through heterogeneous graph indexing and lightweight topology-enhanced retrieval.

## Abstract
The growing demand for efficient and lightweight Retrieval-Augmented Generation (RAG) systems has highlighted significant challenges when deploying Small Language Models (SLMs) in existing RAG frameworks. Current approaches face severe performance degradation due to SLMs' limited semantic understanding and text processing capabilities, creating barriers for widespread adoption in resource-constrained scenarios. To address these fundamental limitations, we present **MiniRAG**, a novel RAG system designed for extreme simplicity and efficiency. **MiniRAG** introduces two key technical innovations: (1) a semantic-aware heterogeneous graph indexing mechanism that combines text chunks and named entities in a unified structure, reducing reliance on complex semantic understanding, and (2) a lightweight topology-enhanced retrieval approach that leverages graph structures for efficient knowledge discovery without requiring advanced language capabilities. Our extensive experiments demonstrate that **MiniRAG** achieves comparable performance to LLM-based methods even when using SLMs while requiring only 25\% of the storage space. Additionally, we contribute a comprehensive benchmark dataset LiHua-World for evaluating lightweight RAG systems under realistic on-device scenarios with complex queries.

## MiniRAG Framework

![MiniRAG](https://files.mdnice.com/user/87760/02baba85-fa69-4223-ac22-914fef7120ae.jpg)

MiniRAG employs a streamlined workflow built on the key components: heterogeneous graph indexing and lightweight graph-based knowledge retrieval. This architecture addresses the unique challenges faced by on-device RAG systems, optimizing for both efficiency and effectiveness.


## Install

* Install from source (Recommend)

```bash
cd MiniRAG
pip install -e .
```
* Install from PyPI (Our code is based on [LightRAG](https://github.com/HKUDS/LightRAG), so you can install it directly)

```bash
pip install lightrag-hku
```

## Quick Start
* All the code can be found in the `./reproduce`.
* Download the dataset you need.
* Put the dataset in the `./dataset` directory.
* Note: We have already put the LiHua-World dataset in `./dataset/LiHua-World/data/` as `LiHuaWorld.zip`. If you want to use other dataset, you can put it in the `./dataset/xxx`.


Then use the following bash command to index the dataset:
```bash
python ./reproduce/Step_0_index.py
python ./reproduce/Step_1_QA.py
```

Or, use the code in `./main.py` to initialize MiniRAG.


### Overall Performance Table
| Model | NaiveRAG | | GraphRAG | | LightRAG | | **MiniRAG** | |
|-------|----------|----------|-----------|----------|-----------|----------|----------|----------|
| | acc↑ | err↓ | acc↑ | err↓ | acc↑ | err↓ | acc↑ | err↓ |
| LiHua-World | | | | | | | | |
| Phi-3.5-mini-instruct | 41.22% | 23.20% | / | / | 39.81% | 25.39% | **53.29%** | 23.35% |
| GLM-Edge-1.5B-Chat | 42.79% | 24.76% | / | / | 35.74% | 25.86% | **52.51%** | 25.71% |
| Qwen2.5-3B-Instruct | 43.73% | 24.14% | / | / | 39.18% | 28.68% | **48.75%** | 26.02% |
| MiniCPM3-4B | 43.42% | 17.08% | / | / | 35.42% | 21.94% | **51.25%** | 21.79% |
| gpt-4o-mini | 46.55% | 19.12% | 35.27% | 37.77% | **56.90%** | 20.85% | 54.08% | 19.44% |
| MultiHop-RAG | | | | | | | | |
| Phi-3.5-mini-instruct | 42.72% | 31.34% | / | / | 27.03% | 11.78% | **49.96%** | 28.44% |
| GLM-Edge-1.5B-Chat | 44.44% | 24.26% | / | / | / | / | **51.41%** | 23.44% |
| Qwen2.5-3B-Instruct | 39.48% | 31.69% | / | / | 21.91% | 13.73% | **48.55%** | 33.10% |
| MiniCPM3-4B | 39.24% | 31.42% | / | / | 19.48% | 10.41% | **47.77%** | 26.88% |
| gpt-4o-mini | 53.60% | 27.19% | 60.92% | 16.86% | 64.91% | 19.37% | **68.43%** | 19.41% |


In the table, / means the method struggles to generate effective responses.

## Reproduce
All the code can be found in the `./reproduce` directory.

## Code Structure

```python
├── dataset
│   └── LiHua-World
│       ├── README.md
│       ├── README_CN.md
│       ├── data
│       │   ├── LiHuaWorld.zip
│       └── qa
│           ├── query_set.csv
│           └── query_set.json
├── minirag
│   ├── kg
│   │   ├── __init__.py
│   │   ├── neo4j_impl.py
│   │   └── oracle_impl.py
│   ├── __init__.py
│   ├── base.py
│   ├── llm.py
│   ├── minirag.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   └── utils.py
├── reproduce
│   ├── Step_0_index.py
│   └── Step_1_QA.py
├── LICENSE
├── main.py
├── README.md
├── README_CN.md
├── requirements.txt
├── setup.py
```

## Dataset: LiHua-World

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

LiHua-World is a dataset specifically designed for on-device RAG scenarios, containing one year of chat records from a virtual user named LiHua. The dataset includes three types of questions: single-hop, multi-hop, and summary, with each question paired with manually annotated answers and supporting documents. For more details, please refer to the [README of LiHua-World](./dataset/LiHua-World/README.md) dataset.

## Star History

<a href="https://star-history.com/#HKUDS/MiniRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
 </picture>
</a>

## Contribution

Thank you to all our contributors!

<a href="https://github.com/HKUDS/MiniRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/MiniRAG" />
</a>


## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository,
[nano-graphrag](https://github.com/gusye1234/nano-graphrag) and [LightRAG](https://github.com/HKUDS/LightRAG). Thanks for their wonderful works.

## 🌟Citation

```python
@article{fan2025minirag,
  title={MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author={Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2501.06713},
  year={2025}
}
```

**Thank you for your interest in our work!**
