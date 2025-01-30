# MiniRAG: 迈向极简检索增强生成

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)


本仓库是论文: **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation** 的代码仓库。

<br />

[Tianyu Fan](https://tianyufan0504.github.io/), [Jingyuan Wang](), [Xubin Ren](https://ren-xubin.github.io/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />
</div>


<a href='https://arxiv.org/abs/2501.06713'><img src='https://img.shields.io/badge/arXiv-2501.06713-b31b1b'>


## TLDR
MiniRAG 是一个极简的检索增强生成框架，它通过异质图索引和轻量级的拓扑增强检索，让小模型也能取得很好的RAG效果。

## Abstract
对高效且轻量级的检索增强生成（RAG）系统日益增长的需求，凸显了在现有RAG框架中部署小型语言模型（SLMs）时所面临的重大挑战。由于SLMs在语义理解和文本处理能力上的局限性，当前方法面临严重的性能下降问题，这在资源受限的场景中阻碍了其广泛应用。为了应对这些根本性限制，我们提出了**MiniRAG**，这是一种专为极简和高效而设计的新型RAG系统。MiniRAG引入了两项关键技术创新：（1）一种语义感知的异构图索引机制，将文本块和命名实体结合在一个统一结构中，减少了对复杂语义理解的依赖；（2）一种轻量级的拓扑增强检索方法，利用图结构实现高效的知识发现，而无需高级语言能力。我们的大量实验表明，MiniRAG在使用SLMs时，性能与基于LLM的方法相当，同时仅需25%的存储空间。此外，我们还贡献了一个全面的基准数据集LiHua-World，用于评估轻量级RAG系统在现实设备场景下处理复杂查询的能力。

## MiniRAG 框架

![MiniRAG](https://files.mdnice.com/user/87760/02baba85-fa69-4223-ac22-914fef7120ae.jpg)

MiniRAG 采用基于两个关键组件构建的精简工作流程:异构图索引和轻量级的基于图的知识检索。这种架构解决了设备端 RAG 系统面临的独特挑战,在效率和效果之间实现了优化。

## 安装

* 从源码安装（推荐）

```bash
cd MiniRAG
pip install -e .
```
* 从 PyPI 安装（我们的代码基于 [LightRAG](https://github.com/HKUDS/LightRAG)，因此可以直接安装）

```bash
pip install lightrag-hku
```

## 快速开始
* 所有复现代码可以在 `./reproduce` 目录下找到。
* 下载您需要的知识库数据集。
* 将数据集放入 `./dataset` 目录下。
* Note：我们已经将 LiHua-World 数据集以 `LiHuaWorld.zip` 的形式放在了 `./dataset/LiHua-World/data/` 目录下。如果您想使用其他数据集，可以将其放在 `./dataset/xxx` 目录下。


然后使用以下命令对数据集进行索引：
```bash
python ./reproduce/Step_0_index.py
python ./reproduce/Step_1_QA.py
```

或者，使用 `./main.py` 中的代码初始化 MiniRAG。


### 整体性能表
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

表中，/ 表示该方法难以生成有效响应。

## 复现
所有代码可以在 `./reproduce` 目录下找到。

## 代码结构

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

## 数据集: LiHua-World

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

LiHua-World 是一个专门为本地 RAG 场景设计的数据集，包含了一个名为 LiHua 的虚拟用户一年内的聊天记录。该数据集包含三种类型的问题：单跳、多跳和总结性问题，每个问题都配有人工标注的答案和支持文档。更多细节请参考 [LiHua-World 数据集的 README](./dataset/LiHua-World/README_CN.md)。


## Star History

<a href="https://star-history.com/#HKUDS/MiniRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
 </picture>
</a>

## Contribution

感谢MiniRAG项目的所有贡献者！

<a href="https://github.com/HKUDS/MiniRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/MiniRAG" />
</a>


## 致谢
你可以参考以下相关工作，它们为我们的框架和代码库奠定了基础：[nano-graphrag](https://github.com/gusye1234/nano-graphrag) 和 [LightRAG](https://github.com/HKUDS/LightRAG)。感谢他们的出色工作。

## 🌟引用

```python
@article{fan2025minirag,
  title={MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author={Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2501.06713},
  year={2025}
}
```

**感谢您对我们工作的关注！**
