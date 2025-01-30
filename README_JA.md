# MiniRAG: 極めてシンプルな検索強化生成に向けて

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)


コードリポジトリ: **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**
<br />

[Tianyu Fan](https://tianyufan0504.github.io/), [Jingyuan Wang](), [Xubin Ren](https://ren-xubin.github.io/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />
</div>

<a href='https://arxiv.org/abs/2501.06713'><img src='https://img.shields.io/badge/arXiv-2501.06713-b31b1b'>


## 🌍 READMEの翻訳

[English](./README.md) | [中文](./README_CN.md)

## TLDR
MiniRAGは、異種グラフインデックスと軽量なトポロジー強化検索を通じて、小さなモデルでも優れたRAGパフォーマンスを実現する極めてシンプルな検索強化生成フレームワークです。

## 概要
効率的で軽量な検索強化生成（RAG）システムの需要が高まる中、既存のRAGフレームワークに小型言語モデル（SLM）を導入する際の重大な課題が浮き彫りになっています。現在のアプローチは、SLMの限られた意味理解とテキスト処理能力のために深刻な性能低下に直面しており、リソースが限られたシナリオでの広範な採用に障害をもたらしています。これらの根本的な制限に対処するために、私たちは極めてシンプルで効率的な新しいRAGシステムである**MiniRAG**を提案します。**MiniRAG**は、2つの重要な技術革新を導入しています：（1）テキストチャンクと名前付きエンティティを統一構造に組み合わせる意味認識異種グラフインデックスメカニズム、これにより複雑な意味理解への依存を減らします。（2）高度な言語能力を必要とせずにグラフ構造を活用して効率的な知識発見を実現する軽量なトポロジー強化検索アプローチ。私たちの広範な実験は、**MiniRAG**がSLMを使用してもLLMベースの方法と同等の性能を達成しながら、ストレージスペースの25％しか必要としないことを示しています。さらに、複雑なクエリを持つ現実的なオンデバイスシナリオで軽量RAGシステムを評価するための包括的なベンチマークデータセットLiHua-Worldも提供します。

## MiniRAGフレームワーク

![MiniRAG](https://files.mdnice.com/user/87760/02baba85-fa69-4223-ac22-914fef7120ae.jpg)

MiniRAGは、異種グラフインデックスと軽量なグラフベースの知識検索という主要なコンポーネントに基づいて構築された簡素化されたワークフローを採用しています。このアーキテクチャは、オンデバイスRAGシステムが直面する独自の課題に対処し、効率と効果の両方を最適化します。


## インストール

* ソースからインストール（推奨）

```bash
cd MiniRAG
pip install -e .
```
* PyPIからインストール（私たちのコードは[LightRAG](https://github.com/HKUDS/LightRAG)に基づいているため、直接インストールできます）

```bash
pip install lightrag-hku
```

## クイックスタート
* すべてのコードは`./reproduce`にあります。
* 必要なデータセットをダウンロードします。
* データセットを`./dataset`ディレクトリに配置します。
* 注：LiHua-Worldデータセットは`./dataset/LiHua-World/data/`に`LiHuaWorld.zip`として既に配置されています。他のデータセットを使用したい場合は、`./dataset/xxx`に配置できます。


次に、以下のbashコマンドを使用してデータセットをインデックスします：
```bash
python ./reproduce/Step_0_index.py
python ./reproduce/Step_1_QA.py
```

または、`./main.py`のコードを使用してMiniRAGを初期化します。


### 全体のパフォーマンステーブル
| モデル | NaiveRAG | | GraphRAG | | LightRAG | | **MiniRAG** | |
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


表中、/はその方法が効果的な応答を生成するのに苦労していることを意味します。

## 再現
すべてのコードは`./reproduce`ディレクトリにあります。

## コード構造

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

## データセット: LiHua-World

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

LiHua-Worldは、仮想ユーザーLiHuaの1年間のチャット記録を含む、オンデバイスRAGシナリオ専用に設計されたデータセットです。このデータセットには、シングルホップ、マルチホップ、およびサマリーの3種類の質問が含まれており、各質問には手動で注釈が付けられた回答とサポート文書がペアになっています。詳細については、[LiHua-WorldデータセットのREADME](./dataset/LiHua-World/README.md)を参照してください。


## Star History

<a href="https://star-history.com/#HKUDS/MiniRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
 </picture>
</a>

## Contribution

MiniRAGプロジェクトのすべての貢献者に感謝します！

<a href="https://github.com/HKUDS/MiniRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/MiniRAG" />
</a>

## 謝辞
私たちのフレームワークとコードリポジトリの基礎となる関連作業については、[nano-graphrag](https://github.com/gusye1234/nano-graphrag)および[LightRAG](https://github.com/HKUDS/LightRAG)を参照してください。素晴らしい仕事に感謝します。

## 🌟引用

```python
@article{fan2025minirag,
  title={MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author={Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2501.06713},
  year={2025}
}
```

**私たちの仕事に興味を持っていただき、ありがとうございます！**
