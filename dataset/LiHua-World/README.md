# LiHua-World Dataset

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

[中文说明](./README_CN.md)


LiHua-World is a dataset specifically designed for local RAG (Retrieval-Augmented Generation) scenarios. It contains one year's worth of chat records from a virtual user named LiHua.

## Dataset Features

- Includes three types of questions:
  - Single-hop
  - Multi-hop
  - Summary
- Each question is accompanied by manually annotated answers and supporting documents
- The chat records cover various aspects of daily life, including:
  - Social interactions
  - Fitness training
  - Entertainment activities
  - Life affairs
  - ...
## Dataset Structure

The dataset mainly consists of the following parts:

### 1. Original Chat Records (/data)
- Chat messages organized in chronological order
- Each message contains:
  - Timestamp
  - Sender
  - Message content
  - Message type

### 2. Q&A Data (/qa)
- query_set.csv: Contains questions, standard answers, and evidence
- query_set.json: JSON format version of the CSV file

### 3. Metadata
- User information
- Time range: January 2026 to December 2026
- List of conversation participants

## Usage Instructions

Step 1. Unzip the `LiHuaWorld.zip` file in the `./data` directory to obtain the original chat records.

Step 2. Use all the chat records in the `./data` directory as the knowledge base.

Step 3. Use `query_set.csv` or `query_set.json` in the `./qa` directory as the question set to conduct RAG testing.
