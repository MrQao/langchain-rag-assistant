# LangChain RAG Assistant

基于 LangChain 和 DeepSeek 的文档问答助手。将 UTF-8 文本资料放入 `data/` 后，可以在终端提问、连续追问，并通过回答中的引用编号核对原文。

## 功能

- **多文档检索**：切分文本，使用本地中文嵌入模型 BGE 生成向量，按相似度检索相关片段。
- **基于资料回答**：将检索到的片段交给 DeepSeek 生成回答。没有片段达到相似度门槛时，直接提示资料不足。
- **原文引用**：展示编号对应的原文和文件路径，并提示超出本轮片段范围的引用编号。
- **多轮对话**：结合当前会话的历史改写追问，再执行检索。输入 `/new` 开启独立会话。
- **索引复用**：将文档向量保存到本地，后续运行直接加载。

## 快速开始

需要 Python 3.11 和 DeepSeek API Key。在项目目录中执行：

```bash
conda create -n langchain-rag-assistant python=3.11 -y
conda activate langchain-rag-assistant
python -m pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中设置 `DEEPSEEK_API_KEY`，然后启动：

```bash
python main.py
```

首次运行会下载 `BAAI/bge-small-zh-v1.5` 并建立本地向量索引。嵌入和检索在本地执行；生成回答时会将问题和检索到的片段发送给 DeepSeek，改写追问时会发送相关会话历史。

## 使用示例

仓库包含两份图书馆示例资料。可以先问「星河图书馆的书可以借多久？」，再追问「那能续借吗？」。终端会展示回答和编号对应的原文。

- `/new`：开始新对话。
- `/exit`：退出程序。会话历史随程序退出清除。
- `python demo.py`：运行检索分数、多轮问答和引用编号校验示例。

新增、删除或修改 `data/` 中的资料后，执行下面的命令更新索引：

```bash
python -c 'from main import build_vector_store; build_vector_store(rebuild_index=True)'
```

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `main.py` | 模型与索引初始化、检索问答链、终端入口 |
| `demo.py` | 检索、会话和引用校验示例 |
| `data/` | 用于检索的 UTF-8 文本资料 |
| `.env.example` | API Key 配置模板 |
| `requirements.txt` | Python 依赖 |

本地配置文件 `.env` 和生成的索引 `vector_store.json` 均已加入 Git 忽略规则。
