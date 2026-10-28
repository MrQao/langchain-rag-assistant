# LangChain 学习项目

逐步搭建本地资料问答助手。目前实现通过 LangChain 调用 DeepSeek。

## 运行

```bash
conda create -n langchain python=3.11 -y
conda activate langchain
python -m pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`，然后运行：

```bash
python main.py
```

`.env` 已被 Git 忽略，请勿提交真实 API Key。
