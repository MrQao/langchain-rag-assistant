# LangChain 学习项目

使用 LangChain 和 DeepSeek 搭建本地资料问答助手。

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

终端里输入 `/new` 开始新对话，输入 `/exit` 退出。要运行检索分数、多轮问答和引用校验示例，执行 `python demo.py`。

`.env` 已被 Git 忽略，请勿提交真实 API Key。
