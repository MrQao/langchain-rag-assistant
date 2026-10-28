from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def test_model():
    model = init_chat_model("deepseek-flash")
    response = model.invoke("请用一句话解释什么是 LangChain。")
    print(response.content)


test_model()