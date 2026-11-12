import argparse

from main import find_invalid_citations, initialize, print_result
from hardware_knowledge import normalize_device_model, search_manuals


def demo_retrieval(vector_store, *, dataset="library", device_model="C200"):
    test_questions = [
        "每位读者最多可以借几本书？",
        "自习室最多预约多久？",
        "图书馆停车收费多少？",
        "怎么做红烧肉？",
    ]
    if dataset == "hardware":
        test_questions = [
            "换了路由器之后摄像头连不上 Wi-Fi。",
            "存储卡一直提示未检测到。",
            "手机预览听不到房间里的声音。",
            "如何做红烧肉？",
        ]

    for question in test_questions:
        print("\n问题：", question)

        if dataset == "hardware":
            results = search_manuals(vector_store, question, device_model)
        else:
            results = vector_store.similarity_search_with_score(question, k=2)

        for doc, score in results:
            print("相似度：", round(float(score), 4))
            print("正文：", doc.page_content)
            print("来源：", doc.metadata["source"])


def demo_conversation(chat_chain, *, dataset="library", device_model="C200"):
    config = {
        "configurable": {
            "session_id": "chat-1"
        }
    }

    questions = [
        "图书馆周一开放吗？",
        "每人最多借几本书？",
        "借期是多少天？",
        "自习室需要预约吗？",
        "那最多预约多久？",
        "借期是多少天？自习室每次最多预约多久？",
    ]
    if dataset == "hardware":
        questions = [
            "摄像头换了路由器后连不上 Wi-Fi，应该怎么排查？",
            "那要用哪个 App 重新设置？",
            "存储卡提示初始化失败，应该先检查什么？",
            "格式化它会删除之前的录像吗？",
        ]

    for question in questions:
        result = chat_chain.invoke(
            {"question": question, "device_model": device_model},
            config=config,
        )
        print_result(result)


def demo_citations():
    test_data = {
        "answer": "借期为 14 天 [1]，可以续借一次 [9]。",
        "evidence": [object(), object()],
    }

    print(find_invalid_citations(test_data))  # [9]


def main():
    parser = argparse.ArgumentParser(description="LangChain 检索、多轮问答与引用演示")
    parser.add_argument("--dataset", choices=["library", "hardware"], default="library")
    parser.add_argument("--device-model", default="C200")
    args = parser.parse_args()
    device_model = normalize_device_model(args.device_model)
    vector_store, chat_chain = initialize(dataset=args.dataset)
    demo_retrieval(vector_store, dataset=args.dataset, device_model=device_model)
    demo_conversation(chat_chain, dataset=args.dataset, device_model=device_model)
    demo_citations()


if __name__ == "__main__":
    main()
