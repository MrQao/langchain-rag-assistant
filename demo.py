from main import find_invalid_citations, initialize, print_result


def demo_retrieval(vector_store):
    test_questions = [
        "每位读者最多可以借几本书？",
        "自习室最多预约多久？",
        "图书馆停车收费多少？",
        "怎么做红烧肉？",
    ]

    for question in test_questions:
        print("\n问题：", question)

        results = vector_store.similarity_search_with_score(
            question,
            k=2,
        )

        for doc, score in results:
            print("相似度：", round(float(score), 4))
            print("正文：", doc.page_content)
            print("来源：", doc.metadata["source"])


def demo_conversation(chat_chain):
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

    for question in questions:
        result = chat_chain.invoke(
            {"question": question},
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
    vector_store, chat_chain = initialize()
    demo_retrieval(vector_store)
    demo_conversation(chat_chain)
    demo_citations()


if __name__ == "__main__":
    main()
