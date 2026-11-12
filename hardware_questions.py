import argparse
import json

from hardware_knowledge import DATA_DIR, build_manual_store, load_manuals, search_manuals


def main():
    parser = argparse.ArgumentParser(description="查看 Tapo 练习问题，或运行本地检索回归检查。")
    parser.add_argument("--check-retrieval", action="store_true", help="使用真实嵌入检索，不调用聊天模型")
    args = parser.parse_args()
    bank = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))

    if not args.check_retrieval:
        for case in bank["cases"]:
            print(f"{case['id']} | {case['device_model']} | {case['topic']} | {case['question']}")
        print("\n多轮演示场景")
        for case in bank["conversations"]:
            print(f"{case['id']} | {case['device_model']} | {case['issue']}")
            print(f"反馈：{case['feedback']}；预期：{case['expected']}")
        return

    valid_ids = {doc.metadata["id"] for doc in load_manuals()}
    case_ids = [case["id"] for case in bank["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("问题编号存在重复")
    for case in bank["cases"]:
        if not set(case["expected_ids"]) <= valid_ids:
            raise ValueError(f"{case['id']} 引用了不存在的资料")

    manual_store = build_manual_store()
    passed = 0
    for case in bank["cases"]:
        results = search_manuals(manual_store, case["question"], case["device_model"])
        retrieved_ids = [doc.metadata["id"] for doc, score in results]
        route = "generate_reply" if results else "no_evidence"
        relevant = not case["expected_ids"] or bool(set(retrieved_ids) & set(case["expected_ids"]))
        correct = route == case["expected_route"] and relevant
        passed += int(correct)
        print(f"{'PASS' if correct else 'FAIL'} {case['id']} | {case['question']}")
        print(f"  资料 {retrieved_ids}，出口 {route}")

    print(f"\n检索回归 {passed}/{len(bank['cases'])} 通过。仅验证证据命中与出口，不代表模型回答或真机修复成功率。")
    if passed != len(bank["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
