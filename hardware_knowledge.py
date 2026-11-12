import json
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


DATA_DIR = Path(__file__).resolve().parent / "hardware_data"
MIN_SCORE = 0.5
RETRIEVAL_K = 3


def normalize_device_model(value):
    value = unicodedata.normalize("NFKC", value).strip().upper()
    compact = re.sub(r"[\s-]+", "", value)
    for prefix in ("TPLINKTAPO", "TAPO", "TPLINK"):
        if compact.startswith(prefix):
            compact = compact[len(prefix):]
            break
    if compact in {"C200", "C210"}:
        return f"Tapo {compact}"
    return value


def load_manuals():
    knowledge = json.loads((DATA_DIR / "knowledge.json").read_text(encoding="utf-8"))
    documents = []
    seen_ids = set()
    for entry in knowledge["entries"]:
        if entry["id"] in seen_ids:
            raise ValueError(f"资料编号重复：{entry['id']}")
        seen_ids.add(entry["id"])
        sources = [knowledge["sources"][source_id] for source_id in entry["source_ids"]]
        documents.append(Document(
            page_content=(
                f"{entry['title']}\n"
                f"常见描述：{'；'.join(entry['symptoms'])}\n"
                f"{entry['content']}\n"
                f"适用范围：{entry['scope']}"
            ),
            metadata={
                "id": entry["id"],
                "device_models": entry["device_models"],
                "title": entry["title"],
                "topic": entry["topic"],
                "sources": sources,
                "reviewed_on": knowledge["reviewed_on"],
            },
        ))
    return documents


def build_manual_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return InMemoryVectorStore.from_documents(load_manuals(), embedding=embeddings)


def search_manuals(manual_store, issue, device_model):
    device_model = normalize_device_model(device_model)
    results = manual_store.similarity_search_with_score(
        issue,
        k=RETRIEVAL_K,
        filter=lambda doc: device_model in doc.metadata["device_models"],
    )
    return [(doc, score) for doc, score in results if score >= MIN_SCORE]
