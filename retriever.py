import json
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9#+\-\.]+", text)
    return [t for t in tokens if len(t) > 1]


def build_document(item: Dict) -> str:
    test_type_map = {
        "A": "ability aptitude cognitive reasoning",
        "B": "biodata situational judgement scenario behaviour",
        "C": "competencies",
        "D": "development 360 feedback",
        "E": "exercises assessment center",
        "K": "knowledge skills technical",
        "P": "personality behaviour traits questionnaire",
        "S": "simulation practical hands-on",
    }
    test_type_words = test_type_map.get(item.get("test_type", ""), "")
    levels = " ".join(item.get("job_levels", []))
    langs = " ".join(item.get("languages", [])[:5])
    remote = "remote online" if item.get("remote_testing") else ""
    adaptive = "adaptive irt" if item.get("adaptive_irt") else ""

    doc = (
        f"{item['name']} {item['name']} "
        f"{item.get('description', '')} "
        f"{test_type_words} "
        f"{levels} "
        f"{langs} "
        f"{remote} {adaptive}"
    )
    return doc


class CatalogRetriever:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)

        self.documents = [build_document(item) for item in self.catalog]
        self.tokenized_docs = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"CatalogRetriever: {len(self.catalog)} assessments indexed")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        import numpy as np
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                item = dict(self.catalog[idx])
                item["_score"] = float(scores[idx])
                results.append(item)
        return results

    def get_by_name(self, name: str):
        name_lower = name.lower()
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    def get_all(self) -> List[Dict]:
        return self.catalog


_retriever = None

def get_retriever() -> CatalogRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever()
    return _retriever