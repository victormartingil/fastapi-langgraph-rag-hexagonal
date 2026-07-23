import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval
DATASET = Path(__file__).parents[2] / "evals" / "dataset.json"


def test_dataset_has_30_unique_cases_and_required_adversarial_categories() -> None:
    payload = json.loads(DATASET.read_text())
    cases = payload["cases"]
    ids = [case["id"] for case in cases]
    categories = {case["category"] for case in cases}

    assert len(cases) == 30
    assert len(ids) == len(set(ids))
    assert {
        "answerable",
        "out_of_domain",
        "paraphrase",
        "multilingual",
        "competing_sources",
        "indirect_prompt_injection",
    } <= categories


def test_every_relevant_document_exists_and_answerability_is_consistent() -> None:
    payload = json.loads(DATASET.read_text())
    document_ids = {document["id"] for document in payload["documents"]}

    for case in payload["cases"]:
        assert set(case["relevant_documents"]) <= document_ids
        assert bool(case["relevant_documents"]) is bool(case["answerable"])
