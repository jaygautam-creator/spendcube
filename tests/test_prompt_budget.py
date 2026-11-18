import pandas as pd

from spendcube.classify.cascade import _prompt_budget
from spendcube.taxonomy.tree import Taxonomy
from spendcube.config.schema import TaxonomyCfg


def _router():
    from spendcube.llm.router import Router
    return Router({"enabled": True, "chain": [
        {"id": "x", "provider": "mock", "model": "x", "rpm": 15, "tpm": 250000}],
        "roles": {"top_spend": ["x"], "bulk": ["x"]}}, job_budget=False)


def test_prompt_budget_skips_definitions_when_tpm_pressure_is_high():
    tax = Taxonomy.load(TaxonomyCfg(file="tests/fixtures/synthC/taxonomy_C.csv", format="auto"))
    units = pd.DataFrame([{"supplier": "Supplier", "gl": "", "ctx": "", "text": "item"}])
    cfg, report = _prompt_budget({"batch": {"max": 60}, "prompt_budget": {"enabled": True,
        "tpm_fraction": .8, "definition_chars_per_leaf": 5000}}, _router(), tax, "", units, [0])
    assert report["definitions_skipped"]
    assert cfg["batch"]["max"] == 60
    assert report["tokens_per_request_with_definitions"] > report["tokens_per_request_without_definitions"]


def test_prompt_budget_reduces_batch_until_under_limit():
    tax = Taxonomy.load(TaxonomyCfg(file="tests/fixtures/synthC/taxonomy_C.csv", format="auto"))
    units = pd.DataFrame([{"supplier": "S" * 40, "gl": "G" * 40, "ctx": "C" * 140,
                           "text": "x" * 160}])
    cfg, report = _prompt_budget({"batch": {"max": 60}, "prompt_budget": {"enabled": True,
        "tpm_fraction": .1, "min_batch": 15}}, _router(), tax, "", units, [0])
    assert report["definitions_skipped"]
    assert cfg["batch"]["max"] < 60
    assert cfg["batch"]["max"] >= 15


def test_ragged_output_paths_keep_exact_names_and_pad_l4():
    from spendcube.taxonomy.tree import Taxonomy
    tax = Taxonomy([("Clinical Supplies", "Consumables", "PPE"),
                    ("Clinical Supplies", "Implants", "Cardiac", "Stents")])
    assert tax.path_padded("T001") == ["Clinical Supplies", "Consumables", "PPE", ""]
    assert tax.path_padded("T002") == ["Clinical Supplies", "Implants", "Cardiac", "Stents"]

