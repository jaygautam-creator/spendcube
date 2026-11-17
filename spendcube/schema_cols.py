"""Frozen canonical column names (MASTER_ARCHITECTURE section 4).

Every stage reads/writes the canonical frame using exactly these names.
Do not rename; add new names here first.
"""
from __future__ import annotations

# ---- output contract -------------------------------------------------------
OUTPUT_COLUMNS = [
    "record_id", "include", "exclusion_reason", "supplier_key", "item_key",
    "spend", "l1", "l2", "l3", "l4", "confidence",
]
EXCLUSION_CODES = ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"]
# precedence: lower index wins (D-03)
EXCLUSION_PRECEDENCE = ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"]

# ---- config-column-key -> canonical column (stage 1) -----------------------
# value is the canonical frame column that holds the *raw string* from the file
MAPPED_FIELDS = {
    "supplier_name": "supplier_raw",
    "vendor_id": "vendor_id",
    "description": "desc_raw",
    "material_id": "material_id",
    "quantity": "quantity_raw",
    "uom": "uom",
    "amount": "amount_raw",
    "tax_rate": "tax_rate_raw",
    "tax_amount": "tax_amount_raw",
    "currency": "currency",
    "fx_rate": "fx_rate_raw",
    "invoice_date": "invoice_date_raw",
    "posting_date": "posting_date_raw",
    "status": "status_raw",
    "po_reference": "po_raw",
    "po_item": "po_item_raw",
    "invoice_number": "invoice_no_raw",
    "gl_account": "gl_account",
    "gl_description": "gl_desc",
    "cost_centre": "cost_centre",
}
# multi-column fields (list of columns, concatenated)
CONCAT_FIELDS = {"description"}
# config keys in `columns` that are not column references
COLUMN_OPTION_KEYS = {
    "llm_context", "po_regex", "amount_basis", "default_tax_rate", "currency_constant", "fx_rate_convention",
    "context", "invoice_date", "posting_date",
}

INGEST_COLUMNS = [
    "record_id", "source", "source_role", "row_order",
    *MAPPED_FIELDS.values(), "context", "llm_context",
]

# ---- columns added by later stages ----------------------------------------
NORMALIZE_COLUMNS = [
    "quantity", "amount", "tax_rate", "net_local", "fx_rate_used", "fx_method",
    "spend_rc", "invoice_date", "posting_date", "po_number", "po_item",
    "invoice_no_norm", "desc_norm", "desc_compact", "desc_blank",
    "supplier_norm", "supplier_core",
]
EXCLUSION_COLUMNS = ["status_class", "excl_code", "excl_detail", "excl_conf"]
RECONCILE_COLUMNS = ["dup_group", "match_ref", "match_type", "match_score"]
ENTITY_COLUMNS = ["supplier_key", "supplier_conf", "item_key", "item_conf"]
CLASSIFY_COLUMNS = ["unit_id", "leaf_id", "cat_conf", "cat_method"]
FINAL_COLUMNS = ["l1", "l2", "l3", "l4", "flags"]

# rows from these roles are emitted; "reference" rows (history) never are
EMIT_ROLES = ("primary", "secondary")
ROLE_REFERENCE = "reference"

CAT_METHODS = [
    "inherit", "label_lookup", "supervised", "llm", "llm_sc",
    "propagate_supplier", "propagate_gl", "fallback", "placeholder",
]
