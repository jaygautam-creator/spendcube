"""Pydantic config models (MASTER_ARCHITECTURE section 3).

Rule: every behaviour that might need to differ for an unseen company has a key
here with a sensible default. Unknown keys are allowed (forward compatible) but
logged by the loader.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class ReadCfg(_Base):
    delimiter: str = "auto"
    encoding: str = "auto"
    sheet: Optional[Union[str, int]] = None


class RecordIdCfg(_Base):
    column: Optional[str] = None      # None -> row number
    prefix: str = ""


class StatusMapCfg(_Base):
    INCLUDE: list[str] = Field(default_factory=list)
    CANCELLED: list[str] = Field(default_factory=list)
    NOT_APPROVED: list[str] = Field(default_factory=list)
    default: Literal["INCLUDE", "NOT_APPROVED"] = "INCLUDE"


class IntraDedupCfg(_Base):
    enabled: bool = False
    keys: list[str] = Field(default_factory=list)


class SourceCfg(_Base):
    name: str
    role: Literal["primary", "secondary", "reference"] = "primary"
    file: Union[str, list[str]]
    required: bool = True
    read: ReadCfg = Field(default_factory=ReadCfg)
    record_id: RecordIdCfg = Field(default_factory=RecordIdCfg)
    columns: dict[str, Any] = Field(default_factory=dict)
    number_format: Union[str, dict[str, str]] = "auto"
    status_map: Optional[StatusMapCfg] = None
    intra_source_dedup: IntraDedupCfg = Field(default_factory=IntraDedupCfg)
    limit: Optional[int] = None


class TaxonomyCfg(_Base):
    file: str
    format: Literal["auto", "wide", "path", "adjacency"] = "auto"
    path_separator: str = " > "
    columns: Optional[list[str]] = None    # explicit level columns (wide) or [id, parent, name] (adjacency) or [path col]
    sheet: Optional[Union[str, int]] = None
    read: ReadCfg = Field(default_factory=ReadCfg)


class FxCfg(_Base):
    files: list[str] = Field(default_factory=list)
    plausibility_band: tuple[float, float] = (0.5, 2.0)
    static_rates: dict[str, float] = Field(default_factory=dict)
    # table layout (columns of the fx file)
    currency_col: str = "currency"
    month_col: str = "month"
    rate_col: str = "units_per_usd"


class VendorPropCfg(_Base):
    enabled: bool = True
    min_share: float = 0.9
    min_rows: int = 3
    codes: list[str] = Field(default_factory=lambda: ["NON_PROCUREMENT"])


class ExclusionRule(_Base):
    code: str
    field: str
    regex: Optional[str] = None
    equals: Optional[list[str]] = None
    prefix: Optional[str] = None
    in_: Optional[list[str]] = Field(default=None, alias="in")
    action: Literal["exclude", "include"] = "exclude"   # include = protects a row (D-17)


class LabelsCfg(_Base):
    file: str
    join_on: str = "record_id"
    columns: dict[str, str] = Field(default_factory=dict)  # canonical -> file column
    id_prefix: str = ""
    read: ReadCfg = Field(default_factory=ReadCfg)


class OutputCfg(_Base):
    float_decimals: int = 2


class Config(_Base):
    company: str
    candidate_name: str = "jaygautam"
    output_suffix: Optional[str] = None
    reporting_currency: str = "USD"
    industry_hint: str = ""
    input_dir: str = "."
    kit_dir: Optional[str] = None
    taxonomy: TaxonomyCfg
    sources: list[SourceCfg]
    fx: FxCfg = Field(default_factory=FxCfg)
    exclusion_rules: list[ExclusionRule] = Field(default_factory=list)
    vendor_propagation: VendorPropCfg = Field(default_factory=VendorPropCfg)
    labels: Optional[LabelsCfg] = None
    reconciliation: Optional[dict[str, Any]] = None
    entities: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    output: OutputCfg = Field(default_factory=OutputCfg)
    valid_years: Optional[tuple[int, int]] = None
    null_description_tokens: list[str] = Field(default_factory=list)
    flags: dict[str, Any] = Field(default_factory=dict)

    @field_validator("exclusion_rules", "null_description_tokens", mode="before")
    @classmethod
    def _none_to_empty(cls, v):      # a YAML key whose rules are all commented out parses as null
        return [] if v is None else v

    @property
    def out_suffix(self) -> str:
        return self.output_suffix or self.company

    @property
    def out_filename(self) -> str:
        return f"{self.candidate_name}__{self.out_suffix}.csv"
