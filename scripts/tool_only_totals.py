import sys
import pandas as pd

o = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
o["spend"] = pd.to_numeric(o["spend"])
t = o[o.record_id.str.startswith("TOOL:")]
print("tool rows", len(t), "included", (t.include == "1").sum(), "ZAR", round(t.loc[t.include == "1", "spend"].sum()))
print(t.exclusion_reason.value_counts().to_dict())
e = o[o.record_id.str.startswith("ERP:")]
print("ERP ZAR", round(e.spend.sum()), "total counted ZAR", round(o.spend.sum()))
