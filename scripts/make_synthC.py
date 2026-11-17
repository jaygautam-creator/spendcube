#!/usr/bin/env python3
"""Deterministic SYNTHETIC "company C" fixture (a hospital group) for tests. No kit rows or strings.

Writes tests/fixtures/synthC/{taxonomy_C.csv, lines_C.csv, truth.csv}.
Features: 4-level ragged PATH tree, ';' delimiter, European decimals, dd.mm.yyyy dates, German statuses,
gross amounts + tax-rate column, EUR/CHF lines without a rate (static fx), supplier-name variants,
intercompany + tax-posting lines, blank descriptions, ~3000 lines.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthC"
N = 3000
STATIC = {"EUR": 0.92, "CHF": 0.88}          # units of currency per 1 USD (config fx.static_rates)

# leaf path -> (description templates, supplier pool, base unit-price range in USD)
TREE = {
    ("Clinical Supplies", "Consumables", "Gloves and PPE"): (["Nitrile examination gloves {n}", "Handschuhe steril Gr. {n}", "Face mask FFP2 box {n}"], ["Medico Supply", "SafeHands"], (20, 400)),
    ("Clinical Supplies", "Consumables", "Wound Care"): (["Wundauflage steril {n}x{n}cm", "Compression bandage {n}", "Suture kit {n}"], ["Medico Supply", "Wundwerk"], (15, 300)),
    ("Clinical Supplies", "Implants", "Orthopaedic", "Hip Systems"): (["Hip stem cementless size {n}", "Acetabular cup {n}mm", "Hueftprothese Set {n}"], ["OrthoNova", "Kinetik Implants"], (900, 6000)),
    ("Clinical Supplies", "Implants", "Cardiac", "Stents"): (["Drug eluting stent {n}mm", "Koronarstent {n}", "Balloon catheter {n}F"], ["CardioPlus", "Kinetik Implants"], (700, 4000)),
    ("Pharmaceuticals", "Generics"): (["Paracetamol 500mg tabs {n}", "Ibuprofen Infusion {n}ml", "Antibiotikum Amoxicillin {n}"], ["PharmaDirekt", "Medico Supply"], (10, 900)),
    ("Pharmaceuticals", "Oncology", "Chemotherapy Agents"): (["Cisplatin vial {n}mg", "Zytostatikum Carboplatin {n}", "Oncology infusion bag {n}"], ["OncoMed", "PharmaDirekt"], (300, 5000)),
    ("Facilities", "Maintenance", "HVAC"): (["Klimaanlage Wartung Block {n}", "HVAC filter replacement {n}", "Chiller service ward {n}"], ["ClimaTech", "BauService"], (100, 3000)),
    ("Facilities", "Cleaning Services"): (["Reinigung Station {n}", "Cleaning services month {n}", "Desinfektion OP Bereich {n}"], ["CleanCo", "BauService"], (200, 5000)),
    ("IT", "Software", "Licences"): (["Software licence renewal seats {n}", "Lizenz Klinikinformationssystem {n}", "Cloud subscription tier {n}"], ["Softwerk", "CloudNine"], (300, 9000)),
    ("IT", "Hardware"): (["Laptop docking station {n}", "Server RAM module {n}GB", "Netzwerk Switch {n} Port"], ["Softwerk", "Hardwerk"], (80, 4000)),
    ("Professional Services", "Legal"): (["Legal advice contract review {n}", "Rechtsberatung Mandat {n}"], ["Kanzlei Berger", "Kanzlei Voss"], (500, 8000)),
    ("Professional Services", "Consulting", "Management Consulting"): (["Strategy consulting phase {n}", "Beratung Prozessoptimierung {n}"], ["Voss Partner", "Kanzlei Berger"], (1000, 15000)),
    ("Food and Catering",): (["Catering patient meals week {n}", "Lebensmittel Lieferung {n}", "Kitchen supplies {n}"], ["FrischKost", "CleanCo"], (50, 2500)),
}
NON_PROC = [
    ("Konzernverrechnung Intercompany", "Intercompany charge entity {n}", "Klinikum Holding GmbH", "INTERCOMPANY"),
    ("Umsatzsteuer", "Umsatzsteuer Vorauszahlung {n}", "Finanzamt Mitte", "TAX"),
]
VARIANTS = {"Medico Supply": ["Medico Supply GmbH", "MEDICO SUPPLY GMBH", "Medico  Supply", "Medico Supply"],
            "Kinetik Implants": ["Kinetik Implants AG", "KINETIK IMPLANTS", "Kinetik  Implants AG"],
            "OrthoNova": ["OrthoNova GmbH", "Ortho Nova GmbH", "ORTHONOVA"]}
STATUSES = [("Bezahlt", 0.62), ("Freigegeben", 0.25), ("Storniert", 0.05), ("Abgelehnt", 0.04), ("In Klaerung", 0.04)]
RATES = ["19", "7", "19%", "0"]


def eu(x: float) -> str:
    """European number format: 1.234,56"""
    s = f"{abs(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("-" + s) if x < 0 else s


def main() -> None:
    rnd = random.Random(20261117)
    OUT.mkdir(parents=True, exist_ok=True)
    leaves = list(TREE)
    with open(OUT / "taxonomy_C.csv", "w", newline="", encoding="utf-8") as f:
        f.write("category_path\n")
        for p in leaves:
            f.write(" > ".join(p) + "\n")
    rows, truth = [], []
    for i in range(1, N + 1):
        lid = f"C{i:06d}"
        if rnd.random() < 0.05:
            gl_txt, tpl, supplier, kind = rnd.choice(NON_PROC)
            leaf, desc = None, tpl.format(n=rnd.randint(1, 99))
            gl_acc = "9" + str(rnd.randint(100, 199)) if kind == "INTERCOMPANY" else "1776"
            vendor = "V9001" if kind == "INTERCOMPANY" else "V9002"
            net_usd = rnd.uniform(500, 20000)
            tax_rate = "0"
            excl = "NON_PROCUREMENT"
        else:
            leaf = rnd.choice(leaves)
            tpls, pool, (lo, hi) = TREE[leaf]
            base = rnd.choice(pool)
            supplier = rnd.choice(VARIANTS.get(base, [base]))
            desc = rnd.choice(tpls).format(n=rnd.randint(1, 60))
            if rnd.random() < 0.15:
                desc = ""                                   # blank descriptions
            gl_acc, gl_txt = "4" + str(rnd.randint(100, 899)), "Aufwand " + leaf[0]
            vendor = "V" + str(1000 + sum(map(ord, base)) % 997)
            net_usd = rnd.uniform(lo, hi) * rnd.choice([1, 1, 1, 2, 5])
            tax_rate = rnd.choice(RATES)
            excl = ""
        cur = rnd.choices(["EUR", "CHF", "USD"], [0.7, 0.2, 0.1])[0]
        rate = 1.0 if cur == "USD" else STATIC[cur]
        net_local = round(net_usd * rate, 2)
        pct = float(tax_rate.rstrip("%")) / 100
        gross = round(net_local * (1 + pct), 2)
        status = rnd.choices([s for s, _ in STATUSES], [w for _, w in STATUSES])[0]
        if excl == "NON_PROCUREMENT" and status in ("Storniert", "Abgelehnt", "In Klaerung"):
            status = "Bezahlt"
        if status == "Storniert":
            excl = "CANCELLED"
        elif status in ("Abgelehnt", "In Klaerung"):
            excl = "NOT_APPROVED"
        day, month = rnd.randint(1, 28), rnd.randint(1, 12)
        rows.append([lid, supplier, vendor, desc, str(rnd.randint(1, 40)), "ST", eu(gross), tax_rate, cur,
                     f"{day:02d}.{month:02d}.2024", status, gl_acc, gl_txt])
        # ground truth is computed from the NET amount (what a correct pipeline recovers from the gross)
        true_spend = 0.0 if excl else round(net_local / rate, 2)
        path = (list(leaf) + ["", "", "", ""])[:4] if leaf else ["", "", "", ""]
        truth.append([lid, *path, excl, f"{net_local / rate:.2f}", f"{true_spend:.2f}"])
    with open(OUT / "lines_C.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["belegnr", "lieferant", "lieferantennr", "beschreibung", "menge", "einheit", "betrag_brutto",
                    "mwst_satz", "waehrung", "rechnungsdatum", "status", "sachkonto", "sachkontotext"])
        w.writerows(rows)
    with open(OUT / "truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "l1", "l2", "l3", "l4", "true_exclusion", "net_usd", "true_spend_usd"])
        w.writerows(truth)
    print("wrote", OUT, len(rows), "lines")


if __name__ == "__main__":
    main()
