#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build D1MT metadata
===================================================
Single source of truth for the Rhesus_D1MT (Singh B et al., D1MT monotherapy)
single-cell run. Reads the sample-sheet xlsx and writes dimt_metadata.pkl,
consumed by both cellranger_count_all.py and Phase1_qc.py so the 8-sample list
lives in exactly one place.

Contract (dict keyed by animal_id -> record):
  cr_id        cellranger --id and output dir name        (= animal_id, per naming decision)
  sample_name  cellranger --sample / FASTQ prefix         (= SK-MA-10X-8s-DK-0N)
  fastq_dirs   list of dirs for cellranger --fastqs        (single run, no cross-FC merge)
  group        D1MT | Untreated                            (the DEG contrast; -> obs['group'])
  sex, age_years, age_days, source, cohort, cryo_status    (animal metadata -> obs)
  tissue, timepoint, lanes, flowcell, run_id               (provenance -> obs)

Keying by animal_id makes cr_id = animal_id automatically (clean dir names that
match Table 1), while --sample stays the DK FASTQ prefix.

Run in Spyder (# %% cells) or standalone.
Author: Jake Lehle / Kaushal Lab
Date: July 2026
"""

# %% Cell 1 - Configuration
import os
import json
import pickle
import pandas as pd

# Cluster defaults; env overrides let this be validated locally without edits.
XLSX_PATH = os.environ.get(
    "D1MT_XLSX",
    "/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT/metadata/Rhesus_D1MT_sample_sheet_v1.xlsx",
)
OUT_DIR = os.environ.get(
    "D1MT_META_OUT",
    "/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT/metadata",
)
SHEET = "Sample_Sheet"
OUT_PKL = os.path.join(OUT_DIR, "dimt_metadata.pkl")
OUT_JSON = os.path.join(OUT_DIR, "dimt_metadata.json")  # human-readable mirror

REQUIRED_COLS = [
    "library_id", "animal_id", "group", "sex", "age_years", "age_days",
    "source", "cohort", "cryo_status", "tissue", "timepoint",
    "lanes", "flowcell", "run_id", "fastq_dir",
]
VALID_GROUPS = {"D1MT", "Untreated"}
EXPECTED_N = 8

os.makedirs(OUT_DIR, exist_ok=True)


# %% Cell 2 - Read and validate the sample sheet
print("=" * 70)
print("BUILDING D1MT METADATA")
print("=" * 70)
print(f"  Reading: {XLSX_PATH}")

df = pd.read_excel(XLSX_PATH, sheet_name=SHEET, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    raise KeyError(f"Sample sheet missing required columns: {missing_cols}")

# Drop fully-empty rows, coerce types
df = df.dropna(subset=["library_id", "animal_id"]).copy()
df["animal_id"] = df["animal_id"].astype(str).str.strip()
df["library_id"] = df["library_id"].astype(str).str.strip()
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce").astype("Int64")
df["age_days"] = pd.to_numeric(df["age_days"], errors="coerce").astype("Int64")

# --- Warn-and-continue validation (never a hard stop mid-inventory) ---
warnings = []
if len(df) != EXPECTED_N:
    warnings.append(f"expected {EXPECTED_N} samples, found {len(df)}")

bad_group = sorted(set(df["group"]) - VALID_GROUPS)
if bad_group:
    warnings.append(f"unexpected group value(s): {bad_group} (expected {VALID_GROUPS})")

dup_animal = df["animal_id"][df["animal_id"].duplicated()].tolist()
if dup_animal:
    warnings.append(f"duplicate animal_id(s): {dup_animal} (keying by animal_id assumes unique)")

dup_lib = df["library_id"][df["library_id"].duplicated()].tolist()
if dup_lib:
    warnings.append(f"duplicate library_id(s): {dup_lib}")

for d in df["fastq_dir"].unique():
    if not os.path.isdir(str(d)):
        warnings.append(f"fastq_dir not found on this host: {d}")

if warnings:
    print("\n  *** VALIDATION WARNINGS ***")
    for w in warnings:
        print(f"    - {w}")
else:
    print("  Validation clean.")


# %% Cell 3 - Assemble the metadata dict
metadata = {}
for _, r in df.iterrows():
    animal = r["animal_id"]
    metadata[animal] = {
        "cr_id": animal,                       # cellranger --id / output dir
        "sample_name": r["library_id"],        # cellranger --sample (FASTQ prefix)
        "fastq_dirs": [str(r["fastq_dir"])],   # list for --fastqs
        "group": str(r["group"]).strip(),
        "sex": str(r["sex"]).strip(),
        "age_years": int(r["age_years"]) if pd.notna(r["age_years"]) else None,
        "age_days": int(r["age_days"]) if pd.notna(r["age_days"]) else None,
        "source": str(r["source"]).strip(),
        "cohort": str(r["cohort"]).strip(),
        "cryo_status": str(r["cryo_status"]).strip(),
        "tissue": str(r["tissue"]).strip(),
        "timepoint": str(r["timepoint"]).strip(),
        "lanes": str(r["lanes"]).strip(),
        "flowcell": str(r["flowcell"]).strip(),
        "run_id": str(r["run_id"]).strip(),
    }

# %% Cell 4 - Write and summarize
with open(OUT_PKL, "wb") as f:
    pickle.dump(metadata, f)
with open(OUT_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

n_d1mt = sum(1 for m in metadata.values() if m["group"] == "D1MT")
n_untx = sum(1 for m in metadata.values() if m["group"] == "Untreated")

print("\n" + "=" * 70)
print(f"  Wrote {len(metadata)} records  (D1MT={n_d1mt}, Untreated={n_untx})")
print(f"    pkl : {OUT_PKL}")
print(f"    json: {OUT_JSON}")
print("=" * 70)
print(f"\n  {'animal':<8}{'group':<11}{'--sample':<22}{'sex':<5}{'cohort':<12}{'cryo'}")
for a, m in metadata.items():
    print(f"  {a:<8}{m['group']:<11}{m['sample_name']:<22}{m['sex']:<5}"
          f"{m['cohort']:<12}{m['cryo_status']}")
