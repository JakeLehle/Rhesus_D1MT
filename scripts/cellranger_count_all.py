#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell Ranger Count - Rhesus D1MT Monotherapy Project
====================================================
Reads dimt_metadata.pkl and runs `cellranger count` for all 8 samples
(4 D1MT, 4 Untreated) at the week-15 BAL timepoint.

Reference: combined Mmul10 + MtbCDC1551 + SIVmac239 (v10).
  NOTE: Mtb and SIV are kept in the reference only for consistency with the
  D1MT+cART project. There is no SIV in these animals at wk15 (pre-coinfection)
  and 3' chemistry cannot capture Mtb, so any pathogen signal is treated as
  NOISE downstream. Phase 1 stashes pathogen counts, reports per-sample
  enrichment, and excludes pathogen genes from HVG/PCA/clustering.

Design notes vs the cART pipeline:
  - Single flowcell (B23JJN7LT3): one fastq dir per sample, no cross-FC merge.
  - 3' chemistry only: chemistry cascade trimmed so a bad auto-detect cannot
    silently succeed as 5'.
  - BAMs retained (--create-bam=true) for downstream work.
  - Output dir per sample is named by animal_id (cr_id), --sample is the DK
    FASTQ prefix.

Metadata contract (per record, keyed by animal_id):
  cr_id, sample_name, fastq_dirs[list], group, sex, age_years, age_days,
  source, cohort, cryo_status, tissue, timepoint, lanes, flowcell, run_id

Author: Jake Lehle / Kaushal Lab
Date: July 2026
"""

import os
import sys
import shutil
import subprocess
import pickle
import json
import traceback
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = "/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT"
TRANSCRIPTOME = "/master/jlehle/WORKING/SC/REF/Mmul10_MtbCDC1551_SIVmac239_v10"
METADATA_PKL = os.path.join(BASE_DIR, "metadata", "dimt_metadata.pkl")
OUTPUT_BASE = os.path.join(BASE_DIR, "cellranger_counts")
LOG_DIR = os.path.join(BASE_DIR, "logs")
NCPUS = 32  # Kept modest: BAM output + multi-contig reference is memory-heavy

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# LOAD METADATA
# =============================================================================
print("=" * 80)
print("LOADING D1MT METADATA")
print("=" * 80)

with open(METADATA_PKL, "rb") as f:
    metadata = pickle.load(f)
print(f"Loaded {len(metadata)} samples")

n_d1mt = sum(1 for m in metadata.values() if m["group"] == "D1MT")
n_untx = sum(1 for m in metadata.values() if m["group"] == "Untreated")
print(f"  D1MT:      {n_d1mt}")
print(f"  Untreated: {n_untx}")

# =============================================================================
# HELPER FUNCTIONS  (unchanged from cART pipeline)
# =============================================================================

def format_file_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def cleanup_directory(directory_path):
    if os.path.exists(directory_path):
        try:
            print(f"  Cleaning up: {directory_path}")
            shutil.rmtree(directory_path)
            return True
        except Exception as e:
            print(f"  WARNING: Could not remove {directory_path}: {e}")
            return False
    return True


def check_cellranger_complete(output_dir):
    """
    Validate filtered_feature_bc_matrix has all 3 required files.
    barcodes.tsv.gz must be >100 bytes; a ~33-byte file is an empty gzip
    header (zero cells called).
    """
    result = {
        "completed": False, "has_barcodes": False, "has_features": False,
        "has_matrix": False, "sizes": {}, "error": None,
    }
    filtered_dir = os.path.join(output_dir, "outs", "filtered_feature_bc_matrix")
    if not os.path.exists(filtered_dir):
        result["error"] = "filtered_feature_bc_matrix not found"
        return result

    required_files = {
        "barcodes": "barcodes.tsv.gz",
        "features": "features.tsv.gz",
        "matrix": "matrix.mtx.gz",
    }
    min_sizes = {"barcodes": 100, "features": 100, "matrix": 100}

    for key, filename in required_files.items():
        filepath = os.path.join(filtered_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > min_sizes[key]:
                result[f"has_{key}"] = True
                result["sizes"][key] = size
            else:
                result["error"] = (f"{filename} too small ({size} bytes) - "
                                   f"likely zero cells called")
        else:
            result["error"] = f"{filename} not found"

    result["completed"] = (result["has_barcodes"] and result["has_features"]
                           and result["has_matrix"])
    if result["completed"]:
        result["error"] = None
    return result


def validate_fastqs(fastq_dirs, sample_name):
    """Validate R1+R2 exist for this sample across all fastq directories."""
    validation = {
        "valid": False, "has_r1": False, "has_r2": False,
        "has_i1": False, "has_i2": False,
        "r1_total_size": 0, "r2_total_size": 0, "error": None,
    }
    for fq_dir in fastq_dirs:
        if not os.path.exists(fq_dir):
            validation["error"] = f"Directory not found: {fq_dir}"
            return validation

        r1 = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_R1_*.fastq.gz"))
        r2 = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_R2_*.fastq.gz"))
        i1 = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_I1_*.fastq.gz"))
        i2 = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_I2_*.fastq.gz"))

        if r1:
            validation["has_r1"] = True
            validation["r1_total_size"] += sum(os.path.getsize(f) for f in r1)
        if r2:
            validation["has_r2"] = True
            validation["r2_total_size"] += sum(os.path.getsize(f) for f in r2)
        if i1:
            validation["has_i1"] = True
        if i2:
            validation["has_i2"] = True

    if not validation["has_r1"]:
        validation["error"] = "No R1 files found"
    elif not validation["has_r2"]:
        validation["error"] = "No R2 files found (single-end not supported)"

    validation["valid"] = validation["has_r1"] and validation["has_r2"]
    return validation


def format_error_output(result):
    lines = []
    if result.stdout:
        lines.append("=== STDOUT (last 20 lines) ===")
        lines.extend(result.stdout.split("\n")[-20:])
    if result.stderr:
        lines.append("=== STDERR (last 20 lines) ===")
        lines.extend(result.stderr.split("\n")[-20:])
    return "\n".join(lines)


def check_if_multiplexed(fastq_dirs):
    for fq_dir in fastq_dirs:
        try:
            files = os.listdir(fq_dir)
            if any("_I1_" in f for f in files) and any("_I2_" in f for f in files):
                return True
        except Exception:
            pass
    return False


# =============================================================================
# PRE-FLIGHT VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("PRE-FLIGHT VALIDATION")
print("=" * 80)

validation_results = {}
valid_samples = []
invalid_samples = []

for animal_id, meta in sorted(metadata.items()):
    sample_name = meta["sample_name"]
    fastq_dirs = meta["fastq_dirs"]

    print(f"\n  Validating {animal_id} ({meta['group']})...")
    print(f"    Sample prefix: {sample_name}")
    print(f"    FASTQ dirs: {len(fastq_dirs)}")
    for d in fastq_dirs:
        print(f"      - {d}")

    val = validate_fastqs(fastq_dirs, sample_name)
    validation_results[animal_id] = val

    if val["valid"]:
        valid_samples.append(animal_id)
        print(f"    VALID - R1: {format_file_size(val['r1_total_size'])}, "
              f"R2: {format_file_size(val['r2_total_size'])}")
        print(f"    Index files: I1={'yes' if val['has_i1'] else 'no'}, "
              f"I2={'yes' if val['has_i2'] else 'no'}")
    else:
        invalid_samples.append(animal_id)
        print(f"    INVALID - {val['error']}")

val_file = os.path.join(BASE_DIR, "metadata", "cellranger_validation.json")
with open(val_file, "w") as f:
    json.dump(validation_results, f, indent=2, default=str)

print(f"\n{'=' * 80}")
print("VALIDATION SUMMARY")
print(f"{'=' * 80}")
print(f"Total samples:      {len(metadata)}")
print(f"Valid (paired-end): {len(valid_samples)}")
print(f"Invalid:            {len(invalid_samples)}")

if len(valid_samples) == 0:
    print("\nERROR: No valid samples found!")
    sys.exit(1)

print(f"\nProceeding with {len(valid_samples)} valid samples...")

# =============================================================================
# CELL RANGER COUNT PROCESSING
# =============================================================================
print("\n" + "=" * 80)
print("CELL RANGER COUNT PROCESSING")
print(f"Reference:   {TRANSCRIPTOME}")
print(f"Output base: {OUTPUT_BASE}")
print("=" * 80)

# 3' chemistry cascade only. auto first, then explicit 3' fallbacks.
# 5' and ARC options intentionally removed: these are 3' DK libraries and a
# bad auto-detect must not be allowed to succeed as 5'.
chemistry_options = [
    "auto",
    "threeprime",
    "SC3Pv4-polyA",
    "SC3Pv4",
    "SC3Pv3-polyA",
    "SC3Pv3",
    "SC3Pv3HT-polyA",
    "SC3Pv3HT",
    "SC3Pv2",
    "SC3Pv1",
]

successful_samples = []
failed_samples = []
already_completed = []

for animal_id in sorted(valid_samples):
    meta = metadata[animal_id]
    sample_name = meta["sample_name"]
    fastq_dirs = meta["fastq_dirs"]

    cr_id = meta["cr_id"]                      # = animal_id -> clean output dir name
    cr_output_dir = os.path.join(OUTPUT_BASE, cr_id)
    is_multiplexed = check_if_multiplexed(fastq_dirs)

    print(f"\n{'=' * 60}")
    print(f"Sample: {animal_id}")
    print(f"  Animal: {animal_id} | {meta['group']} | {meta['timepoint']}")
    print(f"  --id={cr_id}")
    print(f"  --sample={sample_name}")
    print(f"  --fastqs={','.join(fastq_dirs)}")
    print(f"  Multiplexed: {'Yes' if is_multiplexed else 'No'}")
    print(f"  Output: {cr_output_dir}")
    print(f"{'=' * 60}")

    # --- Skip if already completed ---
    check = check_cellranger_complete(cr_output_dir)
    if check["completed"]:
        print("  ALREADY COMPLETED - skipping")
        for key, size in check["sizes"].items():
            print(f"    {key}: {format_file_size(size)}")
        already_completed.append(animal_id)
        successful_samples.append(animal_id)
        continue

    # --- Clean up an incomplete previous run ---
    if os.path.exists(cr_output_dir):
        print(f"  Incomplete previous run detected: {check['error']}")
        cleanup_directory(cr_output_dir)

    fastqs_arg = ",".join(fastq_dirs)
    os.chdir(OUTPUT_BASE)  # cellranger creates its dir here

    success = False
    last_errors = {}

    for chemistry in chemistry_options:
        cleanup_directory(cr_output_dir)  # clear any partial attempt
        try:
            print(f"\n  Trying chemistry: {chemistry}")
            cmd = [
                "cellranger", "count",
                f"--id={cr_id}",
                f"--fastqs={fastqs_arg}",
                f"--sample={sample_name}",
                f"--transcriptome={TRANSCRIPTOME}",
                f"--localcores={NCPUS}",
                "--create-bam=true",
                "--expect-cells=10000",
                f"--chemistry={chemistry}",
            ]
            print(f"  Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode == 0:
                final_check = check_cellranger_complete(cr_output_dir)
                if final_check["completed"]:
                    print(f"  SUCCESS with chemistry={chemistry}")
                    for key, size in final_check["sizes"].items():
                        print(f"    {key}: {format_file_size(size)}")
                    success = True
                    successful_samples.append(animal_id)
                    break
                else:
                    print(f"  Returned 0 but outputs incomplete: {final_check['error']}")
                    last_errors[chemistry] = f"Incomplete: {final_check['error']}"
                    cleanup_directory(cr_output_dir)
            else:
                error_out = format_error_output(result)
                last_errors[chemistry] = error_out
                first_line = error_out.split("\n")[0] if error_out else "Unknown"
                print(f"  Failed: {first_line}")
                cleanup_directory(cr_output_dir)

        except Exception as e:
            last_errors[chemistry] = traceback.format_exc()
            print(f"  Exception: {e}")
            cleanup_directory(cr_output_dir)

    if not success:
        print(f"\n  ALL CHEMISTRIES FAILED for {animal_id}")
        for chem, err in list(last_errors.items())[-3:]:
            preview = err.splitlines()[0] if err else "Unknown"
            print(f"    {chem}: {preview}")
        failed_samples.append({
            "animal_id": animal_id,
            "group": meta["group"],
            "last_errors": {k: v[:500] for k, v in last_errors.items()},
        })

# =============================================================================
# SAVE PROCESSING SUMMARY + DOWNSTREAM OUTPUT MAP
# =============================================================================
summary = {
    "reference": TRANSCRIPTOME,
    "output_base": OUTPUT_BASE,
    "validation": {
        "total": len(metadata),
        "valid": len(valid_samples),
        "invalid": len(invalid_samples),
        "invalid_samples": invalid_samples,
    },
    "processing": {
        "successful": len(successful_samples),
        "already_completed": len(already_completed),
        "newly_processed": len(successful_samples) - len(already_completed),
        "failed": len(failed_samples),
    },
    "successful_samples": successful_samples,
    "already_completed_samples": already_completed,
    "failed_details": [
        {"animal_id": f["animal_id"], "group": f["group"],
         "chemistries_tried": list(f["last_errors"].keys())}
        for f in failed_samples
    ],
}

summary_file = os.path.join(BASE_DIR, "metadata", "cellranger_processing_summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

# sample_id -> output path map for Phase 1 (keyed by animal_id)
output_map = {}
for animal_id in successful_samples:
    meta = metadata[animal_id]
    cr_dir = os.path.join(OUTPUT_BASE, meta["cr_id"])
    output_map[animal_id] = {
        "cellranger_dir": cr_dir,
        "matrix_dir": os.path.join(cr_dir, "outs", "filtered_feature_bc_matrix"),
        "animal_id": animal_id,
        "group": meta["group"],
        "sex": meta.get("sex"),
        "source": meta.get("source"),
        "cohort": meta.get("cohort"),
        "cryo_status": meta.get("cryo_status"),
        "timepoint": meta.get("timepoint"),
    }

map_file = os.path.join(BASE_DIR, "metadata", "sample_output_map.json")
with open(map_file, "w") as f:
    json.dump(output_map, f, indent=2)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
print(f"\nValidation:")
print(f"  Total samples:   {len(metadata)}")
print(f"  Valid:           {len(valid_samples)}")
print(f"  Invalid:         {len(invalid_samples)}")
print(f"\nCell Ranger Count:")
print(f"  Already done:    {len(already_completed)}")
print(f"  Newly processed: {len(successful_samples) - len(already_completed)}")
print(f"  Total successful:{len(successful_samples)}")
print(f"  Failed:          {len(failed_samples)}")

if failed_samples:
    print(f"\nFailed samples:")
    for f in failed_samples:
        print(f"  - {f['animal_id']} ({f['group']})")

print(f"\nOutputs saved to:")
print(f"  Count matrices: {OUTPUT_BASE}/")
print(f"  Summary:        {summary_file}")
print(f"  Output map:     {map_file}")
print(f"  Validation:     {val_file}")

print("\n" + "=" * 80)
if len(failed_samples) == 0 and len(valid_samples) > 0:
    print("All valid samples processed successfully!")
elif len(failed_samples) > 0:
    print(f"{len(failed_samples)} sample(s) failed - check summary for details")
print("=" * 80)

sys.exit(0)
