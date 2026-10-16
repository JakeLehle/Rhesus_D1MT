#!/bin/bash
#SBATCH -J Rhesus_D1MT
#SBATCH -o /master/jlehle/WORKING/LOGS/Rhesus_D1MT_%j.out
#SBATCH -e /master/jlehle/WORKING/LOGS/Rhesus_D1MT_%j.err
#SBATCH -t 7-00:00:00
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 50
#SBATCH --mem 900GB


set -euo pipefail

# ============================================================
# Rhesus D1MT Monotherapy scRNA-seq — FULL Pipeline
# ============================================================
# One-shot: metadata -> cellranger count -> Phase 1 -> 2 -> 3 -> 4.
# Focal comparison: D1MT vs Untreated (BAL, week 15, no ART, no SIV).
#
# Resource sizing is set for the CELLRANGER step (BAMs + multi-contig
# reference => 800GB, per the cART OOM note; cores held at 32 for the same
# reason). Once cellranger has completed, you can comment out STEP 1 below
# and drop --mem back to ~256GB / -t 1-00:00:00 for analysis-only reruns.
#
# Comment out any completed step to resume from a given point.
# ============================================================

# --- Environment ---
source $HOME/anaconda3/bin/activate
conda activate sc_pre
export PATH="$HOME/cellranger-10.0.0/bin:${PATH}"   # cellranger 10.0.0 on PATH
export PATH="${CONDA_PREFIX}/bin:${PATH}"           # conda bin FIRST: conda samtools wins over
                                                    # broken /usr/local/bin/samtools (v0.1.18)

WORKDIR="/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT"
SCRIPTS="${WORKDIR}/scripts"
cd "${WORKDIR}"

echo "============================================================"
echo "RHESUS D1MT FULL PIPELINE — $(date)"
echo "Working dir: ${WORKDIR}"
echo "Conda env:   ${CONDA_DEFAULT_ENV}"
echo "cellranger:  $(which cellranger)"
echo "samtools:    $(which samtools)"
echo "============================================================"

# ============================================================
# STEP 0: Metadata (single source of truth; read by cellranger AND Phase 1)
# ============================================================
echo ""
echo "=== STEP 0: Build metadata — $(date) ==="
python "${SCRIPTS}/Build_D1MT_Metadata.py"

# ============================================================
# STEP 1: CellRanger count (combined Mmul10+Mtb+SIV ref, BAMs on)
#   -> cellranger_counts/<animal_id>/outs/filtered_feature_bc_matrix
#   Resumable: skips samples whose outputs already exist.
#   >>> Comment out this block once all 8 libraries are counted. <<<
# ============================================================
echo ""
echo "=== STEP 1: CellRanger count — $(date) ==="
python "${SCRIPTS}/cellranger_count_all.py"

# Informational: how many libraries produced a valid cell matrix
n_done=$(ls cellranger_counts/*/outs/filtered_feature_bc_matrix/barcodes.tsv.gz 2>/dev/null | wc -l)
echo "CellRanger: ${n_done} library matrices present (expected 8)."

# ============================================================
# PHASE 1: Load, QC, Normalize, Embed
#   -> analysis/adata_d1mt_qcd.h5ad, analysis/phase1_qc/
# ============================================================
echo ""
echo "=== PHASE 1: QC + Embed — $(date) ==="
python "${SCRIPTS}/Phase1_scanpy_qc.py"

# ============================================================
# PHASE 2: Tiered Annotation + Proportion Comparison
#   -> analysis/adata_d1mt_annotated.h5ad, phase2_annotation/, phase2_proportions/
# ============================================================
echo ""
echo "=== PHASE 2: Annotation — $(date) ==="
python "${SCRIPTS}/Phase2_annotation.py"

# ============================================================
# PHASE 3: DEG (D1MT vs Untreated per cluster) + per-animal consistency
#   -> analysis/phase3_deg/
# ============================================================
echo ""
echo "=== PHASE 3: DEG — $(date) ==="
python "${SCRIPTS}/Phase3_DEG.py"

# ============================================================
# PHASE 4: LIANA cell-cell communication (T cell <-> macrophage)
#   -> analysis/phase4_liana/
# ============================================================
echo ""
echo "=== PHASE 4: LIANA — $(date) ==="
python "${SCRIPTS}/Phase4_LIANA.py"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo "Outputs under ${WORKDIR}/:"
echo "  cellranger_counts/<animal>/        (Step 1)"
echo "  analysis/adata_d1mt_qcd.h5ad        (Phase 1)"
echo "  analysis/adata_d1mt_annotated.h5ad  (Phase 2)"
echo "  analysis/phase1_qc/ phase2_annotation/ phase2_proportions/"
echo "  analysis/phase3_deg/ phase4_liana/"
echo "============================================================"

