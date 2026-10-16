#!/usr/bin/env python3
"""
Phase 2 — Tiered Annotation, Scoring, and Diagnostics  (Rhesus D1MT Project)
============================================================================
Takes the QC'd, embedded AnnData from Phase 1 and performs:
  1. Hierarchical cell type annotation (Tier 1 -> 2 -> 3)
  2. Biological gene set scoring (IDO, exhaustion, Th1/Th17, macrophage, etc.)
  3. Diagnostic UMAPs and dotplots
  4. Single-timepoint (week 15) per-group proportion comparison

Groups (from Phase 1):
  - D1MT      : IDO inhibitor (1-methyl-D-tryptophan) monotherapy, no ART
  - Untreated : no treatment (3 of 4 are historical Sharan JCI 2022 controls)

Single timepoint (week 15). NO time series: the cART project's trajectory and
fold-change-from-W5 analysis is removed and replaced with a D1MT-vs-Untreated
proportion comparison (per-animal points, mean +/- SEM, Mann-Whitney flag).

ANNOTATION SCHEME NOTE:
  Tier 1 T-cell and NK-cell marker sets were REFINED for this dataset (see
  MARKER REFINEMENT below) after a read-only diagnostic pass on the real D1MT
  clusters. The remaining Tier 1/2/3 sets are still carried over from the cART
  project and do not yet one-to-one match the Singh B draft scheme (IDO+
  inflammatory vs Mac-IFN split; proliferating / gamma-delta / IFN-responding
  T; T-doublet bin). Further reconciliation, including the Tier 2 CD4-vs-CD8
  fix flagged below, is a DELIBERATE separate pass.

MARKER REFINEMENT (July 2026, two-sample D1MT diagnostic):
  Tier 1 T cells: was [CD3D, CD3E]; now [CD3D, CD3E, CD3G, CD247].
    CD3D/CD3E drop out independently in low-RNA BAL lymphocytes, sending real
    T cells to Unassigned. Adding CD3G + CD247 (CD3-zeta) unions their coverage
    and rescued ~54% of previously-unassigned lymphoid cells; all four are ~0%
    in macrophages (no myeloid leakage).
  Tier 1 NK cells: was [KLRC3, KLRD1, GNLY, PRF1, GZMB]; now
    [KLRC3, KLRD1, NCAM1, NCR1]. GNLY/PRF1/GZMB are shared cytotoxic genes that
    fired on CD8 effector T cells (mis-calling them NK). Purified anchors give a
    ~19x score separation (NK island 0.63 vs T island 0.03) and flipped ~57% of
    the mis-assigned NK-in-T cells back to T.
  KNOWN NEXT FIX (not in this version): the Tier 2 CD4 set leads with CD4,
    which is a macrophage gene in rhesus (0.45 in macrophages vs 0.08 in T cells
    here), and also carries pan-T genes (BCL11B, LTB) that inflate CD4 over CD8.
    The CD4-vs-CD8 split needs its own refinement pass.

  Genuinely low-RNA ambiguous cells (median ~819 genes, top-score 0) are left
  Unassigned on purpose; they lack the coverage to place confidently.

New columns added to adata.obs:
  leiden (diagnostic clustering; NOT used for assignment)
  tier1_celltype, tier1_score
  tier2_celltype, tier2_score, tier2_full
  tier3_state, tier3_naive_score
  score_Th1, score_Th17, exhaustion_score, IDO_pathway_score,
  SIV_host_factor_score (computed, not featured), granuloma_trafficking_score,
  AM_like_score, IFN_responsive_score, M1_IM_like_score

Run in Spyder (# %% cells) or standalone.
Output dirs: analysis/phase2_annotation/, analysis/phase2_proportions/

Author: Jake Lehle / Kaushal Lab
Date: July 2026
"""

# %% Cell 1 — Configuration and Load Phase 1 Output
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless for SLURM; comment out for interactive Spyder
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import mannwhitneyu
import shutil
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === ADJUSTABLE PARAMETERS ===
SCORE_THRESHOLD = 0.1       # Minimum score to assign a cell type (Tier 1 and 2)
NAIVE_THRESHOLD = 0.1       # TCF7/LEF1 score above this = Naive T cell

# === PATHS ===
BASE_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT'
ADATA_IN = os.path.join(BASE_DIR, 'analysis', 'adata_d1mt_qcd.h5ad')
ADATA_OUT = os.path.join(BASE_DIR, 'analysis', 'adata_d1mt_annotated.h5ad')
OUT_DIR = os.path.join(BASE_DIR, 'analysis', 'phase2_annotation')
PROP_DIR = os.path.join(BASE_DIR, 'analysis', 'phase2_proportions')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PROP_DIR, exist_ok=True)

# === GROUP CONFIG ===
GROUP_ORDER = ['Untreated', 'D1MT']
GROUP_CONFIG = {
    'Untreated': {'color': '#1976D2', 'short': 'Untreated'},
    'D1MT':      {'color': '#D32F2F', 'short': 'D1MT'},
}
GROUP_PALETTE = {k: v['color'] for k, v in GROUP_CONFIG.items()}
GROUP_COL = 'group'
ANIMAL_COL = 'animal_id'
SAMPLE_COL = 'sample_id'

# === LOAD ===
print("Loading Phase 1 output...")
adata = sc.read_h5ad(ADATA_IN)
adata.obs[GROUP_COL] = pd.Categorical(adata.obs[GROUP_COL], categories=GROUP_ORDER)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  Groups:  {sorted(adata.obs[GROUP_COL].unique().tolist())}")
print(f"  Animals: {sorted(adata.obs[ANIMAL_COL].unique())}")

# Discrete Leiden clustering saved for diagnostics / figures ONLY.
# Assignment stays per-cell score-based; clusters never label cells.
# res + seed match the marker-refinement diagnostic so saved cluster numbering
# aligns with the 12/16/17 discussed there.
if 'neighbors' in adata.uns:
    try:
        sc.tl.leiden(adata, resolution=1.0, key_added='leiden', random_state=0,
                     flavor='igraph', n_iterations=2, directed=False)
    except Exception as e:
        print(f"  [warn] igraph leiden failed ({e}); default flavor")
        sc.tl.leiden(adata, resolution=1.0, key_added='leiden', random_state=0)
    print(f"  Saved Leiden clustering: {adata.obs['leiden'].nunique()} clusters")
else:
    print("  [warn] no neighbor graph in .uns; skipping Leiden save")


# %% Cell 2 — Tier 1: Broad Lineage Scoring and Assignment
print("\n" + "=" * 80)
print("TIER 1: Broad Lineage Assignment (8 populations)")
print("=" * 80)

# NOTE: T-cell and NK-cell sets refined for this dataset (see header
# MARKER REFINEMENT). T uses the full CD3 complex to survive dropout; NK uses
# exclusive anchors only so shared cytotoxic genes stop mis-calling CD8 T cells.
TIER1_MARKERS = {
    'T cells':             {'markers': ['CD3D', 'CD3E', 'CD3G', 'CD247']},
    'Macrophages':         {'markers': ['CD163', 'MRC1', 'MARCO', 'MERTK', 'PPARG']},
    'Monocytes':           {'markers': ['S100A8', 'S100A9', 'VCAN', 'CAMP', 'CD14', 'LYZ']},
    'DCs':                 {'markers': ['FLT3', 'CLEC9A', 'CLEC10A', 'IL3RA', 'CLEC4C']},
    'NK cells':            {'markers': ['KLRC3', 'KLRD1', 'NCAM1', 'NCR1']},
    'B cells':             {'markers': ['CD79A', 'MS4A1', 'CD19', 'BANK1']},
    'Mast cells':          {'markers': ['CPA3', 'KIT', 'HPGD', 'ENPP3', 'FCER1A']},
    'Ciliated epithelial': {'markers': ['DNAH9', 'HYDIN', 'MUC16', 'TSPAN1']},
}

tier1_score_cols = {}
for lineage, info in TIER1_MARKERS.items():
    present = [g for g in info['markers'] if g in adata.var_names]
    score_key = f"tier1_score_{lineage.replace(' ', '_')}"
    if len(present) >= 2:
        sc.tl.score_genes(adata, gene_list=present, score_name=score_key)
        tier1_score_cols[lineage] = score_key
        print(f"  {lineage}: {len(present)}/{len(info['markers'])} markers, "
              f"mean={adata.obs[score_key].mean():.3f}, max={adata.obs[score_key].max():.3f}")
    else:
        print(f"  SKIP {lineage}: only {len(present)} markers present")

lineage_names = list(tier1_score_cols.keys())
lineage_cols = [tier1_score_cols[n] for n in lineage_names]
tier1_matrix = adata.obs[lineage_cols].values
best_idx = np.argmax(tier1_matrix, axis=1)
best_score = np.max(tier1_matrix, axis=1)
tier1_labels = np.where(best_score >= SCORE_THRESHOLD,
                        [lineage_names[i] for i in best_idx], 'Unassigned')
adata.obs['tier1_celltype'] = pd.Categorical(tier1_labels)
adata.obs['tier1_score'] = best_score

print(f"\n  Tier 1 assignment:")
print(adata.obs['tier1_celltype'].value_counts().to_string())


# %% Cell 3 — Tier 2: Subtype Within Lineage
print("\n" + "=" * 80)
print("TIER 2: Subtype Assignment Within Each Lineage")
print("=" * 80)

# NOTE: CD4 set still leads with CD4 (a macrophage gene in rhesus) and carries
# pan-T genes (BCL11B, LTB). This biases CD4 over CD8 and is the flagged NEXT
# refinement pass (see header). Left unchanged in this version on purpose.
TIER2_MARKERS = {
    'T cells': {
        'CD4': ['CD4', 'BCL11B', 'LTB', 'IL7R'],
        'CD8': ['CD8A', 'CD8B'],
    },
    'Macrophages': {
        'AM-like':        ['PPARG', 'COLEC12', 'MARCO', 'MERTK', 'MRC1', 'CD163'],
        'TREM2+':         ['TREM2', 'C1QA', 'C1QB', 'C1QC', 'MAMU-DRA', 'CD74'],
        'M2 IM-like':     ['CD163', 'MAMU-DRA', 'MAMU-DRB1'],
        'IFN-responsive': ['IDO1', 'CXCL9', 'CXCL10', 'CXCL11', 'CD274', 'SOD2'],
        'M1 IM-like':     ['STAT1', 'IRF1', 'TNF', 'IL1B', 'CCL2', 'CCL3'],
    },
    'Monocytes': {
        'Classical':     ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'CAMP'],
        'Non-classical': ['ITGAX', 'MAMU-DRA'],
    },
    'DCs': {
        'cDC1': ['CLEC9A', 'XCR1', 'BATF3', 'ZBTB46'],
        'cDC2': ['FLT3', 'CD1C', 'CLEC10A', 'CPVL'],
        'pDC':  ['IL3RA', 'CLEC4C', 'TCF4', 'BCL2', 'TLR7', 'PACSIN1', 'COBLL1'],
    },
}

tier2_score_cols = {}
for lineage, subtypes in TIER2_MARKERS.items():
    tier2_score_cols[lineage] = {}
    print(f"\n  --- {lineage} subtypes ---")
    for subtype, markers in subtypes.items():
        present = [g for g in markers if g in adata.var_names]
        score_key = (f"tier2_score_{lineage.replace(' ', '_')}_"
                     f"{subtype.replace(' ', '_').replace('-', '_').replace('+', 'pos')}")
        if len(present) >= 2:
            sc.tl.score_genes(adata, gene_list=present, score_name=score_key)
            tier2_score_cols[lineage][subtype] = score_key
            print(f"    {subtype}: {len(present)} markers scored")
        elif len(present) == 1:
            gene = present[0]
            expr = adata[:, gene].X
            expr = expr.toarray().flatten() if sparse.issparse(expr) else np.array(expr).flatten()
            adata.obs[score_key] = expr
            tier2_score_cols[lineage][subtype] = score_key
            print(f"    {subtype}: 1 marker ({gene}), using direct expression")
        else:
            print(f"    SKIP {subtype}: no markers present")

tier2_labels, tier2_scores = [], []
for i in range(adata.n_obs):
    t1 = adata.obs['tier1_celltype'].iloc[i]
    if t1 in TIER2_MARKERS and t1 in tier2_score_cols and tier2_score_cols[t1]:
        subtypes = tier2_score_cols[t1]
        subtype_names = list(subtypes.keys())
        subtype_vals = [adata.obs[subtypes[s]].iloc[i] for s in subtype_names]
        best = np.argmax(subtype_vals)
        tier2_labels.append(subtype_names[best])
        tier2_scores.append(subtype_vals[best])
    elif t1 == 'Unassigned':
        tier2_labels.append('Unassigned')
        tier2_scores.append(0.0)
    else:
        tier2_labels.append(t1)
        tier2_scores.append(adata.obs['tier1_score'].iloc[i])

adata.obs['tier2_celltype'] = pd.Categorical(tier2_labels)
adata.obs['tier2_score'] = tier2_scores
adata.obs['tier2_full'] = [
    f"{t1}: {t2}" if t1 != t2 and t1 != 'Unassigned' else t1
    for t1, t2 in zip(adata.obs['tier1_celltype'], adata.obs['tier2_celltype'])
]

print(f"\n  Tier 2 assignment:")
print(adata.obs['tier2_full'].value_counts().to_string())


# %% Cell 4 — Biological Gene Set Scoring
print("\n" + "=" * 80)
print("BIOLOGICAL GENE SET SCORING")
print("=" * 80)

def score_gene_set(adata_obj, gene_list, score_name):
    present = [g for g in gene_list if g in adata_obj.var_names]
    if len(present) < 2:
        print(f"  SKIP {score_name}: only {len(present)}/{len(gene_list)} genes present")
        adata_obj.obs[score_name] = 0.0
        return 0
    sc.tl.score_genes(adata_obj, gene_list=present, score_name=score_name)
    print(f"  {score_name}: scored with {len(present)}/{len(gene_list)} genes")
    return len(present)

score_gene_set(adata, ['TBX21', 'TNF', 'BHLHE40', 'IFNG'], 'score_Th1')
score_gene_set(adata, ['RORC', 'STAT3', 'RORA'], 'score_Th17')
score_gene_set(adata, ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'CTLA4',
                       'TOX', 'ENTPD1', 'CD160', 'CD244'], 'exhaustion_score')
score_gene_set(adata, ['IDO1', 'IDO2', 'TDO2', 'KMO', 'KYNU', 'HAAO',
                       'IFNG', 'STAT1', 'IRF1', 'CD274', 'CD38', 'AHR'], 'IDO_pathway_score')
# SIV host factors: host genes, computed for reversibility but NOT featured
# (no SIV story in this paper).
score_gene_set(adata, ['CD4', 'CCR5', 'CXCR4', 'APOBEC3G', 'APOBEC3F',
                       'TRIM5', 'BST2', 'SAMHD1', 'IFITM1', 'IFITM2', 'IFITM3', 'MX2'],
               'SIV_host_factor_score')
score_gene_set(adata, ['CCR7', 'CCL19', 'CCL21', 'SELL', 'CXCL13', 'LTB', 'CD27'],
               'granuloma_trafficking_score')
score_gene_set(adata, ['MRC1', 'CD163', 'MARCO', 'MERTK', 'TREM2',
                       'C1QA', 'C1QB', 'C1QC'], 'AM_like_score')
score_gene_set(adata, ['CXCL9', 'CXCL10', 'CXCL11', 'IDO1', 'CD274',
                       'GBP1', 'GBP2', 'GBP5', 'STAT1', 'IRF1', 'IRF7'], 'IFN_responsive_score')
score_gene_set(adata, ['STAT1', 'IRF1', 'TNF', 'IL1B', 'CCL2', 'CCL3'], 'M1_IM_like_score')


# %% Cell 5 — Tier 3: T Cell Functional States
print("\n" + "=" * 80)
print("TIER 3: T Cell Functional States")
print("=" * 80)

NAIVE_MARKERS = ['TCF7', 'LEF1']
naive_present = [g for g in NAIVE_MARKERS if g in adata.var_names]
if len(naive_present) >= 2:
    sc.tl.score_genes(adata, gene_list=naive_present, score_name='tier3_naive_score')
    print(f"  Naive score computed with {len(naive_present)} markers")
elif len(naive_present) == 1:
    gene = naive_present[0]
    expr = adata[:, gene].X
    expr = expr.toarray().flatten() if sparse.issparse(expr) else np.array(expr).flatten()
    adata.obs['tier3_naive_score'] = expr
    print(f"  Naive score from single marker: {gene}")
else:
    adata.obs['tier3_naive_score'] = 0.0
    print("  WARNING: No naive markers found")

t_cell_mask = adata.obs['tier1_celltype'] == 'T cells'
print(f"  Total T cells: {t_cell_mask.sum():,}")

tier3_labels = pd.Series('N/A', index=adata.obs.index)
for i in adata.obs.index[t_cell_mask]:
    t2 = adata.obs.loc[i, 'tier2_celltype']
    naive_score = adata.obs.loc[i, 'tier3_naive_score']
    tier3_labels[i] = f'{t2} Naive' if naive_score >= NAIVE_THRESHOLD else f'{t2} Effector'
adata.obs['tier3_state'] = pd.Categorical(tier3_labels)

print(f"\n  Tier 3 T cell states:")
print(adata.obs.loc[t_cell_mask, 'tier3_state'].value_counts().to_string())

# Featured continuous scores (SIV host factors intentionally excluded)
continuous_scores = {
    'Th1':                   'score_Th1',
    'Th17':                  'score_Th17',
    'Exhaustion':            'exhaustion_score',
    'Granuloma trafficking': 'granuloma_trafficking_score',
    'IDO pathway':           'IDO_pathway_score',
}

cd4_eff_mask = adata.obs['tier3_state'] == 'CD4 Effector'
print(f"\n  CD4 Effector cells: {cd4_eff_mask.sum():,}")
if cd4_eff_mask.sum() > 0:
    print(f"\n  Continuous scores within CD4 Effector:")
    for label, col in continuous_scores.items():
        if col in adata.obs.columns:
            vals = adata.obs.loc[cd4_eff_mask, col]
            print(f"    {label}: mean={vals.mean():.3f}, std={vals.std():.3f}")


# %% Cell 6 — UMAPs
print("\n" + "=" * 80)
print("UMAPS — Tiered Cell Type Annotation")
print("=" * 80)

# --- 6A: Tier 1 overview (tier1 / group / cohort) ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
sc.pl.umap(adata, color='tier1_celltype', ax=axes[0], show=False,
           title='Tier 1: Broad Lineages', frameon=False, legend_loc='right margin')
sc.pl.umap(adata, color=GROUP_COL, ax=axes[1], show=False,
           title='Group', frameon=False, legend_loc='right margin',
           palette=[GROUP_PALETTE[g] for g in GROUP_ORDER])
sc.pl.umap(adata, color='cohort', ax=axes[2], show=False,
           title='Cohort (current vs historical)', frameon=False, legend_loc='right margin')
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(OUT_DIR, 'umap_tier1_overview.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(OUT_DIR, 'umap_tier1_overview.png'), bbox_inches='tight', dpi=300)
print("  Saved: umap_tier1_overview")
plt.close(fig)

# --- 6B: Tier 2 breakout UMAPs per lineage ---
breakout_lineages = {
    'T cells':     {'extra': 'tier3_state'},
    'Macrophages': {'extra': None},
    'Monocytes':   {'extra': None},
    'DCs':         {'extra': None},
}
for lineage, config in breakout_lineages.items():
    mask = adata.obs['tier1_celltype'] == lineage
    n_cells = int(mask.sum())
    if n_cells < 100:
        print(f"  Skip {lineage}: only {n_cells} cells")
        continue
    sub = adata[mask].copy()
    safe_name = lineage.replace(' ', '_').lower()

    panels = ['tier2_celltype']
    panel_titles = [f'{lineage}: Tier 2 Subtypes']
    if config['extra']:
        panels.append(config['extra'])
        panel_titles.append(f'{lineage}: Tier 3 States')
    panels.append(GROUP_COL)
    panel_titles.append(f'{lineage}: Group')

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]
    for i, (panel_col, panel_title) in enumerate(zip(panels, panel_titles)):
        kw = {}
        if panel_col == GROUP_COL:
            kw['palette'] = [GROUP_PALETTE[g] for g in GROUP_ORDER]
        sc.pl.umap(sub, color=panel_col, ax=axes[i], show=False,
                   title=panel_title, frameon=False, legend_loc='right margin', **kw)
    plt.suptitle(f'{lineage} Breakout ({n_cells:,} cells)', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, f'umap_breakout_{safe_name}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f'umap_breakout_{safe_name}.png'), bbox_inches='tight', dpi=300)
    print(f"  Saved: umap_breakout_{safe_name}")
    plt.close(fig)

# --- 6C: Full Tier 2 UMAP ---
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(adata, color='tier2_full', ax=ax, show=False,
           title='All Tier 2 Subtypes', frameon=False,
           legend_loc='right margin', legend_fontsize=7)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(OUT_DIR, 'umap_tier2_full.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(OUT_DIR, 'umap_tier2_full.png'), bbox_inches='tight', dpi=300)
print("  Saved: umap_tier2_full")
plt.close(fig)

# --- 6D: Biological score UMAPs (SIV score excluded) ---
sig_cols = ['IDO_pathway_score', 'exhaustion_score', 'AM_like_score',
            'IFN_responsive_score', 'M1_IM_like_score', 'granuloma_trafficking_score']
sig_present = [s for s in sig_cols if s in adata.obs.columns]
if sig_present:
    ncols = min(3, len(sig_present))
    nrows = int(np.ceil(len(sig_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()
    for i, sig in enumerate(sig_present):
        sc.pl.umap(adata, color=sig, ax=axes[i], show=False,
                   title=sig.replace('_', ' '), cmap='RdYlBu_r', frameon=False)
    for i in range(len(sig_present), len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, 'umap_biological_scores.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'umap_biological_scores.png'), bbox_inches='tight', dpi=300)
    print("  Saved: umap_biological_scores")
    plt.close(fig)


# %% Cell 7 — Dotplots: Marker Gene Specificity
print("\n" + "=" * 80)
print("DOTPLOTS — Marker Gene Specificity")
print("=" * 80)

# Dotplot marker panels mirror the refined Tier 1 T / NK sets so the QC
# dotplots show the genes actually driving assignment.
TIER1_DOTPLOT_MARKERS = {
    'T cells':             ['CD3D', 'CD3E', 'CD3G', 'CD247'],
    'Macrophages':         ['CD163', 'MRC1', 'MARCO', 'MERTK', 'PPARG'],
    'Monocytes':           ['S100A8', 'S100A9', 'VCAN', 'CD14', 'LYZ'],
    'DCs':                 ['FLT3', 'CLEC9A', 'CLEC10A', 'IL3RA'],
    'NK cells':            ['KLRC3', 'KLRD1', 'NCAM1', 'NCR1'],
    'B cells':             ['CD79A', 'MS4A1', 'CD19', 'BANK1'],
    'Mast cells':          ['CPA3', 'KIT', 'HPGD', 'ENPP3'],
    'Ciliated epithelial': ['DNAH9', 'HYDIN', 'MUC16', 'TSPAN1'],
}
TIER2_DOTPLOT_MARKERS = {
    'T cells: CD4':                ['CD4', 'BCL11B', 'LTB', 'IL7R'],
    'T cells: CD8':                ['CD8A', 'CD8B'],
    'Macrophages: AM-like':        ['PPARG', 'COLEC12', 'MARCO', 'MERTK'],
    'Macrophages: TREM2+':         ['TREM2', 'C1QA', 'C1QB', 'CD74'],
    'Macrophages: M2 IM-like':     ['CD163', 'MAMU-DRA', 'MAMU-DRB1'],
    'Macrophages: IFN-responsive': ['IDO1', 'CXCL9', 'CXCL10', 'CD274'],
    'Macrophages: M1 IM-like':     ['STAT1', 'IRF1', 'TNF', 'IL1B', 'CCL2', 'CCL3'],
    'Monocytes: Classical':        ['CD14', 'S100A8', 'S100A9', 'VCAN'],
    'Monocytes: Non-classical':    ['ITGAX', 'MAMU-DRA'],
    'DCs: cDC1':                   ['CLEC9A', 'XCR1', 'BATF3'],
    'DCs: cDC2':                   ['CD1C', 'CLEC10A', 'CPVL'],
    'DCs: pDC':                    ['IL3RA', 'CLEC4C', 'TCF4', 'PACSIN1'],
    'NK cells':                    ['KLRC3', 'KLRD1', 'NCAM1', 'NCR1'],
    'B cells':                     ['CD79A', 'MS4A1', 'CD19', 'BANK1'],
    'Mast cells':                  ['CPA3', 'KIT', 'HPGD', 'ENPP3'],
    'Ciliated epithelial':         ['DNAH9', 'HYDIN', 'MUC16', 'TSPAN1'],
}

def save_dotplot(adata_sub, gene_groups, groupby, order, stem):
    """Render a scanpy dotplot and save PNG + PDF directly to OUT_DIR.
    Bypasses scanpy's figdir + 'dotplot_' prefix: passing save='_..' produced a
    double-underscore filename (dotplot__stem) that the old copy step looked for
    under a single underscore and silently missed, so OUT_DIR never refreshed.
    Writing the returned figure straight to OUT_DIR is robust and version-proof.
    Category order is filtered to what is actually present so an empty subtype
    can't trip the plot."""
    present_order = [c for c in order if c in list(adata_sub.obs[groupby].unique())]
    axd = sc.pl.dotplot(adata_sub, var_names=gene_groups, groupby=groupby,
                        categories_order=present_order, standard_scale='var', show=False)
    fig = next(iter(axd.values())).get_figure()
    fig.savefig(os.path.join(OUT_DIR, f'{stem}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{stem}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {stem} -> {OUT_DIR}")

print("\n  Generating Tier 1 marker dotplot...")
t1_order = list(TIER1_DOTPLOT_MARKERS.keys())
adata_t1 = adata[adata.obs['tier1_celltype'].isin(t1_order)].copy()
t1_gene_groups = {ct: [g for g in m if g in adata_t1.var_names]
                  for ct, m in TIER1_DOTPLOT_MARKERS.items()
                  if any(g in adata_t1.var_names for g in m)}
save_dotplot(adata_t1, t1_gene_groups, 'tier1_celltype', t1_order, 'dotplot_tier1_markers')

print("\n  Generating Tier 2 marker dotplot...")
has_tier2 = {'T cells', 'Macrophages', 'Monocytes', 'DCs'}
adata.obs['dotplot_group'] = adata.obs.apply(
    lambda row: row['tier2_full'] if row['tier1_celltype'] in has_tier2
    else row['tier1_celltype'], axis=1)
t2_order = list(TIER2_DOTPLOT_MARKERS.keys())
t2_mask = adata.obs['dotplot_group'].isin(t2_order)
adata_t2 = adata[t2_mask].copy()
print(f"  Cells included: {int(t2_mask.sum()):,} (excluded {int((~t2_mask).sum()):,})")
t2_gene_groups = {ct: [g for g in m if g in adata_t2.var_names]
                  for ct, m in TIER2_DOTPLOT_MARKERS.items()
                  if any(g in adata_t2.var_names for g in m)}
save_dotplot(adata_t2, t2_gene_groups, 'dotplot_group', t2_order, 'dotplot_tier2_markers')


# %% Cell 8 — Diagnostic Tables (by group; per-animal)
print("\n" + "=" * 80)
print("DIAGNOSTIC TABLES")
print("=" * 80)

print("\n--- Tier 1 cell type proportions (%) by group ---")
t1_props = pd.crosstab(adata.obs[GROUP_COL], adata.obs['tier1_celltype'], normalize='index') * 100
print(t1_props.round(1).to_string())
t1_props.to_csv(os.path.join(OUT_DIR, 'tier1_proportions.csv'))

print("\n\n--- Tier 2 cell type proportions (%) by group ---")
t2_props = pd.crosstab(adata.obs[GROUP_COL], adata.obs['tier2_full'], normalize='index') * 100
print(t2_props.round(1).to_string())
t2_props.to_csv(os.path.join(OUT_DIR, 'tier2_proportions.csv'))

print("\n\n--- CD4 Effector continuous scores (mean +/- std) by group ---")
cd4_eff_obs = adata.obs.loc[adata.obs['tier3_state'] == 'CD4 Effector']
score_cols_present = [c for c in continuous_scores.values() if c in cd4_eff_obs.columns]
if score_cols_present and len(cd4_eff_obs) > 0:
    cd4_scores = cd4_eff_obs.groupby(GROUP_COL, observed=True)[score_cols_present].agg(['mean', 'std', 'count']).round(3)
    print(cd4_scores.to_string())
    cd4_scores.to_csv(os.path.join(OUT_DIR, 'cd4_effector_scores.csv'))

print("\n\n--- Annotation sanity check ---")
n_unassigned = int((adata.obs['tier1_celltype'] == 'Unassigned').sum())
print(f"  Unassigned cells: {n_unassigned:,} ({n_unassigned/adata.n_obs*100:.1f}%)")
for lineage in TIER1_MARKERS:
    n = int((adata.obs['tier1_celltype'] == lineage).sum())
    pct = n / adata.n_obs * 100
    flag = ""
    if pct < 0.1 and lineage not in ['Ciliated epithelial', 'Mast cells']:
        flag = " *** LOW"
    elif pct > 60:
        flag = " *** DOMINANT"
    print(f"  {lineage}: {n:>8,} ({pct:5.1f}%){flag}")
print(f"\n  Diagnostic tables saved to {OUT_DIR}")


# %% Cell 9 — Single-Timepoint Proportion Comparison (D1MT vs Untreated)
# =========================================================================
# Replaces the cART project's time-series / fold-change section. One week-15
# timepoint: per-animal cluster frequencies, group mean +/- SEM, Mann-Whitney
# flag. n=4/arm is underpowered for proportion stats — the per-animal dots are
# the honest story; p-values are exploratory flags only.
# =========================================================================
print("\n" + "=" * 80)
print("PROPORTION COMPARISON — D1MT vs Untreated (week 15)")
print("=" * 80)


def per_animal_proportions(adata_obj, celltype_col):
    """Per-animal proportion of each cell type, zero-filled across all types."""
    counts = (adata_obj.obs
              .groupby([GROUP_COL, ANIMAL_COL, celltype_col], observed=True)
              .size().reset_index(name='n_cells'))
    totals = (counts.groupby([GROUP_COL, ANIMAL_COL], observed=True)['n_cells']
              .sum().reset_index(name='total_cells'))
    counts = counts.merge(totals, on=[GROUP_COL, ANIMAL_COL])
    counts['proportion'] = counts['n_cells'] / counts['total_cells']
    all_types = sorted(adata_obj.obs[celltype_col].astype(str).unique())
    combos = (counts[[GROUP_COL, ANIMAL_COL]].drop_duplicates().assign(key=1)
              .merge(pd.DataFrame({celltype_col: all_types, 'key': 1}), on='key').drop('key', axis=1))
    counts[celltype_col] = counts[celltype_col].astype(str)
    counts = combos.merge(counts, on=[GROUP_COL, ANIMAL_COL, celltype_col], how='left')
    counts['proportion'] = counts['proportion'].fillna(0.0)
    return counts


def plot_group_proportions(per_animal, celltypes, celltype_col, title, filename, ncols=3):
    """Vertical per-animal bars on a SHARED 0-100% y-axis, so a few-percent shift
    reads as small and is never zoom-exaggerated. The exact percentage is printed
    above each bar (rare types stay legible even when the bar is tiny), and a
    connector line through the per-group means shows the trend direction and, on
    the fixed axis, its true magnitude. n=1/arm here: descriptive, no stats."""
    Y_MAX = 100.0
    n = len(celltypes)
    if n == 0:
        return []
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    # Shared animal x-layout across every subplot: animals clustered by group,
    # a gap between groups, group center = mean x of that group's bars.
    GROUP_GAP = 1.0
    layout = {}          # animal_id -> x position
    group_centers = {}   # group -> center x
    x = 0.0
    for group in GROUP_ORDER:
        animals = sorted(per_animal.loc[per_animal[GROUP_COL] == group, ANIMAL_COL].unique())
        xs = []
        for a in animals:
            layout[a] = x
            xs.append(x)
            x += 1.0
        if xs:
            group_centers[group] = float(np.mean(xs))
        x += GROUP_GAP
    min_x = min(layout.values()) if layout else 0.0
    max_x = max(layout.values()) if layout else 1.0

    stat_rows = []
    for idx, ct in enumerate(celltypes):
        ax = axes[idx // ncols, idx % ncols]
        ct_data = per_animal[per_animal[celltype_col] == ct]
        group_means = {}
        for group in GROUP_ORDER:
            gd = ct_data[ct_data[GROUP_COL] == group]
            vals = []
            for _, r in gd.iterrows():
                xpos = layout[r[ANIMAL_COL]]
                v = float(r['proportion']) * 100.0
                ax.bar(xpos, v, width=0.7, color=GROUP_CONFIG[group]['color'],
                       alpha=0.88, edgecolor='white', linewidth=0.6, zorder=2)
                ax.text(xpos, v + Y_MAX * 0.015, f"{v:.1f}%", ha='center', va='bottom',
                        fontsize=8, clip_on=False, zorder=4)
                vals.append(v)
            if vals:
                group_means[group] = float(np.mean(vals))
        # trend line through per-group means (at n=1/arm each mean sits on its bar)
        pts = [(group_centers[g], group_means[g]) for g in GROUP_ORDER
               if g in group_centers and g in group_means]
        if len(pts) >= 2:
            xs_, ys_ = zip(*pts)
            ax.plot(xs_, ys_, color='#444444', linewidth=1.6, marker='o',
                    markersize=4, zorder=3)
        ax.set_ylim(0, Y_MAX)
        ax.set_xlim(min_x - 0.7, max_x + 0.7)
        ax.set_title(ct, fontsize=10, fontweight='bold')
        ax.set_xticks([group_centers[g] for g in GROUP_ORDER if g in group_centers])
        ax.set_xticklabels([g for g in GROUP_ORDER if g in group_centers], fontsize=9)
        ax.set_ylabel('% of cells', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        stat_rows.append({
            'population': ct, 'celltype_col': celltype_col,
            'mean_Untreated': group_means.get('Untreated', np.nan),
            'mean_D1MT': group_means.get('D1MT', np.nan),
        })
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PROP_DIR, f'{filename}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(PROP_DIR, f'{filename}.png'), bbox_inches='tight', dpi=300)
    print(f"  Saved: {filename}")
    plt.close(fig)
    return stat_rows


all_stats = []

# Tier 1 lineages
pa_t1 = per_animal_proportions(adata, 'tier1_celltype')
t1_types = [t for t in ['T cells', 'Macrophages', 'Monocytes', 'DCs',
                        'NK cells', 'B cells', 'Mast cells', 'Ciliated epithelial']
            if t in pa_t1['tier1_celltype'].unique()]
all_stats += plot_group_proportions(pa_t1, t1_types, 'tier1_celltype',
                                     'Tier 1 — Lineage Frequency (D1MT vs Untreated)',
                                     'tier1_lineage_proportions')

# Tier 2 macrophages / T cells / DCs
pa_t2 = per_animal_proportions(adata, 'tier2_full')
mac_types = sorted([t for t in pa_t2['tier2_full'].unique() if 'Macrophage' in str(t)])
if mac_types:
    all_stats += plot_group_proportions(pa_t2, mac_types, 'tier2_full',
                                         'Tier 2 — Macrophage Subtype Frequency',
                                         'tier2_macrophage_proportions')
tcell_types = sorted([t for t in pa_t2['tier2_full'].unique() if 'T cells' in str(t)])
if tcell_types:
    all_stats += plot_group_proportions(pa_t2, tcell_types, 'tier2_full',
                                         'Tier 2 — T Cell Subtype Frequency',
                                         'tier2_tcell_proportions', ncols=2)
dc_types = sorted([t for t in pa_t2['tier2_full'].unique() if 'DC' in str(t)])
if dc_types:
    all_stats += plot_group_proportions(pa_t2, dc_types, 'tier2_full',
                                         'Tier 2 — DC Subtype Frequency',
                                         'tier2_dc_proportions')

# Tier 3 T cell states (within T cells)
t_only = adata[adata.obs['tier1_celltype'] == 'T cells'].copy()
pa_t3 = per_animal_proportions(t_only, 'tier3_state')
t3_types = sorted([t for t in pa_t3['tier3_state'].unique() if t != 'N/A'])
if t3_types:
    all_stats += plot_group_proportions(pa_t3, t3_types, 'tier3_state',
                                         'Tier 3 — T Cell State Frequency (% of T cells)',
                                         'tier3_tcell_proportions', ncols=2)

# CD4 Effector continuous scores: per-animal mean, group comparison
cd4_eff = adata[adata.obs['tier3_state'] == 'CD4 Effector'].copy()
if cd4_eff.n_obs > 50:
    print("\n  CD4 Effector continuous scores by group (per-animal mean):")
    score_stat_rows = []
    for label, col in continuous_scores.items():
        if col not in cd4_eff.obs.columns:
            continue
        per_an = (cd4_eff.obs.groupby([GROUP_COL, ANIMAL_COL], observed=True)[col]
                  .mean().reset_index())
        u = p = np.nan
        gv = {g: per_an.loc[per_an[GROUP_COL] == g, col].values for g in GROUP_ORDER}
        if len(gv['Untreated']) > 0 and len(gv['D1MT']) > 0:
            try:
                u, p = mannwhitneyu(gv['Untreated'], gv['D1MT'], alternative='two-sided')
            except ValueError:
                pass
        print(f"    {label:22s} Untreated={np.mean(gv['Untreated']):.3f}  "
              f"D1MT={np.mean(gv['D1MT']):.3f}  MWU p={p if np.isnan(p) else round(p,3)}")
        score_stat_rows.append({'score': label, 'column': col,
                                'mean_Untreated': float(np.mean(gv['Untreated'])),
                                'mean_D1MT': float(np.mean(gv['D1MT'])),
                                'mwu_U': u, 'mwu_p': p})
    pd.DataFrame(score_stat_rows).to_csv(
        os.path.join(PROP_DIR, 'cd4_effector_score_comparison.csv'), index=False)

if all_stats:
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv(os.path.join(PROP_DIR, 'proportion_comparison_stats.csv'), index=False)
    print("\n  Saved: proportion_comparison_stats.csv (per-group mean proportions)")
    print("  NOTE: n=1/arm two-sample analysis — proportion bars are descriptive")
    print("        (shared 0-100% axis, % printed per bar); no significance testing.")


# %% Cell 10 — Save Annotated AnnData
print("\n" + "=" * 80)
print("SAVING ANNOTATED ADATA")
print("=" * 80)

adata.write_h5ad(ADATA_OUT)
print(f"  Saved: {ADATA_OUT}")
print(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

tier1_summary = adata.obs['tier1_celltype'].value_counts().to_frame('n_cells')
tier1_summary['pct'] = (tier1_summary['n_cells'] / adata.n_obs * 100).round(1)
tier1_summary.to_csv(os.path.join(OUT_DIR, 'tier1_counts.csv'))

tier2_summary = adata.obs['tier2_full'].value_counts().to_frame('n_cells')
tier2_summary['pct'] = (tier2_summary['n_cells'] / adata.n_obs * 100).round(1)
tier2_summary.to_csv(os.path.join(OUT_DIR, 'tier2_counts.csv'))

if t_cell_mask.any():
    tier3_summary = adata.obs.loc[t_cell_mask, 'tier3_state'].value_counts().to_frame('n_cells')
    tier3_summary.to_csv(os.path.join(OUT_DIR, 'tier3_tcell_counts.csv'))

new_cols = [c for c in adata.obs.columns if c.startswith(('leiden', 'tier', 'score_', 'exhaustion',
            'IDO_', 'SIV_', 'granuloma', 'AM_', 'IFN_', 'M1_', 'dotplot'))]
print(f"\n  New annotation columns added ({len(new_cols)}):")
for c in sorted(new_cols):
    print(f"    {c}")

print(f"\n{'=' * 80}")
print("PHASE 2 COMPLETE")
print(f"{'=' * 80}")
print(f"  Annotation output:  {OUT_DIR}")
print(f"  Proportion output:  {PROP_DIR}")
print(f"  Annotated AnnData:  {ADATA_OUT}")
print(f"  Groups: {sorted(adata.obs[GROUP_COL].unique().tolist())}")
print(f"  Tier 1 lineages: {adata.obs['tier1_celltype'].nunique()}")
print(f"  Tier 2 subtypes: {adata.obs['tier2_full'].nunique()}")
print(f"  Tier 3 T cell states: {adata.obs.loc[t_cell_mask, 'tier3_state'].nunique()}")
print(f"\n  Next: Phase 3 — DEG (D1MT vs Untreated per cluster)")
