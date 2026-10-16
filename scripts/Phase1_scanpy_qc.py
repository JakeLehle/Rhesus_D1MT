#!/usr/bin/env python3
"""
Phase 1 — Load, QC, and Embed  (Rhesus D1MT Monotherapy Project)
================================================================
Standalone pipeline for the Singh B et al. D1MT-monotherapy BAL scRNA-seq.

Reads cellranger count matrices for the 8 week-15 BAL libraries (4 D1MT,
4 Untreated), performs mitochondrial and expression-based QC, normalizes,
batch-corrects (BBKNN on library), and produces a UMAP ready for annotation.

Groups:
  - D1MT      : IDO inhibitor (1-methyl-D-tryptophan) monotherapy, no ART
  - Untreated : no treatment (3 of 4 are historical Sharan JCI 2022 controls)

Single timepoint (week 15, pre-SIV). Unlike the D1MT+cART project there is NO
time-series axis, and pathogen signal is treated as NOISE:
  - Mtb and SIV genes are tallied per library and per cell (stash), a per-sample
    enrichment table is written so single-library concentration (index hop /
    ambient) is visible, and pathogen genes are then EXCLUDED from HVG/PCA/
    clustering. They are retained in the matrix so counts stay queryable.

Custom reference: Mmul10 (rhesus) + MtbCDC1551 + SIVmac239

Gene identification notes:
  - Rhesus MT genes use KEG06_ locus tags (NOT MT- prefix)
  - SIV genes: gene_id = "gene-gag", var_name = "gag" (no prefix on var_names)
  - Mtb genes: gene_id = "MT_RS#####"

Run in Spyder (# %% cells) or standalone.  Output dir: analysis/phase1_qc/

Author: Jake Lehle / Kaushal Lab
Date: July 2026
"""

# %% Cell 1 — Configuration, Paths, and Sample Registry
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless for SLURM; comment out for interactive Spyder
import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === ADJUSTABLE QC PARAMETERS ===
MT_THRESHOLD = 20
MIN_GENES = 200
MIN_CELLS_PER_GENE = 20
MAD_MULTIPLIER = 5
COMPROMISED_MT_FLOOR = 20
TOP_N_GENE_PCT = 20
EXCLUDE_COMPROMISED = True          # n=4/arm: FLAG compromised samples, do not auto-drop; review first
PATHOGEN_CONCENTRATION_FLAG = 0.50   # warn if one library holds > this share of total pathogen UMIs

# === PATHOGEN GENES (treated as NOISE in this project) ===
SIV_GENE_LIST = ['gag', 'pol', 'vif', 'vpx', 'vpr', 'tat', 'rev', 'env', 'nef']

# === PATHS ===
BASE_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT'
METADATA_PKL = os.path.join(BASE_DIR, 'metadata', 'dimt_metadata.pkl')
CELLRANGER_BASE = os.path.join(BASE_DIR, 'cellranger_counts')
OUT_DIR = os.path.join(BASE_DIR, 'analysis', 'phase1_qc')
ADATA_OUT = os.path.join(BASE_DIR, 'analysis', 'adata_d1mt_qcd.h5ad')
os.makedirs(OUT_DIR, exist_ok=True)

# === COLORS (hex) ===
GROUP_ORDER = ['Untreated', 'D1MT']
GROUP_COLORS = {'Untreated': '#1976D2', 'D1MT': '#D32F2F'}
COHORT_ORDER = ['Current', 'Historical']
COHORT_COLORS = {'Current': '#2E7D32', 'Historical': '#F9A825'}
CRYO_ORDER = ['Fresh', 'Thawed']
CRYO_COLORS = {'Fresh': '#2E7D32', 'Thawed': '#C62828'}

# === SAMPLE REGISTRY (from metadata pkl — single source of truth) ===
with open(METADATA_PKL, 'rb') as f:
    _meta = pickle.load(f)

SAMPLE_REGISTRY = []
for animal_id, m in _meta.items():
    SAMPLE_REGISTRY.append({
        'sample_id': m['cr_id'],          # cellranger output dir == animal_id
        'animal_id': animal_id,
        'group': m['group'],
        'sex': m.get('sex'),
        'source': m.get('source'),
        'cohort': m.get('cohort'),
        'cryo_status': m.get('cryo_status'),
        'timepoint': m.get('timepoint', 'Week 15'),
    })
SAMPLE_REGISTRY.sort(key=lambda s: (GROUP_ORDER.index(s['group']), s['sample_id']))

EXPECTED_DESIGN = {g: [] for g in GROUP_ORDER}
for s in SAMPLE_REGISTRY:
    EXPECTED_DESIGN[s['group']].append(s['animal_id'])

print(f"Sample registry: {len(SAMPLE_REGISTRY)} samples")
for g in GROUP_ORDER:
    print(f"  {g}: {len(EXPECTED_DESIGN[g])} — {EXPECTED_DESIGN[g]}")


# %% Cell 2 — Read cellranger matrices and build combined AnnData
print("\n" + "=" * 80)
print("LOADING CELLRANGER COUNT MATRICES")
print("=" * 80)

adatas = []
load_summary = []

for entry in SAMPLE_REGISTRY:
    sid = entry['sample_id']
    mtx_path = os.path.join(CELLRANGER_BASE, sid, 'outs', 'filtered_feature_bc_matrix')

    if not os.path.exists(mtx_path):
        print(f"  *** MISSING: {mtx_path}")
        load_summary.append({**entry, 'n_cells': 0, 'n_genes': 0, 'status': 'MISSING'})
        continue

    try:
        a = sc.read_10x_mtx(mtx_path, var_names='gene_symbols', cache=False)
    except Exception as e:
        print(f"  *** ERROR reading {sid}: {e}")
        load_summary.append({**entry, 'n_cells': 0, 'n_genes': 0, 'status': f'ERROR: {e}'})
        continue

    a.obs_names = [f"{sid}_{bc}" for bc in a.obs_names]
    a.obs['sample_id'] = sid
    a.obs['animal_id'] = entry['animal_id']
    a.obs['group'] = entry['group']
    a.obs['sex'] = entry['sex']
    a.obs['source'] = entry['source']
    a.obs['cohort'] = entry['cohort']
    a.obs['cryo_status'] = entry['cryo_status']
    a.obs['timepoint'] = entry['timepoint']

    adatas.append(a)
    load_summary.append({**entry, 'n_cells': a.shape[0], 'n_genes': a.shape[1], 'status': 'OK'})
    print(f"  {sid:12s} [{entry['group']:9s}]  {a.shape[0]:>7,} cells  {a.shape[1]:>6,} genes  OK")

if len(adatas) == 0:
    raise RuntimeError("No matrices loaded — check CELLRANGER_BASE and that cellranger completed.")

print(f"\nConcatenating {len(adatas)} samples...")
adata = sc.concat(adatas, join='outer', merge='same')
adata.obs_names_make_unique()

# Fixed category orders for stable palettes
adata.obs['group'] = pd.Categorical(adata.obs['group'], categories=GROUP_ORDER)
adata.obs['cohort'] = pd.Categorical(adata.obs['cohort'], categories=COHORT_ORDER)
adata.obs['cryo_status'] = pd.Categorical(adata.obs['cryo_status'], categories=CRYO_ORDER)

print(f"\nCombined AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

df_load = pd.DataFrame(load_summary)
print("\n--- Load Summary ---")
print(df_load[['sample_id', 'group', 'animal_id', 'cohort', 'n_cells', 'status']].to_string(index=False))

missing = df_load[df_load['status'] != 'OK']
if len(missing) > 0:
    print(f"\n*** WARNING: {len(missing)} samples failed to load ***")
    print(missing[['sample_id', 'status']].to_string(index=False))

df_load.to_csv(os.path.join(OUT_DIR, 'load_summary.csv'), index=False)


# %% Cell 3 — Gene Class Identification, QC Metrics, and Pathogen Provenance
print("\n" + "=" * 80)
print("GENE CLASS IDENTIFICATION AND QC METRICS")
print("=" * 80)

# --- Mitochondrial genes (rhesus Mmul10 uses KEG06_ locus tags) ---
adata.var['mt'] = adata.var_names.str.startswith('KEG06_')
n_mt = adata.var['mt'].sum()
if n_mt == 0:
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    n_mt = adata.var['mt'].sum()
    if n_mt > 0:
        print(f"  NOTE: Using MT- prefix fallback ({n_mt} genes)")
    else:
        print("  *** WARNING: No MT genes found (tried KEG06_ and MT-) ***")
        print(f"  First 20 genes: {adata.var_names[:20].tolist()}")

# --- Pathogen gene identification (via gene_ids; fallback to var_names) ---
if 'gene_ids' in adata.var.columns:
    print("  Using gene_ids column for pathogen identification")
    adata.var['pathogen_mtb'] = adata.var['gene_ids'].str.startswith('MT_RS')
    adata.var['pathogen_siv'] = adata.var['gene_ids'].str.startswith('gene-')
else:
    print("  WARNING: gene_ids not found — using var_names fallback")
    adata.var['pathogen_mtb'] = adata.var_names.str.startswith('MT_RS')
    adata.var['pathogen_siv'] = adata.var_names.isin(SIV_GENE_LIST)

adata.var['pathogen'] = adata.var['pathogen_mtb'] | adata.var['pathogen_siv']
adata.var['host'] = ~adata.var['pathogen']

n_mtb = adata.var['pathogen_mtb'].sum()
n_siv = adata.var['pathogen_siv'].sum()
n_host = adata.var['host'].sum()
siv_found = adata.var_names[adata.var['pathogen_siv']].tolist()

print(f"\nGene classes:")
print(f"  Host (Mmul10):  {n_host:,} genes")
print(f"  Mitochondrial:  {n_mt} genes (KEG06_ prefix)")
print(f"  Pathogen (Mtb): {n_mtb} genes  [NOISE in this project]")
print(f"  Pathogen (SIV): {n_siv} genes: {siv_found}  [NOISE in this project]")

# --- QC metrics (pathogen tallies retained for the stash) ---
sc.pp.calculate_qc_metrics(
    adata, qc_vars=['mt', 'pathogen_mtb', 'pathogen_siv'],
    percent_top=[TOP_N_GENE_PCT], inplace=True
)

# --- PATHOGEN PROVENANCE / STASH -----------------------------------------
# We expect ~0 real pathogen signal (no SIV pre-coinfection; 3' can't read Mtb).
# Any nonzero is ambient / index-hopping / mismapping. Surface it per library
# so single-sample concentration is obvious, and export the nonzero cells.
print("\n--- PATHOGEN PROVENANCE (stash; treated as noise) ---")
prov = (
    adata.obs.groupby('sample_id', observed=True)
    .agg(
        group=('group', 'first'),
        cohort=('cohort', 'first'),
        n_cells=('total_counts', 'size'),
        n_cells_siv=('total_counts_pathogen_siv', lambda x: int((x > 0).sum())),
        n_cells_mtb=('total_counts_pathogen_mtb', lambda x: int((x > 0).sum())),
        siv_umi=('total_counts_pathogen_siv', 'sum'),
        mtb_umi=('total_counts_pathogen_mtb', 'sum'),
    )
    .reset_index()
)
prov['pct_cells_siv'] = 100 * prov['n_cells_siv'] / prov['n_cells']
prov['pct_cells_mtb'] = 100 * prov['n_cells_mtb'] / prov['n_cells']
tot_siv = prov['siv_umi'].sum()
tot_mtb = prov['mtb_umi'].sum()
prov['share_siv_umi'] = prov['siv_umi'] / tot_siv if tot_siv > 0 else 0.0
prov['share_mtb_umi'] = prov['mtb_umi'] / tot_mtb if tot_mtb > 0 else 0.0
print(prov.to_string(index=False))
prov.to_csv(os.path.join(OUT_DIR, 'pathogen_provenance_per_sample.csv'), index=False)

# Concentration flags (uniform share across 8 libraries would be ~0.125)
for label, tot, col in [('SIV', tot_siv, 'share_siv_umi'), ('Mtb', tot_mtb, 'share_mtb_umi')]:
    if tot > 0:
        top = prov.loc[prov[col].idxmax()]
        if top[col] > PATHOGEN_CONCENTRATION_FLAG:
            print(f"  *** {label} NOISE CONCENTRATED: {top['sample_id']} holds "
                  f"{100*top[col]:.0f}% of {tot:,.0f} total {label} UMIs "
                  f"(index-hop / ambient suspect) ***")
    else:
        print(f"  {label}: 0 total UMIs across all libraries (clean).")

# Per-cell export of every nonzero-pathogen cell
path_mask_cells = (
    (adata.obs['total_counts_pathogen_siv'] > 0) |
    (adata.obs['total_counts_pathogen_mtb'] > 0)
)
path_cols = ['sample_id', 'animal_id', 'group', 'cohort',
             'total_counts_pathogen_siv', 'total_counts_pathogen_mtb',
             'total_counts', 'n_genes_by_counts']
adata.obs.loc[path_mask_cells, path_cols].to_csv(
    os.path.join(OUT_DIR, 'pathogen_positive_cells.csv'))
print(f"  Nonzero-pathogen cells exported: {int(path_mask_cells.sum()):,}")

# --- Per-sample MT summary ---
mt_summary = (
    adata.obs.groupby('sample_id', observed=True)
    .agg(
        group=('group', 'first'),
        animal_id=('animal_id', 'first'),
        cohort=('cohort', 'first'),
        n_cells=('pct_counts_mt', 'size'),
        median_mt=('pct_counts_mt', 'median'),
        mean_mt=('pct_counts_mt', 'mean'),
        pct95_mt=('pct_counts_mt', lambda x: np.percentile(x, 95)),
        median_genes=('n_genes_by_counts', 'median'),
        median_umis=('total_counts', 'median'),
    )
    .sort_values(['group', 'animal_id'])
    .reset_index()
)
mt_summary['compromised'] = mt_summary['median_mt'] > COMPROMISED_MT_FLOOR

print("\n--- Per-Sample QC Summary ---")
print(mt_summary.to_string(index=False))

n_compromised = int(mt_summary['compromised'].sum())
if n_compromised > 0:
    print(f"\n*** {n_compromised} COMPROMISED SAMPLE(S) (median MT% > {COMPROMISED_MT_FLOOR}%) ***")
    print(mt_summary[mt_summary['compromised']][
        ['sample_id', 'group', 'cohort', 'median_mt', 'median_genes']].to_string(index=False))
    print("  NOTE: with n=4/arm these are FLAGGED, not auto-dropped "
          f"(EXCLUDE_COMPROMISED={EXCLUDE_COMPROMISED}).")
else:
    print(f"\nNo compromised samples (all median MT% < {COMPROMISED_MT_FLOOR}%)")
mt_summary.to_csv(os.path.join(OUT_DIR, 'mt_qc_summary.csv'), index=False)

# --- MT% distribution plots ---
print("\nGenerating MT% distribution plots...")
samples_sorted = mt_summary['sample_id'].tolist()
n_cols = 4
n_rows = int(np.ceil(len(samples_sorted) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = np.atleast_1d(axes).flatten()
for i, sid in enumerate(samples_sorted):
    ax = axes[i]
    subset = adata.obs[adata.obs['sample_id'] == sid]
    info = mt_summary[mt_summary['sample_id'] == sid].iloc[0]
    bar_color = GROUP_COLORS.get(info['group'], '#607D8B')
    ax.hist(subset['pct_counts_mt'], bins=50, color=bar_color, alpha=0.7, edgecolor='none')
    ax.axvline(info['median_mt'], color='#000000', linestyle='--', linewidth=1.5,
               label=f"median={info['median_mt']:.1f}%")
    if MT_THRESHOLD is not None:
        ax.axvline(MT_THRESHOLD, color='#F9A825', linestyle='-', linewidth=1.5)
    title_color = '#C62828' if info['compromised'] else '#000000'
    ax.set_title(f"{sid} [{info['group']}/{info['cohort']}]\n"
                 f"n={len(subset):,}  med_genes={info['median_genes']:.0f}",
                 fontsize=8, color=title_color)
    ax.set_xlabel('MT %', fontsize=8)
    ax.set_ylabel('Cells', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlim(0, 100)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Mitochondrial % Distribution per Sample', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mt_distribution_per_sample.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'mt_distribution_per_sample.pdf'), bbox_inches='tight')
plt.show()
plt.close()

# --- MT% vs n_genes scatter, by group and by cohort (batch/cryo check) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for group in GROUP_ORDER:
    mask = adata.obs['group'] == group
    axes[0].scatter(adata.obs.loc[mask, 'pct_counts_mt'],
                    adata.obs.loc[mask, 'n_genes_by_counts'],
                    s=0.5, alpha=0.1, c=GROUP_COLORS[group], label=group, rasterized=True)
axes[0].set_xlabel('MT %'); axes[0].set_ylabel('Genes detected')
axes[0].set_title('MT% vs Genes by Group'); axes[0].legend(markerscale=10)
for cohort in COHORT_ORDER:
    mask = adata.obs['cohort'] == cohort
    axes[1].scatter(adata.obs.loc[mask, 'pct_counts_mt'],
                    adata.obs.loc[mask, 'n_genes_by_counts'],
                    s=0.5, alpha=0.1, c=COHORT_COLORS[cohort], label=cohort, rasterized=True)
axes[1].set_xlabel('MT %'); axes[1].set_ylabel('Genes detected')
axes[1].set_title('MT% vs Genes by Cohort (fresh vs thawed)'); axes[1].legend(markerscale=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mt_vs_genes_scatter.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'mt_vs_genes_scatter.pdf'), bbox_inches='tight')
plt.show()
plt.close()


# %% Cell 4 — QC Filtering
print("\n" + "=" * 80)
print("QC FILTERING")
print("=" * 80)

n_before = adata.shape[0]
print(f"Starting cells: {n_before:,}")

# Step 1: Compromised samples — FLAG; only drop if explicitly enabled
compromised_ids = mt_summary.loc[mt_summary['compromised'], 'sample_id'].tolist()
if compromised_ids and EXCLUDE_COMPROMISED:
    print(f"\n  EXCLUDE_COMPROMISED=True — dropping {len(compromised_ids)}: {compromised_ids}")
    adata = adata[~adata.obs['sample_id'].isin(compromised_ids)].copy()
    print(f"  Cells remaining: {adata.shape[0]:,}")
elif compromised_ids:
    print(f"\n  {len(compromised_ids)} compromised sample(s) FLAGGED but retained "
          f"(n=4/arm; review before dropping): {compromised_ids}")
else:
    print("\n  No compromised samples.")

# Step 2: MT threshold filter
if MT_THRESHOLD is not None:
    n_pre = adata.shape[0]
    adata = adata[adata.obs['pct_counts_mt'] <= MT_THRESHOLD].copy()
    print(f"\n  MT filter (>{MT_THRESHOLD}%): removed {n_pre - adata.shape[0]:,}")
else:
    print(f"\n  MT threshold: None (tracked as metadata only)")

# Step 3: Min genes
n_pre = adata.shape[0]
adata = adata[adata.obs['n_genes_by_counts'] >= MIN_GENES].copy()
print(f"  Min genes filter (<{MIN_GENES}): removed {n_pre - adata.shape[0]:,}")

# Step 4: Per-sample MAD
print(f"\n  Per-sample MAD filtering (multiplier={MAD_MULTIPLIER})...")
cells_to_keep = pd.Series(True, index=adata.obs_names)
mad_report = []
for sid in adata.obs['sample_id'].unique():
    mask = adata.obs['sample_id'] == sid
    subset = adata.obs.loc[mask]
    n_start = int(mask.sum())
    sample_keep = pd.Series(True, index=subset.index)

    genes = subset['n_genes_by_counts']
    med_g = genes.median()
    mad_g = np.median(np.abs(genes - med_g)) * 1.4826
    sample_keep &= (genes >= max(med_g - MAD_MULTIPLIER * mad_g, MIN_GENES))

    umis = subset['total_counts']
    med_u = umis.median()
    mad_u = np.median(np.abs(umis - med_u)) * 1.4826
    sample_keep &= (umis >= max(med_u - MAD_MULTIPLIER * mad_u, 0))

    top_col = f'pct_counts_in_top_{TOP_N_GENE_PCT}_genes'
    if top_col in subset.columns:
        topgene = subset[top_col]
        med_t = topgene.median()
        mad_t = np.median(np.abs(topgene - med_t)) * 1.4826
        sample_keep &= (topgene <= med_t + MAD_MULTIPLIER * mad_t)

    cells_to_keep.loc[subset.index] = sample_keep
    mad_report.append({'sample_id': sid, 'n_before': n_start,
                       'n_removed_mad': int(n_start - sample_keep.sum()),
                       'n_after': int(sample_keep.sum())})

n_pre = adata.shape[0]
adata = adata[cells_to_keep].copy()
print(f"  MAD removed {n_pre - adata.shape[0]:,} cells")
print(f"  Cells remaining: {adata.shape[0]:,}")
df_mad = pd.DataFrame(mad_report).sort_values('sample_id')
print("\n--- MAD QC Report ---")
print(df_mad.to_string(index=False))
df_mad.to_csv(os.path.join(OUT_DIR, 'mad_qc_report.csv'), index=False)

# Step 5: Gene filtering (HOST ONLY — pathogen genes protected from filter_genes)
print(f"\n  Gene filtering (min_cells={MIN_CELLS_PER_GENE}, host only)...")
n_genes_before = adata.shape[1]
host_mask = adata.var['host']
pathogen_mask = adata.var['pathogen']

host_genes = adata[:, host_mask].copy()
sc.pp.filter_genes(host_genes, min_cells=MIN_CELLS_PER_GENE)
host_genes_kept = host_genes.var_names.tolist()
pathogen_genes_kept = adata.var_names[pathogen_mask].tolist()  # keep ALL pathogen genes (stash)
all_genes_kept = host_genes_kept + pathogen_genes_kept
adata = adata[:, adata.var_names.isin(all_genes_kept)].copy()

n_siv_kept = int(adata.var['pathogen_siv'].sum())
n_mtb_kept = int(adata.var['pathogen_mtb'].sum())
print(f"  Host: {int(host_mask.sum()):,} → {len(host_genes_kept):,}")
print(f"  Pathogen kept (stash): {len(pathogen_genes_kept)} (Mtb={n_mtb_kept}, SIV={n_siv_kept})")
print(f"  Total: {n_genes_before:,} → {adata.shape[1]:,}")

n_after = adata.shape[0]
print(f"\n{'=' * 60}")
print(f"QC SUMMARY: {n_before:,} → {n_after:,} cells "
      f"(removed {n_before-n_after:,}, {(n_before-n_after)/n_before*100:.1f}%)")
print(f"{'=' * 60}")


# %% Cell 5 — Post-QC Diagnostics (single timepoint: by group and animal)
print("\n" + "=" * 80)
print("POST-QC DIAGNOSTICS")
print("=" * 80)

ct = pd.crosstab(adata.obs['group'], adata.obs['animal_id'], margins=True)
print("\nCell counts — Group × Animal:")
print(ct.to_string())
ct.to_csv(os.path.join(OUT_DIR, 'postqc_group_animal.csv'))

# Design completeness (which expected animals survived QC)
print("\n--- DESIGN COMPLETENESS ---")
for group in GROUP_ORDER:
    mask = adata.obs['group'] == group
    present = sorted(adata.obs.loc[mask, 'animal_id'].unique())
    gaps = [a for a in EXPECTED_DESIGN[group] if a not in present]
    status = "OK" if not gaps else f"GAPS: {gaps}"
    print(f"  {group:9s}: {int(mask.sum()):>7,} cells, "
          f"{len(present)}/{len(EXPECTED_DESIGN[group])} animals  [{status}]")

# Per-animal yield within group (one library per animal — watch for a dominant library)
print("\n--- PER-ANIMAL YIELD (within group) ---")
for group in GROUP_ORDER:
    sub = adata.obs.loc[adata.obs['group'] == group]
    counts = sub['animal_id'].value_counts()
    total = counts.sum()
    print(f"  {group} (n={total:,}):")
    for animal, count in counts.items():
        pct = 100 * count / total
        flag = " *** DOMINANT" if pct > 60 else ""
        print(f"    {animal}: {count:>7,} ({pct:5.1f}%){flag}")

# Post-QC pathogen verification (should still be ~0)
print("\n--- POST-QC PATHOGEN VERIFICATION (noise; expect ~0) ---")
siv_post = int((adata.obs['total_counts_pathogen_siv'] > 0).sum())
mtb_post = int((adata.obs['total_counts_pathogen_mtb'] > 0).sum())
print(f"  SIV+ cells: {siv_post:,} / {adata.n_obs:,} ({100*siv_post/adata.n_obs:.2f}%)")
print(f"  Mtb+ cells: {mtb_post:,} / {adata.n_obs:,} ({100*mtb_post/adata.n_obs:.2f}%)")


# %% Cell 6 — Normalize, Batch Correct, Embed, and Save
print("\n" + "=" * 80)
print("NORMALIZATION AND EMBEDDING")
print("=" * 80)

print("\n  Log-normalizing (target_sum=1e4)...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

print("  Finding highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='sample_id')

# --- EXCLUDE pathogen genes from HVG so noise cannot enter PCA/clustering ---
path_mask = adata.var['pathogen'].values
n_path_hvg = int((adata.var['highly_variable'].values & path_mask).sum())
if n_path_hvg > 0:
    print(f"  *** {n_path_hvg} pathogen gene(s) were flagged HVG — forcing OFF "
          f"(ambient pathogen was variable; noted as noise) ***")
adata.var.loc[path_mask, 'highly_variable'] = False
print(f"  HVGs used (pathogen excluded): {int(adata.var['highly_variable'].sum())}")

print("  Running PCA (n_comps=50)...")
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

print("  Running BBKNN (batch_key='sample_id' == library/GEM-well)...")
try:
    import bbknn
    bbknn.bbknn(adata, batch_key='sample_id')
    print("  BBKNN complete.")
except ImportError:
    print("  *** BBKNN not installed — falling back to standard neighbors ***")
    sc.pp.neighbors(adata, n_pcs=30)

print("  Computing UMAP...")
sc.tl.umap(adata)

# --- QC UMAP overview (group / animal / cohort / cryo / genes / MT) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
sc.pl.umap(adata, color='group', ax=axes[0, 0], show=False, frameon=False,
           title='Group', palette=[GROUP_COLORS[g] for g in GROUP_ORDER])
sc.pl.umap(adata, color='animal_id', ax=axes[0, 1], show=False, frameon=False, title='Animal ID')
sc.pl.umap(adata, color='cohort', ax=axes[0, 2], show=False, frameon=False,
           title='Cohort (current vs historical)', palette=[COHORT_COLORS[c] for c in COHORT_ORDER])
sc.pl.umap(adata, color='cryo_status', ax=axes[1, 0], show=False, frameon=False,
           title='Cryo (fresh vs thawed)', palette=[CRYO_COLORS[c] for c in CRYO_ORDER])
sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1, 1], show=False,
           frameon=False, title='Genes Detected', cmap='viridis')
sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 2], show=False,
           frameon=False, title='MT %', cmap='inferno')
plt.suptitle(f'Post-QC UMAP — {adata.shape[0]:,} cells', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'umap_qc_overview.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'umap_qc_overview.pdf'), bbox_inches='tight')
plt.show()
plt.close()

# --- Per-group UMAP colored by animal (anchor check: does current-untreated
#     43109 embed with the historical untreated, or drift toward D1MT?) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, group in enumerate(GROUP_ORDER):
    mask = adata.obs['group'] == group
    sc.pl.umap(adata[mask], color='animal_id', ax=axes[i], show=False, frameon=False,
               title=f'{group}\n({int(mask.sum()):,} cells)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'umap_by_group_animal.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'umap_by_group_animal.pdf'), bbox_inches='tight')
plt.show()
plt.close()

# --- Save ---
print(f"\n  Saving QC'd AnnData to: {ADATA_OUT}")
adata.write_h5ad(ADATA_OUT)

print(f"\n{'=' * 80}")
print("PHASE 1 COMPLETE")
print(f"{'=' * 80}")
print(f"  Output directory: {OUT_DIR}")
print(f"  AnnData file:     {ADATA_OUT}")
print(f"  Total cells:      {adata.shape[0]:,}")
print(f"  Total genes:      {adata.shape[1]:,}")
print(f"  Groups:           {sorted(adata.obs['group'].unique().tolist())}")
print(f"  Animals:          {sorted(adata.obs['animal_id'].unique())}")
print(f"  MT genes:         {int(adata.var['mt'].sum())} (KEG06_ prefix)")
print(f"  Pathogen (stash): Mtb={int(adata.var['pathogen_mtb'].sum())}, "
      f"SIV={int(adata.var['pathogen_siv'].sum())} — excluded from HVG/PCA")
print(f"\n  Next: Phase 2 — Tiered Annotation")
