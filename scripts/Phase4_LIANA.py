#!/usr/bin/env python3
"""
Phase 4 — Cell-Cell Communication (LIANA+)  (Rhesus D1MT Project)
=================================================================
Infers ligand-receptor interactions among macrophage subtypes, T cells,
monocytes, and NK cells, comparing D1MT vs Untreated at the single week-15
BAL timepoint. Addresses the paper's central claim about enhanced
T cell <-> macrophage immune interactions.

CURRENT RUN IS n=1 PER ARM (43109 Untreated male, 43102 D1MT female).
Every differential below is one animal vs one animal, with sex perfectly
aliased with treatment and a library-depth gap between them. Treat all of it
as EXPLORATORY / hypothesis-generating. No treatment claim is supportable.

Key biological axes (T cell <-> macrophage):
  - IFN-g activation:      T cell IFNG -> Mac IFNGR1/2
  - PD-L1 / PD-1:          Mac CD274 -> T cell PDCD1 (suppressive)
  - Chemokine recruitment: Mac CXCL9/10/11 -> T cell CXCR3
  - Antigen presentation:  Mac MHC-II -> T cell CD4
  - Co-stimulation:        Mac CD80/CD86 -> T cell CD28

UPDATES IN THIS VERSION
  1. SYMMETRIC cell-type gate. Previously MIN_CELLS was applied per group
     independently, so IFN-responsive (D1MT 18 / Untx 1) and M1 IM-like
     (D1MT 6 / Untx 1) entered only the D1MT run. That crashed the Untreated
     dotplots AND manufactured "D1MT-exclusive" co-stimulation edges that were
     1-vs-18-cell artifacts, not biology. A cell type now must clear
     MIN_CELLS_PER_GROUP in BOTH groups or it is dropped from BOTH runs.
  2. Complex-aware key-interaction matching. `str.contains('CD4')` also matched
     CD40/CD44 and `'TNF'` matched TNFSF13B. Now splits complexes on '_' and
     tests exact membership.
  3. Sex-confound flagging. AR-driven edges (CALM3->AR, SERTAD1->AR were top
     "stronger in Untreated" last run) are the male Untreated animal, not
     treatment. Flagged in the CSV and excluded from headline lists.
  4. Significance requirement on the differential, so rank shifts between two
     non-specific interactions stop topping the list.
  5. DEG convergence panel: surfaces only interactions whose ligand is a DEG in
     the source cell type and/or receptor is a DEG in the target cell type,
     with directional concordance. This is the honest signal filter.
  6. Key-gene detection diagnostic, so we can see whether EXPR_PROP is silently
     removing the low-expressed axes we care about (PDCD1, CXCR3).
  7. Rerun figures: key-axis dumbbell (D1MT vs Untreated), fixed per-group
     dotplots, convergence figure.

Gene symbols: Mmul10 uses MAMU- for MHC; LIANA's consensus resource uses HLA-.
A working copy is renamed MAMU->HLA. MHC-I (MAMU-A/B) interactions are
haplotype-confounded and are flagged, matching the Phase 3 MHC-I exclusion.

Install: pip install liana --break-system-packages
Run in Spyder (# %% cells) or standalone. Output dir: analysis/phase4_liana/

Author: Jake Lehle / Kaushal Lab
Date: July 2026
"""

# %% Cell 1 — Configuration, Load, and Diagnostics
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless for SLURM; comment out for interactive Spyder
import matplotlib.pyplot as plt
import scipy.sparse as sp
import textwrap
import glob
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === PATHS ===
BASE_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT'
ADATA_IN = os.path.join(BASE_DIR, 'analysis', 'adata_d1mt_annotated.h5ad')
DEG_DIR = os.path.join(BASE_DIR, 'analysis', 'phase3_deg')
OUT_DIR = os.path.join(BASE_DIR, 'analysis', 'phase4_liana')
os.makedirs(OUT_DIR, exist_ok=True)

# === LIANA PARAMETERS ===
RESOURCE_NAME = 'consensus'
EXPR_PROP = 0.1          # OPEN: 0.05 would retain low-expressed PDCD1/CXCR3.
                         # Cell 2b reports detection so we can decide from data.
MIN_CELLS = 5            # LIANA's internal per-run floor
MIN_CELLS_PER_GROUP = 30 # symmetric gate; matches Phase 3 DEG MIN_CELLS
N_PERMS = 1000
N_JOBS = 4

# === DIFFERENTIAL / CONVERGENCE ===
SPEC_SIG = 0.05          # specificity_rank threshold for "significant"
REQUIRE_SIG_EITHER = True  # differential must be significant in >=1 group
TOP_N_REPORT = 10

# === GROUPS ===
GROUP_COL = 'group'
CELLTYPE_COL = 'tier2_full'
GROUP_ORDER = ['Untreated', 'D1MT']
COLOR_UNTX = '#1976D2'
COLOR_D1MT = '#D32F2F'
GROUP_CONFIG = {'Untreated': {'color': COLOR_UNTX}, 'D1MT': {'color': COLOR_D1MT}}

# === CELL TYPES (match Phase 2 tier2_full labels) ===
# Kept in full; the symmetric gate drops underpowered ones with a printed reason,
# so a powered re-run brings them back automatically.
CELLTYPES_TO_INCLUDE = [
    'Macrophages: AM-like',
    'Macrophages: TREM2+',
    'Macrophages: M2 IM-like',
    'Macrophages: IFN-responsive',
    'Macrophages: M1 IM-like',
    'T cells: CD4',
    'T cells: CD8',
    'Monocytes: Classical',
    'NK cells',
]

# === MAMU -> HLA gene renaming (LIANA resource is human-symboled) ===
MAMU_TO_HLA = {
    'MAMU-DRA': 'HLA-DRA', 'MAMU-DRB1': 'HLA-DRB1', 'MAMU-DRB5': 'HLA-DRB5',
    'MAMU-DPA1': 'HLA-DPA1', 'MAMU-DPB1': 'HLA-DPB1',
    'MAMU-DQA1': 'HLA-DQA1', 'MAMU-DQB1': 'HLA-DQB1',
    'MAMU-A': 'HLA-A', 'MAMU-B': 'HLA-B', 'MAMU-E': 'HLA-E', 'MAMU-F': 'HLA-F',
}

# === CONFOUNDED GENES (flagged, not silently dropped) ===
# AR is X-linked and expressed in both sexes (lower in females), so it is NOT a
# clean sex marker — but with sex perfectly aliased here, AR-driven edges track
# the male Untreated animal. Flag and exclude from headline lists.
SEX_CONFOUNDED = {'AR', 'XIST', 'RPS4Y1', 'RPS4Y2', 'DDX3Y', 'UTY', 'USP9Y',
                  'ZFY', 'EIF1AY', 'KDM5D', 'NLGN4Y', 'TXLNGY', 'TMSB4Y'}
MHC1_CONFOUNDED = {'HLA-A', 'HLA-B'}   # haplotype-confounded (post-rename)

# === KEY INTERACTIONS (T cell <-> macrophage axes) ===
KEY_INTERACTIONS = [
    ('IFNG', 'IFNGR1'), ('IFNG', 'IFNGR2'),
    ('CD274', 'PDCD1'),
    ('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'), ('CXCL11', 'CXCR3'),
    ('HLA-DRA', 'CD4'), ('HLA-DRB1', 'CD4'), ('CD74', 'CD4'),
    ('TNF', 'TNFRSF1A'), ('TNF', 'TNFRSF1B'),
    ('CD80', 'CD28'), ('CD86', 'CD28'), ('CD80', 'CTLA4'), ('CD86', 'CTLA4'),
]


def wrap(s, width=34):
    s = str(s)
    return '\n'.join(textwrap.wrap(s, width=width)) if len(s) > width else s


def complex_members(complex_str):
    """LIANA joins subunits with '_' (e.g. ITGAX_ITGB2). Exact membership test
    instead of substring, so 'CD4' no longer matches CD40/CD44."""
    return set(str(complex_str).split('_'))


def deg_safe_name(celltype):
    """Reproduce Phase 3's comparison-name mangling so we can find its CSVs."""
    return (celltype.replace('Macrophages: ', 'Mac_').replace('Monocytes: ', 'Mono_')
            .replace('T cells: ', 'T_').replace('DCs: ', 'DC_')
            .replace(' ', '_').replace('-', '_').replace('+', 'pos').replace(':', ''))


# === LOAD ===
print("Loading annotated data...")
adata = sc.read_h5ad(ADATA_IN)
adata.obs[GROUP_COL] = adata.obs[GROUP_COL].astype(str)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

n_animals = adata.obs.groupby(GROUP_COL, observed=True)['animal_id'].nunique().to_dict()
print(f"  Animals per group: {n_animals}")
UNDERPOWERED = any(v < 3 for v in n_animals.values())
if UNDERPOWERED:
    print("  *** n<3 per arm: all differentials are EXPLORATORY, sex-confounded ***")

for patt in ('liana_*.csv', 'key_*.csv', 'differential_*.csv', 'convergent_*.csv',
             'dotplot_*.p*', 'dumbbell_*.p*', 'convergence_*.p*', 'celltype_*.csv',
             'key_gene_detection.csv'):
    for f in glob.glob(os.path.join(OUT_DIR, patt)):
        os.remove(f)
print(f"  Cleared prior Phase 4 outputs in {OUT_DIR}")

try:
    import liana as li
    print(f"  LIANA version: {li.__version__}")
except ImportError:
    raise ImportError("LIANA not installed. Run: pip install liana --break-system-packages")


# %% Cell 2 — Prepare AnnData for LIANA (SYMMETRIC cell-type gate)
print("\n" + "=" * 80)
print("PREPARING DATA FOR LIANA")
print("=" * 80)

ct_mask = adata.obs[CELLTYPE_COL].isin(CELLTYPES_TO_INCLUDE)
adata_liana = adata[ct_mask].copy()
print(f"\n  Candidate cell types: {len(CELLTYPES_TO_INCLUDE)}  "
      f"({adata_liana.n_obs:,} cells from {adata.n_obs:,})")

per_bin = (adata_liana.obs.groupby([GROUP_COL, CELLTYPE_COL], observed=True)
           .size().reset_index(name='n_cells'))
per_bin_wide = (per_bin.pivot_table(index=CELLTYPE_COL, columns=GROUP_COL,
                                    values='n_cells', fill_value=0)
                .reindex(columns=GROUP_ORDER, fill_value=0))
print(f"\n  Cell counts per group × celltype:")
print(per_bin_wide.to_string())

# --- SYMMETRIC GATE: must clear the floor in EVERY group ---
keep_ct, drop_ct = [], []
for ct in CELLTYPES_TO_INCLUDE:
    if ct not in per_bin_wide.index:
        drop_ct.append((ct, 'absent'))
        continue
    counts = {g: int(per_bin_wide.loc[ct, g]) for g in GROUP_ORDER}
    if all(v >= MIN_CELLS_PER_GROUP for v in counts.values()):
        keep_ct.append(ct)
    else:
        drop_ct.append((ct, ', '.join(f"{g}={v}" for g, v in counts.items())))

print(f"\n  SYMMETRIC GATE (>= {MIN_CELLS_PER_GROUP} cells in BOTH groups):")
print(f"    KEPT ({len(keep_ct)}): {keep_ct}")
if drop_ct:
    print(f"    DROPPED ({len(drop_ct)}) — excluded from BOTH runs so the")
    print(f"    differential stays comparable and no one-sided phantom edges appear:")
    for ct, why in drop_ct:
        print(f"      {ct:32s} {why}")
per_bin_wide.assign(kept=[ct in keep_ct for ct in per_bin_wide.index]) \
    .to_csv(os.path.join(OUT_DIR, 'celltype_counts_per_bin.csv'))

if len(keep_ct) < 2:
    raise RuntimeError("Fewer than 2 cell types survive the symmetric gate.")

adata_liana = adata_liana[adata_liana.obs[CELLTYPE_COL].isin(keep_ct)].copy()
adata_liana.obs[CELLTYPE_COL] = adata_liana.obs[CELLTYPE_COL].astype(str)
print(f"\n  Analysis set: {adata_liana.n_obs:,} cells across {len(keep_ct)} cell types")

# --- Rename MAMU -> HLA (in var_names and .raw) ---
print(f"\n  Renaming MAMU -> HLA for resource compatibility...")
rename_map = {m: h for m, h in MAMU_TO_HLA.items() if m in adata_liana.var_names}
for m, h in rename_map.items():
    print(f"    {m} -> {h}")
if rename_map:
    adata_liana.var_names = [rename_map.get(g, g) for g in adata_liana.var_names]
    if adata_liana.raw is not None:
        raw_adata = adata_liana.raw.to_adata()
        raw_adata.var_names = [rename_map.get(g, g) for g in raw_adata.var_names]
        adata_liana.raw = raw_adata
    print(f"  Renamed {len(rename_map)} genes.")
else:
    print("  No MAMU genes found to rename.")

# --- Key interaction gene presence + resource check ---
all_key_genes = sorted({g for pair in KEY_INTERACTIONS for g in pair})
present_key = [g for g in all_key_genes if g in adata_liana.var_names]
missing_key = [g for g in all_key_genes if g not in adata_liana.var_names]
print(f"\n  Key interaction genes present: {len(present_key)}/{len(all_key_genes)}")
if missing_key:
    print(f"    Missing (these axes cannot score): {missing_key}")

resource = li.rs.select_resource(RESOURCE_NAME)
resource_pairs = set(zip(resource['ligand'].values, resource['receptor'].values))
print(f"\n  LIANA '{RESOURCE_NAME}' resource: {len(resource)} L-R pairs")
for lig, rec in KEY_INTERACTIONS:
    in_res = (lig, rec) in resource_pairs
    both = lig in adata_liana.var_names and rec in adata_liana.var_names
    status = ("OK" if (in_res and both) else
              "NOT IN RESOURCE" if not in_res else
              f"GENE MISSING: {lig if lig not in adata_liana.var_names else rec}")
    print(f"    {lig:12s} -> {rec:12s}  [{status}]")


# %% Cell 2b — Key-gene detection diagnostic (does EXPR_PROP silence our axes?)
print("\n" + "=" * 80)
print(f"KEY-GENE DETECTION vs EXPR_PROP={EXPR_PROP}")
print("=" * 80)
print("  Fraction of cells expressing each key gene, per cell type x group.")
print("  Values below EXPR_PROP are dropped by LIANA before scoring.\n")

det_rows = []
src = adata_liana.raw.to_adata() if adata_liana.raw is not None else adata_liana
for g in present_key:
    if g not in src.var_names:
        continue
    x = src[:, g].X
    v = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    s = pd.Series(v > 0, index=adata_liana.obs_names)
    for ct in keep_ct:
        for grp in GROUP_ORDER:
            m = ((adata_liana.obs[CELLTYPE_COL] == ct) &
                 (adata_liana.obs[GROUP_COL] == grp)).values
            if m.sum() == 0:
                continue
            det_rows.append({'gene': g, 'celltype': ct, 'group': grp,
                             'detection': round(float(s.values[m].mean()), 3),
                             'n_cells': int(m.sum())})
det = pd.DataFrame(det_rows)
if not det.empty:
    det['below_expr_prop'] = det['detection'] < EXPR_PROP
    det.to_csv(os.path.join(OUT_DIR, 'key_gene_detection.csv'), index=False)
    silenced = (det.groupby('gene')['below_expr_prop'].all()
                .pipe(lambda s: s[s].index.tolist()))
    partial = (det.groupby('gene')['below_expr_prop'].any()
               .pipe(lambda s: s[s].index.tolist()))
    partial = [g for g in partial if g not in silenced]
    print(f"  Below EXPR_PROP in EVERY bin (axis effectively dead): {silenced}")
    print(f"  Below EXPR_PROP in SOME bins (asymmetric scoring risk): {partial}")
    print(f"\n  Full table -> key_gene_detection.csv")
    print("  If PDCD1 / CXCR3 appear above, lowering EXPR_PROP to 0.05 is worth a test.")


# %% Cell 3 — Run LIANA per group (2 runs, identical cell-type set)
print("\n" + "=" * 80)
print("RUNNING LIANA RANK AGGREGATE (per group)")
print("=" * 80)

groups = [g for g in GROUP_ORDER if g in adata_liana.obs[GROUP_COL].unique()]
liana_results = {}
run_summary = []

for group in groups:
    sub = adata_liana[adata_liana.obs[GROUP_COL] == group].copy()
    n_cells = sub.n_obs
    ct_counts = sub.obs[CELLTYPE_COL].value_counts()
    print(f"\n  --- {group} ---")
    print(f"  Cells: {n_cells:,}  |  cell types: {len(ct_counts)} (symmetric set)")

    if n_cells < 100:
        print(f"  *** SKIPPING: only {n_cells} cells ***")
        run_summary.append({'group': group, 'n_cells': n_cells,
                            'n_celltypes': len(ct_counts), 'n_interactions': 0,
                            'n_significant': 0, 'status': 'SKIPPED'})
        continue
    try:
        li.mt.rank_aggregate(
            sub, groupby=CELLTYPE_COL, resource_name=RESOURCE_NAME,
            expr_prop=EXPR_PROP, min_cells=MIN_CELLS, n_perms=N_PERMS,
            n_jobs=N_JOBS, use_raw=True, verbose=True, inplace=True,
            key_added='liana_res',
        )
        res = sub.uns['liana_res'].copy()
        res['group'] = group
        liana_results[group] = res
        n_sig = int((res['specificity_rank'] <= SPEC_SIG).sum()) \
            if 'specificity_rank' in res.columns else 0
        print(f"  Results: {len(res):,} interactions, {n_sig:,} significant "
              f"(spec_rank <= {SPEC_SIG})")
        run_summary.append({'group': group, 'n_cells': n_cells,
                            'n_celltypes': len(ct_counts), 'n_interactions': len(res),
                            'n_significant': n_sig, 'status': 'OK'})
        res.to_csv(os.path.join(OUT_DIR, f'liana_results_{group}.csv'), index=False)
    except Exception as e:
        print(f"  *** ERROR: {e} ***")
        run_summary.append({'group': group, 'n_cells': n_cells,
                            'n_celltypes': len(ct_counts), 'n_interactions': 0,
                            'n_significant': 0, 'status': f'ERROR: {e}'})

df_runs = pd.DataFrame(run_summary)
print(f"\n--- Run Summary ---")
print(df_runs.to_string(index=False))
df_runs.to_csv(os.path.join(OUT_DIR, 'liana_run_summary.csv'), index=False)

df_all = pd.concat(liana_results.values(), ignore_index=True) if liana_results else pd.DataFrame()
if not df_all.empty:
    df_all.to_csv(os.path.join(OUT_DIR, 'liana_all_results.csv'), index=False)


# %% Cell 4 — Track key interactions (complex-aware exact matching)
print("\n" + "=" * 80)
print("KEY BIOLOGICAL INTERACTIONS — D1MT vs Untreated")
print("=" * 80)

df_key = pd.DataFrame()
if not df_all.empty:
    key_rows = []
    for group, res_df in liana_results.items():
        lig_sets = res_df['ligand_complex'].map(complex_members)
        rec_sets = res_df['receptor_complex'].map(complex_members)
        for lig, rec in KEY_INTERACTIONS:
            m = lig_sets.map(lambda s: lig in s) & rec_sets.map(lambda s: rec in s)
            for _, row in res_df[m.values].iterrows():
                key_rows.append({
                    'group': group, 'ligand': lig, 'receptor': rec,
                    'ligand_complex': row.get('ligand_complex', ''),
                    'receptor_complex': row.get('receptor_complex', ''),
                    'source': row['source'], 'target': row['target'],
                    'magnitude_rank': row.get('magnitude_rank', np.nan),
                    'specificity_rank': row.get('specificity_rank', np.nan),
                })
    if key_rows:
        df_key = pd.DataFrame(key_rows).sort_values(
            ['ligand', 'receptor', 'source', 'target', 'group'])
        df_key['mhc1_confounded'] = df_key['ligand'].isin(MHC1_CONFOUNDED) | \
                                    df_key['receptor'].isin(MHC1_CONFOUNDED)
        for (lig, rec), grp in df_key.groupby(['ligand', 'receptor']):
            print(f"\n  {lig} -> {rec}:")
            for _, row in grp.iterrows():
                flag = "*" if row['specificity_rank'] <= SPEC_SIG else " "
                print(f"    {flag} {row['group']:10s} | {row['source']:28s} -> "
                      f"{row['target']:20s} | mag={row['magnitude_rank']:.4f} "
                      f"spec={row['specificity_rank']:.4f}")
        df_key.to_csv(os.path.join(OUT_DIR, 'key_interactions_tracked.csv'), index=False)
        print(f"\n  ('*' = specificity_rank <= {SPEC_SIG})")
    else:
        print("  None of the key interactions found — check Cell 2 gene presence.")
else:
    print("  No LIANA results to analyze.")


# %% Cell 5 — Dotplots per group (mac <-> T cell, both directions)
print("\n" + "=" * 80)
print("DOTPLOTS — mac <-> T cell interactions per group")
print("=" * 80)

if not df_all.empty:
    mac_labels = [ct for ct in keep_ct if 'Macrophage' in ct]
    tc_labels = [ct for ct in keep_ct if 'T cells' in ct]
    if not mac_labels or not tc_labels:
        print("  No mac/T cell types survived the gate; skipping dotplots.")
    for group, res_df in liana_results.items():
        sub = adata_liana[adata_liana.obs[GROUP_COL] == group].copy()
        sub.uns['liana_res'] = res_df
        for src_l, tgt_l, tag in [(mac_labels, tc_labels, 'mac_to_tcell'),
                                  (tc_labels, mac_labels, 'tcell_to_mac')]:
            if not src_l or not tgt_l:
                continue
            try:
                p = li.pl.dotplot(
                    adata=sub, colour='magnitude_rank', size='specificity_rank',
                    inverse_size=True, inverse_colour=True,
                    source_labels=src_l, target_labels=tgt_l, top_n=20,
                    orderby='magnitude_rank', orderby_ascending=True,
                    figure_size=(12, 10),
                    filter_fun=lambda x: x['specificity_rank'] <= SPEC_SIG,
                )
                for ext in ('pdf', 'png'):
                    p.save(os.path.join(OUT_DIR, f'dotplot_{tag}_{group}.{ext}'),
                           dpi=300, verbose=False)
                print(f"  Saved: dotplot_{tag}_{group}")
            except Exception as e:
                print(f"  *** Dotplot {tag} {group} failed: {str(e)[:100]} ***")
else:
    print("  No results to plot.")


# %% Cell 6 — Differential interactions (significance-filtered, confound-flagged)
print("\n" + "=" * 80)
print("DIFFERENTIAL INTERACTIONS — D1MT vs Untreated")
print("=" * 80)

merged = pd.DataFrame()
if all(g in liana_results for g in ['D1MT', 'Untreated']):
    mc = ['ligand_complex', 'receptor_complex', 'source', 'target']
    keep = mc + ['magnitude_rank', 'specificity_rank']
    merged = liana_results['D1MT'][keep].merge(
        liana_results['Untreated'][keep], on=mc,
        suffixes=('_d1mt', '_untx'), how='inner')
    merged['mag_rank_diff'] = merged['magnitude_rank_d1mt'] - merged['magnitude_rank_untx']
    merged['sig_either'] = ((merged['specificity_rank_d1mt'] <= SPEC_SIG) |
                            (merged['specificity_rank_untx'] <= SPEC_SIG))

    lig_sets = merged['ligand_complex'].map(complex_members)
    rec_sets = merged['receptor_complex'].map(complex_members)
    merged['sex_confounded'] = [bool(l & SEX_CONFOUNDED or r & SEX_CONFOUNDED)
                                for l, r in zip(lig_sets, rec_sets)]
    merged['mhc1_confounded'] = [bool(l & MHC1_CONFOUNDED or r & MHC1_CONFOUNDED)
                                 for l, r in zip(lig_sets, rec_sets)]
    merged['stronger_in'] = np.where(merged['mag_rank_diff'] < 0, 'D1MT', 'Untreated')

    print(f"  Total comparable interactions: {len(merged):,}")
    print(f"    significant in >=1 group:      {int(merged['sig_either'].sum()):,}")
    print(f"    sex-confounded (flagged):      {int(merged['sex_confounded'].sum()):,}")
    print(f"    MHC-I haplotype (flagged):     {int(merged['mhc1_confounded'].sum()):,}")

    rep = merged[~merged['sex_confounded']]
    if REQUIRE_SIG_EITHER:
        rep = rep[rep['sig_either']]
    print(f"\n  Reportable after filters: {len(rep):,}")

    print(f"\n  Top {TOP_N_REPORT} STRONGER in D1MT (negative diff = stronger):")
    for _, row in rep.nsmallest(TOP_N_REPORT, 'mag_rank_diff').iterrows():
        mh = ' [MHC-I]' if row['mhc1_confounded'] else ''
        print(f"    {row['ligand_complex']:15s} -> {row['receptor_complex']:15s} | "
              f"{row['source']:26s} -> {row['target']:20s} | "
              f"diff={row['mag_rank_diff']:.4f}{mh}")
    print(f"\n  Top {TOP_N_REPORT} STRONGER in Untreated (positive diff):")
    for _, row in rep.nlargest(TOP_N_REPORT, 'mag_rank_diff').iterrows():
        mh = ' [MHC-I]' if row['mhc1_confounded'] else ''
        print(f"    {row['ligand_complex']:15s} -> {row['receptor_complex']:15s} | "
              f"{row['source']:26s} -> {row['target']:20s} | "
              f"diff={row['mag_rank_diff']:.4f}{mh}")

    merged.to_csv(os.path.join(OUT_DIR, 'differential_interactions.csv'), index=False)
    print(f"\n  Saved: differential_interactions.csv ({len(merged):,} rows, all flags kept)")

    # --- key-axis-only view of the differential ---
    key_pairs = set(KEY_INTERACTIONS)
    key_diff_rows = []
    for _, row in merged.iterrows():
        ls, rs = complex_members(row['ligand_complex']), complex_members(row['receptor_complex'])
        for lig, rec in key_pairs:
            if lig in ls and rec in rs:
                d = row.to_dict(); d['ligand'] = lig; d['receptor'] = rec
                key_diff_rows.append(d)
    df_key_diff = pd.DataFrame(key_diff_rows)
    if not df_key_diff.empty:
        df_key_diff.to_csv(os.path.join(OUT_DIR, 'differential_key_axes.csv'), index=False)
        print(f"  Saved: differential_key_axes.csv ({len(df_key_diff)} rows)")
else:
    print("  Need LIANA results from BOTH groups to compute the differential.")


# %% Cell 7 — DEG convergence (the honest signal filter)
print("\n" + "=" * 80)
print("DEG CONVERGENCE — interactions whose L/R is also a DEG")
print("=" * 80)

deg_by_ct = {}
for ct in keep_ct:
    p = os.path.join(DEG_DIR, f'deg_{deg_safe_name(ct)}_D1MT_v_Untx_sig.csv')
    if os.path.exists(p):
        d = pd.read_csv(p)
        # Phase 3 used MAMU-* symbols; LIANA uses HLA-*. Align before matching.
        d['names'] = d['names'].replace(MAMU_TO_HLA)
        deg_by_ct[ct] = dict(zip(d['names'], d['logfoldchanges']))
        print(f"  {ct:32s} {len(d):,} sig DEGs")
    else:
        print(f"  {ct:32s} no DEG file (cluster skipped in Phase 3)")

convergent = pd.DataFrame()
if deg_by_ct and not merged.empty:
    rows = []
    for _, row in merged.iterrows():
        if row['sex_confounded']:
            continue
        s_map = deg_by_ct.get(row['source'], {})
        t_map = deg_by_ct.get(row['target'], {})
        ls = complex_members(row['ligand_complex'])
        rs = complex_members(row['receptor_complex'])
        lig_hits = {g: s_map[g] for g in ls if g in s_map}
        rec_hits = {g: t_map[g] for g in rs if g in t_map}
        if not lig_hits and not rec_hits:
            continue
        lfcs = list(lig_hits.values()) + list(rec_hits.values())
        deg_dir = 'up_D1MT' if np.mean(lfcs) > 0 else 'down_D1MT'
        # concordant = DEG direction agrees with which group the edge is stronger in
        concordant = ((deg_dir == 'up_D1MT' and row['stronger_in'] == 'D1MT') or
                      (deg_dir == 'down_D1MT' and row['stronger_in'] == 'Untreated'))
        rows.append({
            'ligand_complex': row['ligand_complex'], 'receptor_complex': row['receptor_complex'],
            'source': row['source'], 'target': row['target'],
            'ligand_DEG_in_source': ';'.join(lig_hits) or None,
            'receptor_DEG_in_target': ';'.join(rec_hits) or None,
            'both_sides': bool(lig_hits) and bool(rec_hits),
            'mean_deg_log2fc': float(np.mean(lfcs)), 'deg_direction': deg_dir,
            'mag_rank_diff': row['mag_rank_diff'], 'stronger_in': row['stronger_in'],
            'sig_either': row['sig_either'], 'concordant': concordant,
            'mhc1_confounded': row['mhc1_confounded'],
        })
    convergent = pd.DataFrame(rows)
    if not convergent.empty:
        convergent = convergent.sort_values(
            ['concordant', 'both_sides', 'mag_rank_diff'], ascending=[False, False, True])
        convergent.to_csv(os.path.join(OUT_DIR, 'convergent_interactions.csv'), index=False)
        n_conc = int(convergent['concordant'].sum())
        n_both = int(convergent['both_sides'].sum())
        print(f"\n  Interactions with DEG support: {len(convergent):,}")
        print(f"    concordant (DEG direction matches edge direction): {n_conc:,}")
        print(f"    supported on BOTH sides (ligand AND receptor):     {n_both:,}")
        strong = convergent[convergent['concordant'] & convergent['sig_either']]
        print(f"\n  Strongest convergent evidence (concordant + significant), top {TOP_N_REPORT}:")
        if strong.empty:
            print("    none — no interaction is both DEG-concordant and specificity-significant.")
        else:
            for _, r in strong.head(TOP_N_REPORT).iterrows():
                mh = ' [MHC-I]' if r['mhc1_confounded'] else ''
                print(f"    {r['ligand_complex']:14s} -> {r['receptor_complex']:14s} | "
                      f"{r['source']:24s} -> {r['target']:18s} | "
                      f"stronger={r['stronger_in']:9s} lfc={r['mean_deg_log2fc']:+.2f}{mh}")
        print(f"\n  Saved: convergent_interactions.csv")
    else:
        print("  No interactions had DEG support on either side.")
else:
    print("  Skipped (need Phase 3 DEG outputs and a differential).")


# %% Cell 8 — Rerun figures
print("\n" + "=" * 80)
print("FIGURES")
print("=" * 80)


def save_fig(fig, stem):
    fig.savefig(os.path.join(OUT_DIR, f'{stem}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f'{stem}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {stem}")


# --- 8A: key-axis dumbbell, mac <-> T only ---
# x = -log10(magnitude_rank): further right = stronger interaction. Paired dots
# per group joined by a line, so direction and size of change read at a glance.
if not df_key.empty:
    kd = df_key.copy()
    kd['pair'] = kd['source'] + '  ->  ' + kd['target']
    is_mac_t = (
        (kd['source'].str.contains('Macrophage') & kd['target'].str.contains('T cells')) |
        (kd['source'].str.contains('T cells') & kd['target'].str.contains('Macrophage')))
    kd = kd[is_mac_t]
    if not kd.empty:
        kd['axis'] = kd['ligand'] + '->' + kd['receptor']
        kd['label'] = kd['axis'] + '  |  ' + kd['pair']
        wide = kd.pivot_table(index='label', columns='group', values='magnitude_rank',
                              aggfunc='min')
        for g in GROUP_ORDER:
            if g not in wide.columns:
                wide[g] = np.nan
        wide = wide.dropna(how='all')
        # keep rows where at least one group is reasonably specific
        spec = kd.pivot_table(index='label', columns='group', values='specificity_rank',
                              aggfunc='min').reindex(wide.index)
        keep_rows = (spec <= SPEC_SIG).any(axis=1)
        wide = wide[keep_rows.fillna(False)]
        if not wide.empty:
            score = -np.log10(wide.clip(lower=1e-4))
            score = score.assign(_delta=(score.get('D1MT', 0) - score.get('Untreated', 0))) \
                         .sort_values('_delta')
            fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.34 * len(score) + 1.4)))
            for i, (lab, r) in enumerate(score.iterrows()):
                u, d = r.get('Untreated', np.nan), r.get('D1MT', np.nan)
                if not (np.isnan(u) or np.isnan(d)):
                    ax.plot([u, d], [i, i], color='#999999', lw=1.2, zorder=1)
                ax.scatter(u, i, s=48, color=COLOR_UNTX, zorder=2,
                           label='Untreated' if i == 0 else None)
                ax.scatter(d, i, s=48, color=COLOR_D1MT, zorder=2,
                           label='D1MT' if i == 0 else None)
            ax.set_yticks(range(len(score)))
            ax.set_yticklabels(score.index, fontsize=7)
            ax.set_xlabel('-log10(magnitude_rank)   —   further right = stronger', fontsize=9)
            ax.set_title('Key T cell <-> macrophage axes\n'
                         'EXPLORATORY: n=1 per arm, sex-confounded',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, frameon=False, loc='lower right')
            for sp_ in ('top', 'right'):
                ax.spines[sp_].set_visible(False)
            fig.tight_layout()
            save_fig(fig, 'dumbbell_key_axes_mac_T')
        else:
            print("  Dumbbell skipped: no mac<->T key axis reached significance.")
    else:
        print("  Dumbbell skipped: no mac<->T key-axis rows.")
else:
    print("  Dumbbell skipped: no key interactions tracked.")

# --- 8B: convergence figure ---
if not convergent.empty:
    cv = convergent[convergent['sig_either']].copy()
    cv = cv.reindex(cv['mag_rank_diff'].abs().sort_values(ascending=False).index).head(25)
    if not cv.empty:
        cv['label'] = (cv['ligand_complex'] + '->' + cv['receptor_complex'] + '  |  ' +
                       cv['source'] + ' -> ' + cv['target'])
        fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.34 * len(cv) + 1.4)))
        cv = cv.iloc[::-1]
        colors = [COLOR_D1MT if s == 'D1MT' else COLOR_UNTX for s in cv['stronger_in']]
        edges = ['#000000' if c else 'white' for c in cv['concordant']]
        ax.barh(range(len(cv)), -cv['mag_rank_diff'], color=colors, alpha=0.85,
                edgecolor=edges, linewidth=1.1, zorder=2)
        ax.axvline(0, color='#333333', lw=1.0, zorder=3)
        ax.set_yticks(range(len(cv)))
        ax.set_yticklabels(cv['label'], fontsize=7)
        ax.set_xlabel('rank shift   <-- stronger in Untreated  |  stronger in D1MT -->',
                      fontsize=9)
        ax.set_title('DEG-supported interactions (black outline = DEG-concordant)\n'
                     'EXPLORATORY: n=1 per arm, sex-confounded',
                     fontsize=10, fontweight='bold')
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
        fig.tight_layout()
        save_fig(fig, 'convergence_deg_supported')
else:
    print("  Convergence figure skipped: no DEG-supported interactions.")


# %% Cell 9 — Summary
print("\n" + "=" * 80)
print("PHASE 4 COMPLETE — LIANA CELL-CELL COMMUNICATION")
print("=" * 80)
print(f"\n  Output directory: {OUT_DIR}")
print(f"  Groups run: {sum(1 for r in run_summary if r['status'] == 'OK')}/{len(run_summary)}")
print(f"  Cell types after symmetric gate: {len(keep_ct)} | "
      f"Resource: {RESOURCE_NAME} | Perms: {N_PERMS} | expr_prop: {EXPR_PROP}")
print(f"\n  Key outputs:")
print(f"    liana_results_<group>.csv        per-group full results")
print(f"    celltype_counts_per_bin.csv      counts + which passed the gate")
print(f"    key_gene_detection.csv           detection vs EXPR_PROP per bin")
print(f"    key_interactions_tracked.csv     T <-> mac axes, both groups")
print(f"    differential_interactions.csv    all comparable, with confound flags")
print(f"    differential_key_axes.csv        differential restricted to key axes")
print(f"    convergent_interactions.csv      DEG-supported interactions")
print(f"    dumbbell_key_axes_mac_T.*        key-axis change figure")
print(f"    convergence_deg_supported.*      DEG-supported change figure")
print(f"    dotplot_*_<group>.*              per-group interaction dotplots")
print(f"\n  Caveats (current two-sample run):")
print(f"    - n=1 per arm: every differential is one animal vs one animal.")
print(f"      EXPLORATORY only; no treatment claim is supportable.")
print(f"    - Sex is aliased with treatment; AR/Y-gene edges are flagged and")
print(f"      excluded from headline lists (see sex_confounded column).")
print(f"    - MHC-I (HLA-A/B) edges are haplotype-confounded; flagged, matching")
print(f"      the Phase 3 MHC-I exclusion.")
print(f"    - Cell types below {MIN_CELLS_PER_GROUP} cells in EITHER group were dropped from")
print(f"      BOTH runs, so no one-sided phantom edges enter the differential.")
