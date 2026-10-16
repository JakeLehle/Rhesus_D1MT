#!/usr/bin/env python3
"""
Phase 3 — DEG Analysis and Pathway Enrichment  (Rhesus D1MT Project)
====================================================================
Differential expression between D1MT and Untreated, per annotated cluster,
at the single week-15 BAL timepoint.

Design realities baked into the method:
  - The pipeline supports n animals per group, but the CURRENT two-sample run
    is n = 1 per arm (43109 Untreated, 43102 D1MT), and sex is perfectly aliased
    with treatment. So the cell-level Wilcoxon is EXPLORATORY, pseudoreplicated,
    and sex-confounded, and there is NO credible per-animal layer (Cell 5 needs
    >= MIN_ANIMALS/arm and self-skips below that). When a powered re-run supplies
    >= 3 animals/arm the credible layer activates automatically.
  - Gene-class exclusions from the DEG universe (dropped, not flagged), because
    at n=1/arm with a depth gap between the two animals these are confound /
    quality / noise signal, not treatment biology:
        MHC-I (MAMU-A/B)   haplotype confounder   (MHC-II MAMU-D* kept)
        Mitochondrial      KEG06_ locus tags      (MT quality)
        Ribosomal          RPL/RPS/RPLP/MRP/...    (translation / depth)
        Sex-linked         XIST + Y genes         (sex aliases group)
        Pathogen           SIV/Mtb                (noise, stashed in Phase 1)
    AR is intentionally NOT excluded: it is X-linked, expressed in both sexes
    (lower in females), and its sex signal appears in LIANA, not DEG.

Outputs (this run's dir is cleared at start so nothing stale can surface):
  deg_*_all.csv / deg_*_sig.csv   cell-level Wilcoxon
  credible_*.csv                  per-animal 4v4 consistency (only if powered)
  pathways/*.csv                  enrichment terms (up / down)
  enrichment_*.pdf/png            diverging enrichment bars (up right / down left)
  volcano_*.pdf/png               pathway-colored volcano per comparison
  dot_*.pdf/png                   a-priori curated dotplots

Run in Spyder (# %% cells) or standalone.  Output dir: analysis/phase3_deg/

Author: Jake Lehle / Kaushal Lab
Date: July 2026
"""

# %% Cell 1 — Load, clear outputs, exclude gene classes, define comparisons
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import textwrap
import glob
import shutil
import os
import warnings
warnings.filterwarnings('ignore')

# === PATHS ===
BASE_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT'
ADATA_PATH = os.path.join(BASE_DIR, 'analysis', 'adata_d1mt_annotated.h5ad')
OUT_DIR = os.path.join(BASE_DIR, 'analysis', 'phase3_deg')
os.makedirs(OUT_DIR, exist_ok=True)

# === COLUMN NAMES (from Phase 2) ===
GROUP_COL = 'group'
ANIMAL_COL = 'animal_id'
TIER2_COL = 'tier2_full'
TIER3_COL = 'tier3_state'

# === GROUP LABELS + colors ===
D1MT = 'D1MT'
UNTX = 'Untreated'
COLOR_D1MT = '#D32F2F'
COLOR_UNTX = '#1976D2'

# === THRESHOLDS ===
PADJ_THRESHOLD = 0.05
LOGFC_THRESHOLD = 0.25
MIN_CELLS = 30            # per group per cluster
MIN_ANIMALS = 3          # per group required to attempt per-animal consistency
CONSISTENCY_P = 0.10     # per-animal MWU flag (4v4 min two-sided p ~0.029)
ENRICH_MIN_GENES = 8     # min genes in a direction to attempt enrichment
ENRICH_TOP_N = 10        # terms per direction on the diverging plot
VOLCANO_TOP_PATHWAYS = 6 # up-pathways colored on the volcano

# === SEX-LINKED GENES (now DROPPED from the universe; sex aliases group) ===
# Placeholder list from project notes; refine as the rhesus Y annotation firms up.
SEX_LINKED = {'XIST', 'RPS4Y1', 'RPS4Y2', 'DDX3Y', 'UTY', 'USP9Y', 'ZFY',
              'EIF1AY', 'KDM5D', 'NLGN4Y', 'TXLNGY', 'TMSB4Y'}


def wrap(s, width=38):
    s = str(s)
    return '\n'.join(textwrap.wrap(s, width=width)) if len(s) > width else s


# === LOAD ===
print("Loading Phase 2 annotated data...")
adata = sc.read_h5ad(ADATA_PATH)
adata.obs[GROUP_COL] = adata.obs[GROUP_COL].astype(str)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  Groups: {sorted(adata.obs[GROUP_COL].unique())}")

# --- Clear this phase's prior outputs so nothing stale can be read or linger ---
for patt in ('deg_*.csv', 'credible_*.csv', '*_summary.csv', 'deg_universe_exclusions.csv',
             'dot_*.p*', 'enrichment_*.p*', 'volcano_*.p*'):
    for f in glob.glob(os.path.join(OUT_DIR, patt)):
        os.remove(f)
if os.path.isdir(os.path.join(OUT_DIR, 'pathways')):
    shutil.rmtree(os.path.join(OUT_DIR, 'pathways'))
print(f"  Cleared prior Phase 3 outputs in {OUT_DIR}")

# --- Gene-class exclusions from the DEG universe (dropped, not flagged) ---
vn = adata.var_names
mhc1_mask = vn.str.startswith('MAMU-A') | vn.str.startswith('MAMU-B')
mt_mask = vn.str.startswith('KEG06_')
# Ribosomal proteins only. Anchored patterns avoid catching RPS6K* kinases
# (e.g. RPS6KA5 is a signaling gene, NOT ribosomal, and must stay in).
ribo_mask = (vn.str.match(r'^RP[LS]\d+[A-Z]?$') | vn.str.match(r'^RPLP\d$') |
             vn.str.match(r'^MRP[LS]\d+[A-Z]?$') | vn.isin(['RPSA', 'FAU']))
sex_mask = vn.isin(SEX_LINKED)
if 'pathogen' in adata.var.columns:
    path_mask = adata.var['pathogen'].values.astype(bool)
else:
    path_mask = np.zeros(adata.n_vars, dtype=bool)

exclusion_classes = [
    ('MHC-I (MAMU-A/B)', np.asarray(mhc1_mask)),
    ('Mitochondrial (KEG06_)', np.asarray(mt_mask)),
    ('Ribosomal', np.asarray(ribo_mask)),
    ('Sex-linked', np.asarray(sex_mask)),
    ('Pathogen (SIV/Mtb)', np.asarray(path_mask)),
]
audit_rows = []
print("\n  Gene-class exclusions from DEG universe:")
for label, m in exclusion_classes:
    genes = vn[m].tolist()
    audit_rows += [{'gene': g, 'class': label} for g in genes]
    preview = genes if len(genes) <= 10 else genes[:10] + ['...']
    print(f"    {len(genes):>4}  {label:24s} {preview}")
pd.DataFrame(audit_rows).to_csv(os.path.join(OUT_DIR, 'deg_universe_exclusions.csv'), index=False)

drop_mask = np.zeros(adata.n_vars, dtype=bool)
for _, m in exclusion_classes:
    drop_mask |= m
adata = adata[:, ~drop_mask].copy()
print(f"  DEG gene universe: {adata.shape[1]:,} genes "
      f"(dropped {int(drop_mask.sum()):,}; full list in deg_universe_exclusions.csv)")

# --- Cluster sizes by group ---
print(f"\n  Cluster sizes (per group, week 15):")
for pop in sorted(adata.obs[TIER2_COL].unique()):
    n_d = int(((adata.obs[TIER2_COL] == pop) & (adata.obs[GROUP_COL] == D1MT)).sum())
    n_u = int(((adata.obs[TIER2_COL] == pop) & (adata.obs[GROUP_COL] == UNTX)).sum())
    flag = " ***" if (n_d < MIN_CELLS or n_u < MIN_CELLS) else ""
    print(f"    {pop:35s}  D1MT={n_d:>6,}  Untreated={n_u:>6,}{flag}")


# %% Cell 2 — DEG helper (cell-level Wilcoxon discovery layer)
print("\n" + "=" * 80)
print("DEG (cell-level Wilcoxon — EXPLORATORY / pseudoreplicated at n=1/arm)")
print("=" * 80)

all_deg = {}
deg_masks = {}   # comparison name -> boolean cluster mask (both groups), for Cell 5

def run_deg(adata_obj, mask_a, mask_b, name, description,
            test_label=D1MT, ref_label=UNTX):
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    print(f"\n--- {name} ---")
    print(f"  {description}")
    print(f"  {test_label}: {n_a:,} cells | {ref_label}: {n_b:,} cells")
    if n_a < MIN_CELLS or n_b < MIN_CELLS:
        print(f"  *** SKIPPED: <{MIN_CELLS} cells (test={n_a}, ref={n_b}) ***")
        all_deg[name] = None
        return None

    both = mask_a | mask_b
    sub = adata_obj[both].copy()
    sub.obs['deg_group'] = ref_label
    sub.obs.loc[mask_a[both], 'deg_group'] = test_label

    sc.tl.rank_genes_groups(sub, groupby='deg_group', groups=[test_label],
                            reference=ref_label, method='wilcoxon',
                            key_added='deg', use_raw=False)
    result = sc.get.rank_genes_groups_df(sub, group=test_label, key='deg')
    result['comparison'] = name
    result['test_label'] = test_label
    result['ref_label'] = ref_label
    result['n_test'] = n_a
    result['n_ref'] = n_b
    result['sex_linked'] = result['names'].isin(SEX_LINKED)  # ~all False now (dropped)

    sig = result[result['pvals_adj'] < PADJ_THRESHOLD]
    sig_up = sig[sig['logfoldchanges'] > 0]
    sig_down = sig[sig['logfoldchanges'] < 0]
    print(f"  Significant (padj<{PADJ_THRESHOLD}): {len(sig):,} "
          f"({len(sig_up)} up in {test_label}, {len(sig_down)} down)")
    if len(sig_up) > 0:
        print(f"  Top 5 up:   {sig_up.head(5)['names'].tolist()}")
    if len(sig_down) > 0:
        print(f"  Top 5 down: {sig_down.head(5)['names'].tolist()}")

    result.to_csv(os.path.join(OUT_DIR, f'deg_{name}_all.csv'), index=False)
    if len(sig) > 0:
        sig.to_csv(os.path.join(OUT_DIR, f'deg_{name}_sig.csv'), index=False)
    all_deg[name] = result
    deg_masks[name] = np.asarray(mask_a | mask_b)   # cluster cells, both groups
    return result


# %% Cell 3 — D1MT vs Untreated per cluster (single timepoint)
print("\n" + "=" * 80)
print("D1MT vs UNTREATED — per cluster")
print("=" * 80)

# Tier 2 clusters
for pop in sorted(adata.obs[TIER2_COL].unique()):
    if pop == 'Unassigned':
        continue
    safe = (pop.replace('Macrophages: ', 'Mac_').replace('Monocytes: ', 'Mono_')
               .replace('T cells: ', 'T_').replace('DCs: ', 'DC_')
               .replace(' ', '_').replace('-', '_').replace('+', 'pos').replace(':', ''))
    pop_mask = adata.obs[TIER2_COL] == pop
    run_deg(adata, pop_mask & (adata.obs[GROUP_COL] == D1MT),
            pop_mask & (adata.obs[GROUP_COL] == UNTX),
            f'{safe}_D1MT_v_Untx', f'{pop}: D1MT vs Untreated')

# Tier 3 T cell states (CD4/CD8 Effector/Naive)
for state in sorted(adata.obs[TIER3_COL].unique()):
    if state == 'N/A':
        continue
    safe = state.replace(' ', '_').replace('-', '_')
    st_mask = adata.obs[TIER3_COL] == state
    run_deg(adata, st_mask & (adata.obs[GROUP_COL] == D1MT),
            st_mask & (adata.obs[GROUP_COL] == UNTX),
            f'T3_{safe}_D1MT_v_Untx', f'{state}: D1MT vs Untreated')


# %% Cell 4 — DEG summary table
print("\n" + "=" * 80)
print("DEG SUMMARY (cell-level; exploratory at n=1/arm)")
print("=" * 80)

summary_rows = []
for name, res in all_deg.items():
    if res is None:
        summary_rows.append({'comparison': name, 'status': 'SKIPPED',
                             'n_sig': 0, 'n_up': 0, 'n_down': 0})
    else:
        sig = res[res['pvals_adj'] < PADJ_THRESHOLD]
        summary_rows.append({
            'comparison': name, 'status': 'OK',
            'n_test': int(res['n_test'].iloc[0]), 'n_ref': int(res['n_ref'].iloc[0]),
            'n_sig': len(sig),
            'n_up': int((sig['logfoldchanges'] > 0).sum()),
            'n_down': int((sig['logfoldchanges'] < 0).sum()),
        })
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(OUT_DIR, 'deg_summary.csv'), index=False)


# %% Cell 5 — Per-animal pseudobulk consistency (credible layer; powered runs only)
# =========================================================================
# Activates only with >= MIN_ANIMALS animals per arm. At n=1/arm it self-skips
# for every comparison and there is NO credible layer: the cell-level Wilcoxon
# is the only output and is exploratory. When powered, each cell-level-sig gene
# is aggregated to a per-animal mean, run through a 4v4 Mann-Whitney, and scored
# k/4 for cross-animal consistency; "credible" = per-animal p < CONSISTENCY_P AND
# all test animals on the expected side AND not sex-linked.
# =========================================================================
print("\n" + "=" * 80)
print("PER-ANIMAL PSEUDOBULK CONSISTENCY (credible DEG layer)")
print("=" * 80)

n_test_animals = adata.obs.loc[adata.obs[GROUP_COL] == D1MT, ANIMAL_COL].nunique()
n_ref_animals = adata.obs.loc[adata.obs[GROUP_COL] == UNTX, ANIMAL_COL].nunique()
CREDIBLE_POSSIBLE = (n_test_animals >= MIN_ANIMALS and n_ref_animals >= MIN_ANIMALS)
print(f"  Animals: D1MT={n_test_animals}, Untreated={n_ref_animals} "
      f"(need >= {MIN_ANIMALS}/arm for the credible layer)")
if not CREDIBLE_POSSIBLE:
    print("  *** NO CREDIBLE LAYER at this n. Cell-level Wilcoxon is the only")
    print("      layer and is EXPLORATORY, pseudoreplicated, and sex-confounded. ***")


def per_animal_consistency(cluster_mask, sig_genes, test_label=D1MT, ref_label=UNTX):
    sub = adata[cluster_mask].copy()
    genes = [g for g in sig_genes if g in sub.var_names]
    if not genes:
        return pd.DataFrame()
    X = sub[:, genes].X
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    df = pd.DataFrame(X, columns=genes)
    df[ANIMAL_COL] = sub.obs[ANIMAL_COL].values
    df[GROUP_COL] = sub.obs[GROUP_COL].values
    per_an = df.groupby([GROUP_COL, ANIMAL_COL], observed=True)[genes].mean()
    test_means = per_an.xs(test_label, level=GROUP_COL) if test_label in \
        per_an.index.get_level_values(GROUP_COL) else pd.DataFrame()
    ref_means = per_an.xs(ref_label, level=GROUP_COL) if ref_label in \
        per_an.index.get_level_values(GROUP_COL) else pd.DataFrame()
    if test_means.shape[0] < MIN_ANIMALS or ref_means.shape[0] < MIN_ANIMALS:
        return pd.DataFrame()

    rows = []
    ref_median = ref_means.median(axis=0)
    ref_max = ref_means.max(axis=0)
    ref_min = ref_means.min(axis=0)
    n_test = test_means.shape[0]
    for g in genes:
        tvals = test_means[g].values
        rvals = ref_means[g].values
        try:
            u, p = mannwhitneyu(tvals, rvals, alternative='two-sided')
        except ValueError:
            u, p = np.nan, np.nan
        up = np.mean(tvals) >= np.median(rvals)
        if up:
            k = int(np.sum(tvals > ref_max[g]))
            k_median = int(np.sum(tvals > ref_median[g]))
        else:
            k = int(np.sum(tvals < ref_min[g]))
            k_median = int(np.sum(tvals < ref_median[g]))
        rows.append({
            'gene': g, 'direction': 'up' if up else 'down',
            'mean_test': float(np.mean(tvals)), 'mean_ref': float(np.mean(rvals)),
            'per_animal_mwu_p': p,
            'k_beyond_ref_range': k, 'n_test_animals': n_test,
            'k_beyond_ref_median': k_median,
            'sex_linked': g in SEX_LINKED,
        })
    out = pd.DataFrame(rows)
    out['credible'] = (out['per_animal_mwu_p'] < CONSISTENCY_P) & \
                      (out['k_beyond_ref_median'] == n_test) & (~out['sex_linked'])
    return out


consistency_summary = []
if CREDIBLE_POSSIBLE:
    for name, res in all_deg.items():
        if res is None:
            continue
        sig = res[res['pvals_adj'] < PADJ_THRESHOLD]
        if len(sig) == 0:
            continue
        cons = per_animal_consistency(deg_masks[name], sig['names'].tolist())
        if cons.empty:
            print(f"  {name}: insufficient animals for per-animal test — skipped")
            continue
        n_cred = int(cons['credible'].sum())
        print(f"  {name}: {len(sig):,} cell-level sig -> {n_cred:,} credible "
              f"({100*n_cred/max(len(sig),1):.0f}% survive)")
        cons.sort_values(['credible', 'per_animal_mwu_p'], ascending=[False, True]) \
            .to_csv(os.path.join(OUT_DIR, f'credible_{name}.csv'), index=False)
        consistency_summary.append({'comparison': name, 'n_cell_level_sig': len(sig),
                                    'n_credible': n_cred})
    if consistency_summary:
        pd.DataFrame(consistency_summary).to_csv(
            os.path.join(OUT_DIR, 'credible_deg_summary.csv'), index=False)
        print("\n  Saved credible_*.csv + credible_deg_summary.csv — trust the credible set.")
else:
    print("  Skipping per-animal consistency (needs a powered re-run).")


# %% Cell 6 — Pathway enrichment (exploratory at n=1/arm) + diverging bar plots
print("\n" + "=" * 80)
print("PATHWAY ENRICHMENT + diverging bars (up -> D1MT, down -> Untreated)")
print("=" * 80)

try:
    import gseapy as gp
    import time
    GSEA_OK = True
except ImportError:
    print("  gseapy not available — skipping enrichment")
    GSEA_OK = False

LIBRARIES = ['KEGG_2021_Human', 'GO_Biological_Process_2023']
enrichment_store = {}   # name -> {'up': terms_df, 'down': terms_df} (incl 'Genes'), for volcano


def get_enrichment_gene_sets(name):
    """Return (up_genes, down_genes, layer_label). Uses the credible set when a
    powered run produced one, else the cleaned cell-level significant genes."""
    if CREDIBLE_POSSIBLE:
        cred_path = os.path.join(OUT_DIR, f'credible_{name}.csv')
        if os.path.exists(cred_path):
            cred = pd.read_csv(cred_path)
            cred = cred[cred['credible']]
            return (cred[cred['direction'] == 'up']['gene'].tolist(),
                    cred[cred['direction'] == 'down']['gene'].tolist(), 'credible per-animal')
    res = all_deg.get(name)
    if res is None:
        return [], [], 'none'
    sig = res[res['pvals_adj'] < PADJ_THRESHOLD]
    up = sig[sig['logfoldchanges'] > LOGFC_THRESHOLD]['names'].tolist()
    down = sig[sig['logfoldchanges'] < -LOGFC_THRESHOLD]['names'].tolist()
    return up, down, 'EXPLORATORY cell-level (n=1/arm)'


def plot_enrichment_diverging(up_terms, down_terms, name, layer_label, top_n=ENRICH_TOP_N):
    """One diverging bar panel: up (D1MT) extends right, down (Untreated) left;
    bar length = -log10(adj p). Combines KEGG + GO, top_n per direction."""
    up_top = up_terms.nsmallest(top_n, 'Adjusted P-value') if len(up_terms) else up_terms
    down_top = down_terms.nsmallest(top_n, 'Adjusted P-value') if len(down_terms) else down_terms
    if len(up_top) == 0 and len(down_top) == 0:
        return False
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.55 * (len(up_top) + len(down_top)) + 1.2)))
    y = 0
    all_scores = []
    for _, r in down_top.iloc[::-1].iterrows():
        s = -np.log10(max(r['Adjusted P-value'], 1e-300))
        ax.barh(y, -s, color=COLOR_UNTX, alpha=0.85, edgecolor='white', height=0.72, zorder=2)
        ax.text(-s - 0.15, y, wrap(r['Term'], 40), ha='right', va='center', fontsize=7)
        all_scores.append(s); y += 1
    for _, r in up_top.iterrows():
        s = -np.log10(max(r['Adjusted P-value'], 1e-300))
        ax.barh(y, s, color=COLOR_D1MT, alpha=0.85, edgecolor='white', height=0.72, zorder=2)
        ax.text(s + 0.15, y, wrap(r['Term'], 40), ha='left', va='center', fontsize=7)
        all_scores.append(s); y += 1
    ax.axvline(0, color='#333333', linewidth=1.0, zorder=3)
    ax.set_yticks([])
    xmax = max(all_scores) if all_scores else 1.0
    ax.set_xlim(-xmax * 2.1, xmax * 2.1)
    ax.set_xlabel('-log10(adj p)     <-- enriched in Untreated  |  enriched in D1MT -->',
                  fontsize=9)
    ax.set_title(f'{name}\nPathway enrichment ({layer_label})', fontsize=10, fontweight='bold')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'enrichment_{name}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f'enrichment_{name}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    return True


if GSEA_OK:
    pathway_dir = os.path.join(OUT_DIR, 'pathways')
    os.makedirs(pathway_dir, exist_ok=True)

    def enrichr_retry(gene_list, lib, max_retries=3, base_delay=30):
        for attempt in range(max_retries):
            try:
                return gp.enrichr(gene_list=gene_list, gene_sets=lib,
                                  organism='human', outdir=None, no_plot=True)
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                else:
                    raise e

    def run_enrichr_combined(gene_list):
        frames = []
        for lib in LIBRARIES:
            try:
                enr = enrichr_retry(gene_list, lib)
                r = enr.results
                r = r[r['Adjusted P-value'] < 0.05][['Term', 'Adjusted P-value', 'Genes']].copy()
                r['library'] = lib
                frames.append(r)
            except Exception as e:
                print(f"      {lib}: {str(e)[:70]}")
            time.sleep(2)
        cols = ['Term', 'Adjusted P-value', 'Genes', 'library']
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    for name, res in all_deg.items():
        if res is None:
            continue
        up, down, layer = get_enrichment_gene_sets(name)
        if len(up) < ENRICH_MIN_GENES and len(down) < ENRICH_MIN_GENES:
            print(f"  {name}: too few genes for enrichment (up={len(up)}, down={len(down)})")
            enrichment_store[name] = {'up': pd.DataFrame(), 'down': pd.DataFrame()}
            continue
        print(f"\n  --- {name} [{layer}]  up={len(up)}  down={len(down)} ---")
        up_terms = run_enrichr_combined(up) if len(up) >= ENRICH_MIN_GENES else pd.DataFrame()
        down_terms = run_enrichr_combined(down) if len(down) >= ENRICH_MIN_GENES else pd.DataFrame()
        enrichment_store[name] = {'up': up_terms, 'down': down_terms}
        if len(up_terms):
            up_terms.assign(direction='up').to_csv(
                os.path.join(pathway_dir, f'{name}_up.csv'), index=False)
        if len(down_terms):
            down_terms.assign(direction='down').to_csv(
                os.path.join(pathway_dir, f'{name}_down.csv'), index=False)
        if plot_enrichment_diverging(up_terms, down_terms, name, layer):
            print(f"      Saved: enrichment_{name}")
        for lbl, tdf in [('up  (D1MT)', up_terms), ('down(Untx)', down_terms)]:
            if len(tdf):
                for _, row in tdf.nsmallest(3, 'Adjusted P-value').iterrows():
                    print(f"        {lbl}: {row['Adjusted P-value']:.2e}  {str(row['Term'])[:52]}")
else:
    print("  Enrichment skipped; volcano will render without pathway colors.")


# %% Cell 7 — Pathway-colored volcano per comparison (first-look)
print("\n" + "=" * 80)
print("VOLCANO PLOTS — significant up genes colored by enriched pathway")
print("=" * 80)


def make_volcano(name, top_pathways=VOLCANO_TOP_PATHWAYS):
    res = all_deg.get(name)
    if res is None or res.empty:
        return
    df = res.copy()
    df['neglog10p'] = -np.log10(df['pvals_adj'].clip(lower=1e-300))
    df['sig'] = (df['pvals_adj'] < PADJ_THRESHOLD) & (df['logfoldchanges'].abs() > LOGFC_THRESHOLD)
    df['is_up'] = df['logfoldchanges'] > 0

    # assign sig-up genes to their most-significant enriched up-pathway
    gene2term, palette = {}, {}
    up_terms = enrichment_store.get(name, {}).get('up', pd.DataFrame())
    if len(up_terms):
        top_terms = up_terms.nsmallest(top_pathways, 'Adjusted P-value')
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (_, tr) in enumerate(top_terms.iterrows()):
            term = tr['Term']
            palette[term] = colors[i % 10]
            for g in str(tr.get('Genes', '')).split(';'):
                g = g.strip()
                if g and g not in gene2term:   # most-significant term wins
                    gene2term[g] = term

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ns = df[~df['sig']]
    ax.scatter(ns['logfoldchanges'], ns['neglog10p'], s=6, c='#d0d0d0',
               alpha=0.5, edgecolors='none', zorder=1)
    sd = df[df['sig'] & ~df['is_up']]
    ax.scatter(sd['logfoldchanges'], sd['neglog10p'], s=11, c='#90a4c8',
               alpha=0.75, edgecolors='none', zorder=2, label='sig down (Untx)')
    su = df[df['sig'] & df['is_up']]
    su_other = su[~su['names'].isin(gene2term.keys())]
    ax.scatter(su_other['logfoldchanges'], su_other['neglog10p'], s=11, c='#e6a0a0',
               alpha=0.75, edgecolors='none', zorder=2, label='sig up (no pathway)')
    for term, col in palette.items():
        gg = su[su['names'].map(lambda x: gene2term.get(x)) == term]
        if len(gg):
            ax.scatter(gg['logfoldchanges'], gg['neglog10p'], s=26, color=col,
                       alpha=0.95, edgecolors='white', linewidths=0.3, zorder=3,
                       label=wrap(term, 28))
    ax.axhline(-np.log10(PADJ_THRESHOLD), ls='--', c='#888888', lw=0.8)
    ax.axvline(LOGFC_THRESHOLD, ls='--', c='#888888', lw=0.8)
    ax.axvline(-LOGFC_THRESHOLD, ls='--', c='#888888', lw=0.8)
    # robust x-limits so a single artifact gene can't stretch the plot
    if df['sig'].any():
        xcap = max(3.0, min(12.0, float(np.nanpercentile(df.loc[df['sig'], 'logfoldchanges'].abs(), 99))))
    else:
        xcap = 3.0
    ax.set_xlim(-xcap * 1.1, xcap * 1.1)
    ax.set_xlabel('log2 fold change  (D1MT vs Untreated)', fontsize=10)
    ax.set_ylabel('-log10 adjusted p', fontsize=10)
    ax.set_title(f'{name}\nEXPLORATORY: cell-level, n=1/arm, sex/MT/ribo excluded',
                 fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              frameon=False, title='up pathways', title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'volcano_{name}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f'volcano_{name}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


made = 0
for name, res in all_deg.items():
    if res is None:
        continue
    if (res['pvals_adj'] < PADJ_THRESHOLD).sum() == 0:
        continue
    make_volcano(name)
    made += 1
print(f"  Saved {made} volcano plot(s).")


# %% Cell 8 — Curated a-priori dotplots (D1MT vs Untreated)
print("\n" + "=" * 80)
print("CURATED DOTPLOTS — a-priori panels, D1MT vs Untreated")
print("=" * 80)

def valid_genes(gene_list):
    return [g for g in gene_list if g in adata.var_names]

def make_dotplot(sub, genes, groupby, title, filename, swap_axes=True):
    genes = valid_genes(genes)
    if len(genes) < 2 or sub.n_obs < 10:
        print(f"    Skipped {filename}: {len(genes)} genes, {sub.n_obs} cells")
        return
    try:
        axd = sc.pl.dotplot(sub, var_names=genes, groupby=groupby, standard_scale='var',
                            swap_axes=swap_axes, title=title, show=False)
        fig = next(iter(axd.values())).get_figure()
        fig.savefig(os.path.join(OUT_DIR, f'{filename}.pdf'), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(OUT_DIR, f'{filename}.png'), bbox_inches='tight', dpi=300)
        print(f"    Saved: {filename}")
        plt.close(fig)
    except Exception as e:
        print(f"    Failed {filename}: {e}")
        plt.close('all')

# NOTE: antigen-presentation genes use MAMU-* symbols in this reference (HLA-*
# only exists after the LIANA rename), so those are given as MAMU-DRA/DRB1.
CD4_PANEL = ['TBX21', 'IFNG', 'TNF', 'IL2', 'STAT4',            # Th1
             'RORC', 'IL17A', 'IL23R', 'CCR6',                  # Th17
             'CD69', 'CD44', 'IL2RA', 'ICOS',                   # activation
             'PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'CTLA4', 'TOX',  # exhaustion
             'MKI67', 'TOP2A',                                  # proliferation
             'BCL2', 'IL7R']                                    # survival
cd4_sub = adata[adata.obs[TIER3_COL] == 'CD4 Effector'].copy()
cd4_sub.obs[GROUP_COL] = pd.Categorical(cd4_sub.obs[GROUP_COL], categories=[UNTX, D1MT])
make_dotplot(cd4_sub, CD4_PANEL, GROUP_COL,
             'CD4 Effector — D1MT vs Untreated', 'dot_cd4_effector_panel')

MAC_PANEL = ['MAMU-DRA', 'MAMU-DRB1', 'CD74', 'CD80', 'CD86',   # antigen presentation
             'TNF', 'IL1B', 'IL6', 'CXCL8', 'CCL2', 'CCL3',     # inflammatory
             'IDO1', 'CD274', 'SOD2',                           # IDO / checkpoint
             'CXCL9', 'CXCL10', 'CXCL11', 'STAT1', 'IRF1',      # IFN response
             'CD163', 'MRC1', 'MARCO', 'PPARG']                 # homeostatic / M2
mac_sub = adata[adata.obs[TIER2_COL].str.contains('Macrophage', na=False)].copy()
if mac_sub.n_obs > 0:
    mac_sub.obs['subtype_group'] = (
        mac_sub.obs[TIER2_COL].astype(str).str.replace('Macrophages: ', '') +
        ' | ' + mac_sub.obs[GROUP_COL].astype(str))
    make_dotplot(mac_sub, MAC_PANEL, 'subtype_group',
                 'Macrophage Activation — subtype x group', 'dot_mac_activation')


# %% Cell 9 — Final summary
print("\n" + "=" * 80)
print("PHASE 3 COMPLETE — DEG (D1MT vs Untreated)")
print("=" * 80)
print(f"\n  Outputs in: {OUT_DIR}")
print(f"  Comparisons run: {len(all_deg)}")
print(f"\n  Layers:")
print(f"    deg_*_all.csv / deg_*_sig.csv   cell-level Wilcoxon (exploratory)")
print(f"    credible_*.csv                  per-animal 4v4 consistency (powered runs only)")
print(f"    pathways/*.csv                  enrichment terms (up / down)")
print(f"    enrichment_*.pdf/png            diverging enrichment bars")
print(f"    volcano_*.pdf/png               pathway-colored volcano")
print(f"    dot_*.pdf/png                   a-priori curated dotplots")
print(f"\n  Reminders (current two-sample run):")
print(f"    - n=1 per arm: cell-level Wilcoxon is EXPLORATORY and pseudoreplicated;")
print(f"      no credible per-animal layer is possible.")
print(f"    - Sex is aliased with treatment: XIST + Y genes DROPPED from the universe.")
print(f"    - MHC-I, mitochondrial (KEG06_), ribosomal, and pathogen genes DROPPED too")
print(f"      (see deg_universe_exclusions.csv).")
print(f"\n  Next: Phase 4 — LIANA (T cell <-> macrophage communication)")
