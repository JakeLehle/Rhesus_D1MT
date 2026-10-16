# %%
# ============================================================================
# marker_refinement_diagnostic.py   (supersedes marker_discovery_lymphoid.py)
#
# READ-ONLY. Uses the ACTUAL current Tier 1 / Tier 2 marker sets (copied from
# Phase2_annotation.py) to diagnose why the lymphoid populations mis-annotate,
# then evaluates provisional refined sets. Keeps the per-cell fuzzy-scoring
# architecture: we are fixing the marker LISTS, not stamping cluster labels.
#
# Three known failures this quantifies:
#   1. Tier1 T = [CD3D, CD3E] only -> both drop out in low-RNA lymphocytes ->
#      Unassigned (cluster 12).
#   2. Tier1 NK = [KLRC3, KLRD1, GNLY, PRF1, GZMB] -> GNLY/PRF1/GZMB are shared
#      cytotoxic genes -> CD8 effectors argmax to NK (cluster 16's 477 "NK").
#   3. Tier2 CD4 leads with CD4, which is a macrophage gene here -> CD4/CD8
#      T split is confounded.
#
# Nothing is written back to the object. Candidate sets are PROVISIONAL and
# parameterized; we finalize them together from this output.
# ============================================================================

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

# --- Cell 1: params + ACTUAL current marker sets + provisional candidates ---
ADATA_PATH = "/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT/analysis/adata_d1mt_annotated.h5ad"
OUTDIR     = "/master/jlehle/WORKING/SC/fastq/Rhesus_D1MT/analysis/phase2_annotation_diag"

LEIDEN_KEY = "leiden_diag"
LEIDEN_RES = 1.0
SEED       = 0
SCORE_THRESHOLD = 0.1     # matches Phase 2

LYMPHOID_CLUSTERS = ["12", "16", "17"]   # unassigned epicenter / T island / NK island
EXPECTED_SIZES = {"12": 629, "16": 3904, "17": 468}
T_CLUSTER, NK_CLUSTER = "16", "17"
MAC_REF_CLUSTER = "3"     # 99.8% macrophage; non-specificity reference
N_TOP = 40

# --- CURRENT sets (verbatim from Phase2_annotation.py) ---
CUR_TIER1 = {
    "T cells":     ["CD3D", "CD3E"],
    "Macrophages": ["CD163", "MRC1", "MARCO", "MERTK", "PPARG"],
    "Monocytes":   ["S100A8", "S100A9", "VCAN", "CAMP", "CD14", "LYZ"],
    "DCs":         ["FLT3", "CLEC9A", "CLEC10A", "IL3RA", "CLEC4C"],
    "NK cells":    ["KLRC3", "KLRD1", "GNLY", "PRF1", "GZMB"],
    "B cells":     ["CD79A", "MS4A1", "CD19", "BANK1"],
    "Mast cells":  ["CPA3", "KIT", "HPGD", "ENPP3", "FCER1A"],
    "Ciliated epithelial": ["DNAH9", "HYDIN", "MUC16", "TSPAN1"],
}
CUR_TIER2_T = {"CD4": ["CD4", "BCL11B", "LTB", "IL7R"], "CD8": ["CD8A", "CD8B"]}

# --- PROVISIONAL candidates (edit after reading the audit + discovery) ---
# T: add dropout-robust CD3-complex members; NK: purge shared cytotoxic genes,
# keep NK-exclusive anchors only.
CAND_T  = ["CD3D", "CD3E", "CD3G", "CD247"]
CAND_NK = ["KLRC3", "KLRD1", "NCAM1", "NCR1"]

# lineage hints for annotating discovered genes (readability only)
LINEAGE_HINT = {}
for g in ["CD3D", "CD3E", "CD3G", "CD247", "CD2", "CD5", "CD7", "CD6", "IL7R",
          "LCK", "LAT", "ITK", "SKAP1", "THEMIS", "BCL11B", "TCF7", "CD28", "LTB", "CD52"]:
    LINEAGE_HINT[g] = "T"
for g in ["KLRC1", "KLRC2", "KLRC3", "KLRD1", "NCAM1", "NCR1", "NCR3", "KLRF1",
          "EOMES", "TBX21", "CD160", "IL2RB"]:
    LINEAGE_HINT[g] = "NK"
for g in ["NKG7", "GNLY", "GZMB", "GZMK", "GZMH", "GZMA", "PRF1", "KLRB1", "CCL5", "CST7"]:
    LINEAGE_HINT[g] = "cytotox(shared)"
for g in ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"]:
    LINEAGE_HINT[g] = "B"
for g in ["LYZ", "CD68", "C1QA", "C1QB", "CD14", "APOE", "MARCO", "MRC1", "CD163"]:
    LINEAGE_HINT[g] = "Mono/Mac"
for g in ["MKI67", "TOP2A", "PCNA", "STMN1"]:
    LINEAGE_HINT[g] = "proliferating"


def hint(names):
    return [LINEAGE_HINT.get(n, "") for n in names]


def gene_vec(ad, g):
    x = ad[:, g].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()


# %%
# --- Cell 2: load + recompute Leiden (match prior numbering) ----------------
adata = sc.read_h5ad(ADATA_PATH)
print(f"Loaded (in memory only): {adata.n_obs:,} cells x {adata.n_vars:,} genes")

leiden_kwargs = dict(resolution=LEIDEN_RES, key_added=LEIDEN_KEY, random_state=SEED)
try:
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False, **leiden_kwargs)
except Exception as e:
    print(f"  [warn] igraph-flavor leiden failed ({e}); default flavor")
    sc.tl.leiden(adata, **leiden_kwargs)

sizes = adata.obs[LEIDEN_KEY].value_counts()
for c, exp in EXPECTED_SIZES.items():
    obs_n = int(sizes.get(c, 0))
    print(f"  cluster {c}: expected {exp}, observed {obs_n}"
          f"{'  *** MISMATCH (map by size)' if obs_n != exp else ''}")

lymph_mask = adata.obs[LEIDEN_KEY].isin(LYMPHOID_CLUSTERS)
macref_mask = adata.obs[LEIDEN_KEY] == MAC_REF_CLUSTER


# %%
# --- Cell 3: CURRENT-MARKER AUDIT (dropout + non-specificity) ---------------
# For each current Tier1 marker: detection in cells labeled that lineage, in the
# lymphoid island, and in a pure-macrophage reference. A good marker is high in
# its target and low in the mac reference. Dropout = low in target; non-specific
# = high in mac reference.
audit_rows = []
for lineage, markers in CUR_TIER1.items():
    own_mask = adata.obs["tier1_celltype"] == lineage
    for g in markers:
        present = g in adata.var_names
        row = {"lineage": lineage, "marker": g, "present": present}
        if present:
            row["det_own_label"] = round(float((gene_vec(adata[own_mask.values], g) > 0).mean()), 3) \
                if own_mask.sum() else np.nan
            row["det_lymphoid_island"] = round(float((gene_vec(adata[lymph_mask.values], g) > 0).mean()), 3)
            row["det_mac_ref"] = round(float((gene_vec(adata[macref_mask.values], g) > 0).mean()), 3)
        audit_rows.append(row)
audit = pd.DataFrame(audit_rows)
print("\n=== CURRENT Tier 1 marker audit ===")
print("(det_own_label uses current labels = imperfect; det_lymphoid_island is the")
print(" cleaner read for T/NK; det_mac_ref exposes non-specific / myeloid leakage)")
with pd.option_context("display.max_rows", None, "display.width", 160):
    print(audit.to_string(index=False))

# Targeted checks for the three known failures
print("\n--- Failure 1: T set drops out in the T island (cluster 16) ---")
for g in CUR_TIER1["T cells"] + ["CD3G", "CD247"]:
    if g in adata.var_names:
        d = (gene_vec(adata[(adata.obs[LEIDEN_KEY] == T_CLUSTER).values], g) > 0).mean()
        print(f"    {g:6s} det in T island = {d:.3f}{'   <- in current T set' if g in CUR_TIER1['T cells'] else '   <- candidate add'}")

print("\n--- Failure 2: NK cytotoxic genes fire on the T island (should be low there) ---")
for g in CUR_TIER1["NK cells"]:
    if g in adata.var_names:
        dT = (gene_vec(adata[(adata.obs[LEIDEN_KEY] == T_CLUSTER).values], g) > 0).mean()
        dNK = (gene_vec(adata[(adata.obs[LEIDEN_KEY] == NK_CLUSTER).values], g) > 0).mean()
        tag = "shared cytotox" if g in ("GNLY", "PRF1", "GZMB") else "NK-exclusive"
        print(f"    {g:6s} det  T={dT:.3f}  NK={dNK:.3f}   ({tag})")

print("\n--- Failure 3: Tier2 CD4 marker is myeloid here ---")
for g in CUR_TIER2_T["CD4"] + CUR_TIER2_T["CD8"]:
    if g in adata.var_names:
        dmac = (gene_vec(adata[macref_mask.values], g) > 0).mean()
        dT = (gene_vec(adata[(adata.obs[LEIDEN_KEY] == T_CLUSTER).values], g) > 0).mean()
        print(f"    {g:7s} det  mac_ref={dmac:.3f}  T_island={dT:.3f}")


# %%
# --- Cell 4: unbiased discovery for each lymphoid cluster vs rest -----------
sc.tl.rank_genes_groups(adata, groupby=LEIDEN_KEY, groups=LYMPHOID_CLUSTERS,
                        method="wilcoxon", pts=True)
top_tables = {}
for c in LYMPHOID_CLUSTERS:
    df = sc.get.rank_genes_groups_df(adata, group=c).head(N_TOP).copy()
    df.insert(1, "lineage_hint", hint(df["names"]))
    df = df.rename(columns={"names": "gene", "logfoldchanges": "log2fc",
                            "pvals_adj": "padj", "pct_nz_group": "pct_in",
                            "pct_nz_reference": "pct_rest"})
    top_tables[c] = df[["gene", "lineage_hint", "log2fc", "padj", "pct_in", "pct_rest"]] \
        .round({"log2fc": 2, "pct_in": 3, "pct_rest": 3})
    print(f"\n=== Cluster {c} vs rest — top {N_TOP} ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(top_tables[c].to_string(index=False))


# %%
# --- Cell 5: direct T-vs-NK contrast (16 vs 17) -----------------------------
# NOTE: with a specific `reference` (not 'rest'), scanpy does not compute
# pts_rest, so rank_genes_groups_df has no pct_nz_reference column. Compute
# per-cluster detection ourselves for both clusters and map it on by gene.
def cluster_detection(cluster):
    sub = adata[(adata.obs[LEIDEN_KEY] == cluster).values]
    det = np.asarray((sub.X > 0).mean(axis=0)).ravel()
    return pd.Series(det, index=adata.var_names)

det_T = cluster_detection(T_CLUSTER)
det_NK = cluster_detection(NK_CLUSTER)

sc.tl.rank_genes_groups(adata, groupby=LEIDEN_KEY, groups=[T_CLUSTER], reference=NK_CLUSTER,
                        method="wilcoxon", key_added="tvsnk")
tvsnk = sc.get.rank_genes_groups_df(adata, group=T_CLUSTER, key="tvsnk").copy()
tvsnk = tvsnk.rename(columns={"names": "gene", "logfoldchanges": "log2fc", "pvals_adj": "padj"})
tvsnk["lineage_hint"] = hint(tvsnk["gene"])
tvsnk[f"pct_{T_CLUSTER}"] = tvsnk["gene"].map(det_T).round(3)
tvsnk[f"pct_{NK_CLUSTER}"] = tvsnk["gene"].map(det_NK).round(3)
tvsnk["log2fc"] = tvsnk["log2fc"].round(2)
tvsnk = tvsnk[["gene", "lineage_hint", "log2fc", "padj", f"pct_{T_CLUSTER}", f"pct_{NK_CLUSTER}"]]

print(f"\n=== T-vs-NK: top UP in {T_CLUSTER} (T) ===")
print(tvsnk.sort_values("log2fc", ascending=False).head(25).to_string(index=False))
print(f"\n=== T-vs-NK: top UP in {NK_CLUSTER} (NK) ===")
print(tvsnk.sort_values("log2fc", ascending=True).head(25).to_string(index=False))


# %%
# --- Cell 6: PROVISIONAL candidate-set evaluation (read-only) ---------------
# Score with candidate T / NK sets into temp columns and measure, WITHOUT
# relabeling: (a) would the current-Unassigned lymphoid cells now clear the T
# or NK threshold, and (b) do the 477 "NK" cells in the T island flip to T
# (candidate T score > candidate NK score)?
def score_temp(genes, name):
    present = [g for g in genes if g in adata.var_names]
    sc.tl.score_genes(adata, gene_list=present, score_name=name)
    return present

t_used = score_temp(CAND_T, "_cand_T")
nk_used = score_temp(CAND_NK, "_cand_NK")
print(f"\nCandidate T genes used:  {t_used}")
print(f"Candidate NK genes used: {nk_used}")

is_unassigned = adata.obs["tier1_celltype"] == "Unassigned"
unassigned_lymph = is_unassigned & lymph_mask
rescue_T = int((adata.obs.loc[unassigned_lymph, "_cand_T"] >= SCORE_THRESHOLD).sum())
rescue_NK = int((adata.obs.loc[unassigned_lymph, "_cand_NK"] >= SCORE_THRESHOLD).sum())
n_ul = int(unassigned_lymph.sum())
print(f"\nUnassigned cells in lymphoid clusters: {n_ul}")
print(f"  would clear candidate T threshold:  {rescue_T} ({100*rescue_T/max(n_ul,1):.1f}%)")
print(f"  would clear candidate NK threshold: {rescue_NK} ({100*rescue_NK/max(n_ul,1):.1f}%)")

nk_in_T = (adata.obs[LEIDEN_KEY] == T_CLUSTER) & (adata.obs["tier1_celltype"] == "NK cells")
n_flip = int((adata.obs.loc[nk_in_T, "_cand_T"] > adata.obs.loc[nk_in_T, "_cand_NK"]).sum())
n_nkT = int(nk_in_T.sum())
print(f"\n'NK' cells inside T island (cluster {T_CLUSTER}): {n_nkT}")
print(f"  candidate T score > candidate NK score (would flip to T): {n_flip} "
      f"({100*n_flip/max(n_nkT,1):.1f}%)")

# mean candidate NK score in T vs NK island (should be LOW in T if purified)
for lab, msk in [("T island", adata.obs[LEIDEN_KEY] == T_CLUSTER),
                 ("NK island", adata.obs[LEIDEN_KEY] == NK_CLUSTER)]:
    print(f"  mean _cand_NK in {lab}: {adata.obs.loc[msk.values, '_cand_NK'].mean():.3f}")


# %%
# --- Cell 7: write tables + summary -----------------------------------------
os.makedirs(OUTDIR, exist_ok=True)
audit.to_csv(os.path.join(OUTDIR, "current_marker_audit.csv"), index=False)
for c, df in top_tables.items():
    df.to_csv(os.path.join(OUTDIR, f"markers_cluster{c}_vs_rest.csv"), index=False)
tvsnk.to_csv(os.path.join(OUTDIR, f"markers_{T_CLUSTER}_vs_{NK_CLUSTER}_TvNK.csv"), index=False)

print("\n" + "=" * 70)
print("MARKER REFINEMENT DIAGNOSTIC COMPLETE (read-only)")
print("=" * 70)
print(f"  Tables -> {OUTDIR}")
print("  Decide together from output:")
print("    - final T set (CD3 complex + dropout-robust, T-biased)")
print("    - final NK set (NK-exclusive anchors; drop shared cytotoxic genes)")
print("    - Tier2 CD4 fix (demote CD4 marker or reweight toward BCL11B/IL7R/LTB)")
