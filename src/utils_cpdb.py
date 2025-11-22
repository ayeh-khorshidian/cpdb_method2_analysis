# =========================
# Imports and configuration
# =========================

from pathlib import Path
from datetime import datetime
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cellphonedb.src.core.methods import cpdb_statistical_analysis_method


# Base directories (relative paths, safe for GitHub)
BASE = Path("cpdb_out")              # root folder for CPDB outputs
CPDB_INPUTS_ROOT = BASE / "cpdb_inputs"
FIG_ROOT = BASE / "figures_method2"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# Path to CellPhoneDB database zip (to be adjusted by user)
# Example: Path.home() / ".cpdb" / "releases" / "5.0.1" / "cellphonedb.zip"
CPDB_ZIP = Path("cellphonedb.zip")   # placeholder; update to your local CPDB zip


# Comparisons used in this analysis
comparisons = {
    "Rplus1_vs_Lminus3": {},
    "Rplus1_vs_preflightAll": {},
    "Recovery_vs_Lminus3": {},
    "Recovery_vs_preflightAll": {},
}
# ======================================
# Run CPDB Method 2 for each comparison
# ======================================

def run_cpdb_method2_for_comparison(
    tag: str,
    inputs_root: Path = CPDB_INPUTS_ROOT,
    out_root: Path = BASE,
    cpdb_zip: Path = CPDB_ZIP,
    counts_data: str = "hgnc_symbol",
    iterations: int = 1000,
    threads: int = 4,
):
    """
    Run CellPhoneDB Method 2 (statistical_analysis) for a single comparison.

    Parameters
    ----------
    tag : str
        Name of the comparison (e.g., 'Rplus1_vs_Lminus3').
    inputs_root : Path
        Folder that contains 'cpdb_inputs/<tag>/meta.txt' and 'counts.txt'.
    out_root : Path
        Base output folder where '<tag>/' will be created for CPDB outputs.
    cpdb_zip : Path
        Path to CellPhoneDB database zip file.
    counts_data : str
        Type of gene identifiers in counts.txt (e.g. 'hgnc_symbol').
    iterations : int
        Number of permutations for the statistical analysis.
    threads : int
        Number of threads to use.
    """

    in_dir = inputs_root / tag
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = in_dir / "meta.txt"
    counts_file = in_dir / "counts.txt"

    print(f"\nRunning CPDB Method 2 for: {tag}")
    print(f"  meta:   {meta_file}")
    print(f"  counts: {counts_file}")
    print(f"  out:    {out_dir}")

    cpdb_statistical_analysis_method.call(
        cpdb_file_path=str(cpdb_zip),
        meta_file_path=str(meta_file),
        counts_file_path=str(counts_file),
        counts_data=counts_data,
        output_path=str(out_dir),
        iterations=iterations,
        threads=threads,
    )

    print(f"Completed Method 2 for: {tag} → {out_dir}")
    return out_dir


def run_cpdb_method2_for_all():
    """
    Run CellPhoneDB Method 2 for all comparisons defined in `comparisons`.

    NOTE:
    - This can be computationally expensive.
    - Adjust `iterations` and `threads` above if needed.
    """
    outputs = {}
    for tag in comparisons.keys():
        out_dir = run_cpdb_method2_for_comparison(tag=tag)
        outputs[tag] = out_dir
    return outputs
# ============================
# Load CPDB output tables
# ============================

def load_cpdb_tables(comp_name: str):
    """
    Load the latest CPDB 'means', 'pvalues', and 'significant_means' tables
    for a given comparison.

    Parameters
    ----------
    comp_name : str
        Comparison name (e.g., 'Rplus1_vs_Lminus3').

    Returns
    -------
    means : pandas.DataFrame
    pvals : pandas.DataFrame
    sig   : pandas.DataFrame
    """
    comp_dir = BASE / comp_name

    def _load(prefix: str) -> pd.DataFrame:
        pattern = str(comp_dir / f"{prefix}*.txt")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        latest = files[-1]
        print(f"[{comp_name}] loading: {Path(latest).name}")
        return pd.read_csv(latest, sep="\t")

    means = _load("statistical_analysis_means_")
    pvals = _load("statistical_analysis_pvalues_")
    sig   = _load("statistical_analysis_significant_means_")

    return means, pvals, sig
# ==============================================
# Build a table of top interactions for a target
# ==============================================

def save_top_interactions_table(
    comp_name: str,
    receiver: str = "CD4_T",
    senders=("CD14_Mono", "CD16_Mono", "DC"),
    p_threshold: float = 0.05,
    top_n: int = 100,
):
    """
    Build and save a summary table of top ligand–receptor interactions for
    a given receiver cell type across multiple sender cell types.

    The output is a CSV file in FIG_ROOT and a DataFrame (returned).
    """

    means, pvals, sig = load_cpdb_tables(comp_name)
    sig_pairs = sig["interacting_pair"].unique()

    means_sig = means[means["interacting_pair"].isin(sig_pairs)].copy()
    pvals_sig = pvals[pvals["interacting_pair"].isin(sig_pairs)].copy()

    records = []
    for s in senders:
        col = f"{s}|{receiver}"
        if col not in means_sig.columns or col not in pvals_sig.columns:
            continue

        df = pd.DataFrame({
            "interacting_pair": means_sig["interacting_pair"],
            "mean": means_sig[col],
            "p": pvals_sig[col],
        }).dropna()

        df = df[df["p"] < p_threshold]

        for _, row in df.iterrows():
            records.append({
                "comparison": comp_name,
                "sender": s,
                "receiver": receiver,
                "interacting_pair": row["interacting_pair"],
                "mean": row["mean"],
                "p": row["p"],
                "-log10p": -np.log10(row["p"] + 1e-12),
            })

    if not records:
        print(f"[{comp_name}] No significant interactions found for receiver={receiver}.")
        return None

    df_out = pd.DataFrame(records)
    df_out["pair_readable"] = df_out["interacting_pair"].str.replace("_", " ", regex=False)
    df_out = df_out.sort_values("-log10p", ascending=False).head(top_n)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = FIG_ROOT / f"table_top_interactions_{comp_name}_{receiver}_top{top_n}_{ts}.csv"
    df_out.to_csv(out_path, index=False)

    print(f"Saved top interaction table: {out_path}")
    return df_out
# =====================
# Heatmap (generic)
# =====================

def simple_heatmap(
    comp_name: str,
    receiver: str = "CD4_T",
    senders=("CD14_Mono", "CD16_Mono", "DC"),
    top_n: int = 40,
):
    """
    Heatmap of -log10(p) for top ligand–receptor pairs targeting a receiver.

    Rows: ligand–receptor pairs
    Columns: sender cell types
    """

    means, pvals, sig = load_cpdb_tables(comp_name)
    sig_pairs = sig["interacting_pair"].unique()

    means_sig = means[means["interacting_pair"].isin(sig_pairs)].copy()
    pvals_sig = pvals[pvals["interacting_pair"].isin(sig_pairs)].copy()

    cols = [f"{s}|{receiver}" for s in senders if f"{s}|{receiver}" in pvals_sig.columns]
    if not cols:
        print(f"[{comp_name}] No pvalue columns found for receiver={receiver}")
        return

    means_sig["max_mean"] = means_sig[cols].max(axis=1)
    top_pairs = means_sig.sort_values("max_mean", ascending=False).head(top_n)["interacting_pair"].values

    pvals_top = pvals_sig[pvals_sig["interacting_pair"].isin(top_pairs)].copy()
    mat = -np.log10(pvals_top[cols].replace(0, np.nan))
    mat.index = pvals_top["interacting_pair"].str.replace("_", " ", regex=False)
    mat.columns = [c.split("|")[0] for c in cols]

    plt.figure(figsize=(10, max(4, 0.4 * len(mat))))
    sns.heatmap(
        mat,
        cmap="viridis",
        linewidths=0.4,
        linecolor="gray",
        cbar_kws={"label": "-log10(p-value)"}
    )
    plt.title(f"{comp_name} — Top {top_n} interactions to {receiver}", fontsize=14)
    plt.xlabel("Sender")
    plt.ylabel("Ligand–receptor pair")
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = FIG_ROOT / f"heatmap_{comp_name}_{receiver}_top{top_n}_{ts}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved heatmap: {out_path}")
# ==========================
# Sender barplot (generic)
# ==========================

def sender_barplot(
    comp_name: str,
    receiver: str = "CD4_T",
    senders=("CD14_Mono", "CD16_Mono", "DC", "B", "CD8_T", "other_T", "other"),
    p_threshold: float = 0.05,
):
    """
    Barplot of number of significant interactions (p < p_threshold) per sender
    for a given receiver cell type.
    """

    _, pvals, sig = load_cpdb_tables(comp_name)
    sig_pairs = sig["interacting_pair"].unique()
    pvals_sig = pvals[pvals["interacting_pair"].isin(sig_pairs)].copy()

    counts = []
    valid_senders = []
    for s in senders:
        col = f"{s}|{receiver}"
        if col not in pvals_sig.columns:
            continue
        n_sig = (pvals_sig[col] < p_threshold).sum()
        counts.append(n_sig)
        valid_senders.append(s)

    if not counts:
        print(f"[{comp_name}] No significant interactions to {receiver}")
        return

    df = pd.DataFrame({"sender": valid_senders, "n_sig": counts})

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="sender", y="n_sig")
    plt.ylabel(f"# significant interactions (p < {p_threshold})")
    plt.xlabel("Sender cell type")
    plt.title(f"{comp_name} — senders to {receiver}")
    plt.xticks(rotation=20)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = FIG_ROOT / f"sender_bar_{comp_name}_{receiver}_{ts}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved sender barplot: {out_path}")
# ===========================
# Ligand-rank plot (generic)
# ===========================

def ligand_rank_plot(
    comp_name: str,
    sender: str = "CD14_Mono",
    receiver: str = "CD4_T",
    p_threshold: float = 0.05,
    top_n: int = 20,
):
    """
    Horizontal barplot of top ligand–receptor pairs from sender to receiver,
    ranked by mean expression and filtered by p-value.
    """

    means, pvals, sig = load_cpdb_tables(comp_name)
    sig_pairs = sig["interacting_pair"].unique()

    means_sig = means[means["interacting_pair"].isin(sig_pairs)].copy()
    pvals_sig = pvals[pvals["interacting_pair"].isin(sig_pairs)].copy()

    col = f"{sender}|{receiver}"
    if col not in means_sig.columns or col not in pvals_sig.columns:
        print(f"[{comp_name}] Column {col} not found.")
        return

    df = pd.DataFrame({
        "interacting_pair": means_sig["interacting_pair"],
        "mean": means_sig[col],
        "p": pvals_sig[col],
    }).dropna()

    df = df[df["p"] < p_threshold]
    if df.empty:
        print(f"[{comp_name}] No significant interactions for {sender}→{receiver}")
        return

    df["pair_readable"] = df["interacting_pair"].str.replace("_", " ", regex=False)
    df = df.sort_values("mean", ascending=False).head(top_n)

    plt.figure(figsize=(8, max(4, 0.4 * len(df))))
    plt.barh(df["pair_readable"], df["mean"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean expression")
    plt.ylabel("Ligand–receptor pair")
    plt.title(f"{comp_name} — top {top_n} {sender}→{receiver}")
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = FIG_ROOT / f"ligand_rank_{comp_name}_{sender}_to_{receiver}_top{top_n}_{ts}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved ligand-rank plot: {out_path}")
# =======================
# Bubble plot (generic)
# =======================

def bubble_plot(
    comp_name: str,
    receiver: str = "CD4_T",
    senders=("CD14_Mono", "CD16_Mono"),
    p_threshold: float = 0.05,
    top_n: int = 30,
):
    """
    Bubble plot of ligand–receptor pairs for a set of senders to a receiver.
    Bubble size ~ mean expression, color ~ -log10(p).
    """

    means, pvals, sig = load_cpdb_tables(comp_name)
    sig_pairs = sig["interacting_pair"].unique()

    means_sig = means[means["interacting_pair"].isin(sig_pairs)].copy()
    pvals_sig = pvals[pvals["interacting_pair"].isin(sig_pairs)].copy()

    records = []
    for s in senders:
        col = f"{s}|{receiver}"
        if col not in means_sig.columns or col not in pvals_sig.columns:
            continue

        df = pd.DataFrame({
            "interacting_pair": means_sig["interacting_pair"],
            "mean": means_sig[col],
            "p": pvals_sig[col],
        }).dropna()

        df = df[df["p"] < p_threshold]
        if df.empty:
            continue

        for _, row in df.iterrows():
            records.append({
                "sender": s,
                "interacting_pair": row["interacting_pair"],
                "mean": row["mean"],
                "-log10p": -np.log10(row["p"] + 1e-12),
            })

    if not records:
        print(f"[{comp_name}] No significant interactions for bubble plot to {receiver}")
        return

    df_b = pd.DataFrame(records)
    df_b["pair_readable"] = df_b["interacting_pair"].str.replace("_", " ", regex=False)
    df_b = df_b.sort_values("-log10p", ascending=False).head(top_n)

    x_labels = df_b["sender"].unique().tolist()
    x_map = {lab: i for i, lab in enumerate(x_labels)}
    df_b["x"] = df_b["sender"].map(x_map)

    plt.figure(figsize=(8, max(4, 0.4 * len(df_b))))
    sc = plt.scatter(
        df_b["x"],
        df_b["pair_readable"],
        s=df_b["mean"] * 120,
        c=df_b["-log10p"],
        cmap="viridis",
        alpha=0.9,
    )
    plt.xticks(range(len(x_labels)), x_labels, rotation=15)
    plt.ylabel("Ligand–receptor pair")
    plt.title(f"{comp_name} — bubble plot to {receiver} (top {top_n})")
    cbar = plt.colorbar(sc)
    cbar.set_label("-log10(p)")
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = FIG_ROOT / f"bubble_{comp_name}_{receiver}_top{top_n}_{ts}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved bubble plot: {out_path}")
# ========================
# Volcano plot (generic)
# ========================

def volcano_plot(
    comp_name: str,
    sender: str = "CD14_Mono",
    receiver: str = "CD4_T",
):
    """
    Volcano-like plot: mean expression vs -log10(p) for all pairs from
    sender to receiver.
    """

    means, pvals, sig = load_cpdb_tables(comp_name)
    sig_pairs = sig["interacting_pair"].unique()

    means_sig = means[means["interacting_pair"].isin(sig_pairs)].copy()
    pvals_sig = pvals[pvals["interacting_pair"].isin(sig_pairs)].copy()

    col = f"{sender}|{receiver}"
    if col not in means_sig.columns or col not in pvals_sig.columns:
        print(f"[{comp_name}] Column {col} not found.")
        return

    df = pd.DataFrame({
        "interacting_pair": means_sig["interacting_pair"],
        "mean": means_sig[col],
        "p": pvals_sig[col],
    }).dropna()

    df["-log10p"] = -np.log10(df["p"] + 1e-12)
    df["pair_readable"] = df["interacting_pair"].str.replace("_", " ", regex=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(df["mean"], df["-log10p"], s=20)
    plt.xlabel(f"Mean expression ({sender}→{receiver})")
    plt.ylabel("-log10(p-value)")
    plt.title(f"{comp_name} — volcano {sender}→{receiver}")
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = FIG_ROOT / f"volcano_{comp_name}_{sender}_to_{receiver}_{ts}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved volcano plot: {out_path}")
# =====================================
# Convenience wrappers for CD4_T plots
# =====================================

def make_all_cd4_plots_for_comparison(comp_name: str):
    """
    Generate all CD4_T-related plots for a single comparison.
    """
    # Heatmap
    simple_heatmap(
        comp_name=comp_name,
        receiver="CD4_T",
        senders=("CD14_Mono", "CD16_Mono", "DC"),
        top_n=40,
    )

    # Sender barplot
    sender_barplot(
        comp_name=comp_name,
        receiver="CD4_T",
        senders=("CD14_Mono", "CD16_Mono", "DC", "B", "CD8_T", "other_T", "other"),
        p_threshold=0.05,
    )

    # Ligand-rank (CD14_Mono → CD4_T)
    ligand_rank_plot(
        comp_name=comp_name,
        sender="CD14_Mono",
        receiver="CD4_T",
        p_threshold=0.05,
        top_n=20,
    )

    # Bubble plot (CD14_Mono / CD16_Mono → CD4_T)
    bubble_plot(
        comp_name=comp_name,
        receiver="CD4_T",
        senders=("CD14_Mono", "CD16_Mono"),
        p_threshold=0.05,
        top_n=30,
    )

    # Volcano (CD14_Mono → CD4_T)
    volcano_plot(
        comp_name=comp_name,
        sender="CD14_Mono",
        receiver="CD4_T",
    )


def make_all_cd4_plots_for_all():
    """
    Generate CD4_T plots for all comparisons.
    """
    for comp in comparisons.keys():
        print(f"\n=== CD4_T plots for {comp} ===")
        make_all_cd4_plots_for_comparison(comp)
# ===================================
# Convenience wrappers for NK plots
# ===================================

def make_all_nk_plots_for_comparison(comp_name: str):
    """
    Generate all NK-related plots for a single comparison.
    """

    # Heatmap to NK
    simple_heatmap(
        comp_name=comp_name,
        receiver="NK",
        senders=("CD14_Mono", "CD16_Mono", "B", "CD4_T", "CD8_T", "other_T", "other"),
        top_n=40,
    )

    # Sender barplot to NK
    sender_barplot(
        comp_name=comp_name,
        receiver="NK",
        senders=("CD14_Mono", "CD16_Mono", "B", "CD8_T", "CD4_T", "other_T", "other"),
        p_threshold=0.05,
    )

    # Ligand-rank (CD14_Mono → NK)
    ligand_rank_plot(
        comp_name=comp_name,
        sender="CD14_Mono",
        receiver="NK",
        p_threshold=0.05,
        top_n=20,
    )

    # Bubble plot (CD14_Mono / CD16_Mono → NK)
    bubble_plot(
        comp_name=comp_name,
        receiver="NK",
        senders=("CD14_Mono", "CD16_Mono"),
        p_threshold=0.05,
        top_n=30,
    )

    # Volcano (CD14_Mono → NK)
    volcano_plot(
        comp_name=comp_name,
        sender="CD14_Mono",
        receiver="NK",
    )


def make_all_nk_plots_for_all():
    """
    Generate NK plots for all comparisons.
    """
    for comp in comparisons.keys():
        print(f"\n=== NK plots for {comp} ===")
        make_all_nk_plots_for_comparison(comp)
