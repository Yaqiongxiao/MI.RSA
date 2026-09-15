"""
Script: MI_RSA analysis.py

Purpose
-------
Analyze the static and dynamic neural representational organization of
unilateral multi-joint motor imagery using EEG and Crossnobis-based
representational similarity analysis (RSA).

Usage
-----
1. Put the FineMI dataset in the `Data/FineMI/` folder.
2. Update `DATA_PATH` and `OUTPUT_DIR` below if necessary.
3. Run this script from beginning to end.

Input
-----
Raw FineMI EEG recordings (.cnt).

Output
------
- Static RSA: RDM, hierarchical clustering, and MDS.
- Theoretical-model RSA across three frequency bands.
- Static distal–proximal contrast statistics.
- Dynamic RSA and sliding-window representational contrasts.
- Cluster-based permutation statistics.
- Low-beta × mu/alpha representational-contrast interaction results.

Important
---------
Do not modify preprocessing parameters, frequency-band definitions,
Crossnobis settings, sliding-window parameters, or permutation settings
unless the analysis plan is intentionally revised.

Author:An Long
Date:2026-09-15
"""

import os, gc, random
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import spearmanr, ttest_1samp, ttest_rel, t as scipy_t
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from mne.stats import permutation_cluster_1samp_test

# =============================================================================
# 0. Settings
# =============================================================================
random.seed(42)
np.random.seed(42)

DATA_PATH = "Data/FineMI"
OUTPUT_DIR = "results_MI_RSA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASK_NAMES = ["HOC", "WFE", "WAA", "EPS", "EFE", "SPS", "SAA", "SFE"]
SUBJECTS = [f"subject{i}" for i in range(1, 19)]
DISTAL_IDX, PROXIMAL_IDX = [0, 1, 2], [3, 4, 5, 6, 7]
ROI_CHANNELS = ['FC3', 'FC1', 'C3', 'C1', 'CP3', 'CP1', 'FC5', 'C5', 'CP5']

BANDS = {
    'alpha': {'range': (8, 12), 'n_freqs': 8, 'n_cycles': 5},
    'beta_low': {'range': (13, 20), 'n_freqs': 8, 'n_cycles': 5},
    'beta_high': {'range': (21, 30), 'n_freqs': 10, 'n_cycles': 5},
}
BAND_DISPLAY = {
    "Mu/Alpha Band": "alpha",
    "Low-beta Band": "beta_low",
    "High-beta Band": "beta_high",
}
BASELINE_WINDOW, STATIC_MI_WINDOW = (-4, -2), (0, 4)
N_REPS, SHRINKAGE = 100, 0.10
N_PERM, CLUSTER_ALPHA = 5000, 0.05

TOTAL_TIME, WIN_LEN, WIN_STEP = 4.0, 0.5, 0.1
N_WINDOWS = int((TOTAL_TIME - WIN_LEN) / WIN_STEP) + 1
TIME_WINDOWS = [(round(i * WIN_STEP, 2), round(i * WIN_STEP + WIN_LEN, 2)) for i in range(N_WINDOWS)]
time_points = np.array([round(a + WIN_LEN / 2, 3) for a, _ in TIME_WINDOWS], dtype=float)

TASK_COLORS = ['red', 'blue', 'blue', 'green', 'green', 'purple', 'purple', 'purple']
TASK_MARKERS = ['o', 's', '^', 'D', 's', 'D', '^', 's']
MODEL_ORDER = ["Distal-Proximal", "Anatomical Distance", "Joint Category", "Movement Type"]


# =============================================================================
# 1. Core utilities
# =============================================================================
def build_pair_indices():
    dd, pp, dp = [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if i in DISTAL_IDX and j in DISTAL_IDX:
                dd.append((i, j))
            elif i in PROXIMAL_IDX and j in PROXIMAL_IDX:
                pp.append((i, j))
            else:
                dp.append((i, j))
    return dd, pp, dp


DD_PAIRS, PP_PAIRS, DP_PAIRS = build_pair_indices()
ALL_WITHIN_PAIRS = DD_PAIRS + PP_PAIRS


def calc_crossnobis_rdm(features, n_reps=N_REPS, shrinkage=SHRINKAGE, seed=None):
    """Repeated random split-half Crossnobis RDM."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"features must be condition × trial × feature, got {x.shape}")
    n_cond, n_trials, n_ch = x.shape
    if n_trials < 2:
        raise ValueError("At least 2 trials are required.")

    residuals = (x - x.mean(axis=1, keepdims=True)).reshape(-1, n_ch)
    sigma = (1 - shrinkage) * np.cov(residuals, rowvar=False) + shrinkage * np.eye(n_ch)
    precision = np.linalg.pinv(sigma)
    half, rdm = n_trials // 2, np.zeros((n_cond, n_cond), dtype=float)
    rng = np.random.RandomState(seed) if seed is not None else np.random

    for a in range(n_cond):
        for b in range(a + 1, n_cond):
            d = np.empty(n_reps, dtype=float)
            for rep in range(n_reps):
                ia, ib = rng.permutation(n_trials), rng.permutation(n_trials)
                da = x[a, ia[:half]].mean(0) - x[b, ib[:half]].mean(0)
                db = x[a, ia[half:]].mean(0) - x[b, ib[half:]].mean(0)
                d[rep] = da @ precision @ db.T
            rdm[a, b] = rdm[b, a] = d.mean()
    return rdm


def pair_mean(rdm, pairs):
    return float(np.mean([rdm[i, j] for i, j in pairs]))


def upper_triangle_vector(rdm):
    return np.asarray(rdm[np.triu_indices(rdm.shape[0], k=1)], dtype=float)


# =============================================================================
# 2. EEG preprocessing and feature extraction
# =============================================================================
def load_raw_subject(subject_name):
    sub_id = int(subject_name.replace("subject", ""))
    path = os.path.join(DATA_PATH, subject_name, "EEG")
    raws = []

    if sub_id == 1:
        r = mne.io.read_raw_cnt(os.path.join(path, "block1-4.cnt"), preload=True, verbose=False)
        r.annotations.delete(np.arange(-40, 0))
        raws.append(r)
        for b in range(5, 9):
            raws.append(mne.io.read_raw_cnt(os.path.join(path, f"block{b}.cnt"), preload=True, verbose=False))
    else:
        for b in range(1, 9):
            r = mne.io.read_raw_cnt(os.path.join(path, f"block{b}.cnt"), preload=True, verbose=False)
            if sub_id == 5 and b == 6:
                r.annotations.delete(0)
            raws.append(r)

    raw = mne.concatenate_raws(raws)
    raw.set_eeg_reference('average', verbose=False)
    return raw


def extract_subject_features(subject_name):
    raw_base = load_raw_subject(subject_name)
    event_ids = {name: i + 1 for i, name in enumerate(TASK_NAMES)}

    # Static features: IIR filter
    raw_static = raw_base.copy().filter(4, 40, method='iir', verbose=False)
    raw_static.pick_channels([c for c in ROI_CHANNELS if c in raw_static.ch_names])
    events, _ = mne.events_from_annotations(raw_static, verbose=False)
    static_tmp = {b: [] for b in BANDS}

    for task in TASK_NAMES:
        epochs = mne.Epochs(raw_static, events, event_id=event_ids[task], tmin=-4, tmax=4,
                            baseline=None, preload=True, verbose=False)
        for band, p in BANDS.items():
            freqs = np.linspace(*p['range'], p['n_freqs'])
            tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=p['n_cycles'],
                                                return_itc=False, average=False, verbose=False)
            tfr.apply_baseline(BASELINE_WINDOW, mode='logratio', verbose=False)
            mask = (tfr.times >= STATIC_MI_WINDOW[0]) & (tfr.times <= STATIC_MI_WINDOW[1])
            pat = tfr.data[:, :, :, mask].mean(axis=(2, 3))
            pat = (pat - pat.mean(1, keepdims=True)) / (pat.std(1, keepdims=True) + 1e-10)
            static_tmp[band].append(pat)
        del epochs

    static_features = {}
    for band in BANDS:
        n = min(map(len, static_tmp[band]))
        static_features[band] = np.array([x[:n] for x in static_tmp[band]])
    del raw_static
    gc.collect()

    # Dynamic features: FIR filter
    raw_dynamic = raw_base.copy().filter(4, 40, verbose=False)
    raw_dynamic.pick_channels([c for c in ROI_CHANNELS if c in raw_dynamic.ch_names])
    dynamic_tmp = {b: [[[] for _ in TASK_NAMES] for _ in range(N_WINDOWS)] for b in BANDS}

    for ti, task in enumerate(TASK_NAMES):
        epochs = mne.Epochs(raw_dynamic, events, event_id=event_ids[task], tmin=-4, tmax=4,
                            baseline=None, preload=True, verbose=False)
        for band, p in BANDS.items():
            freqs = np.linspace(*p['range'], p['n_freqs'])
            tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=p['n_cycles'],
                                                return_itc=False, average=False, verbose=False)
            tfr.apply_baseline(BASELINE_WINDOW, mode='logratio', verbose=False)
            for wi, (ws, we) in enumerate(TIME_WINDOWS):
                mask = (tfr.times >= ws) & (tfr.times <= we)
                if mask.any():
                    pat = tfr.data[:, :, :, mask].mean(axis=(2, 3))
                    pat = (pat - pat.mean(1, keepdims=True)) / (pat.std(1, keepdims=True) + 1e-10)
                    dynamic_tmp[band][wi][ti] = pat
        del epochs
        gc.collect()

    dynamic_features = {b: [] for b in BANDS}
    for band in BANDS:
        for wi in range(N_WINDOWS):
            n = min(len(x) for x in dynamic_tmp[band][wi])
            dynamic_features[band].append(np.array([x[:n] for x in dynamic_tmp[band][wi]]))

    del raw_dynamic, raw_base
    gc.collect()
    return static_features, dynamic_features


# =============================================================================
# 3. Static RSA
#    3.1 RDM
#    3.2 Dendrogram
#    3.3 MDS
# =============================================================================
def run_static_rsa(static_features_all):
    print("\n>>> [Static RSA] Computing participant RDMs...")
    all_rdms = {b: [] for b in BANDS}

    for si, subj in enumerate(SUBJECTS, start=1):
        for band in BANDS:
            seed = 42 + si * 100 + sum(map(ord, band))
            all_rdms[band].append(calc_crossnobis_rdm(static_features_all[subj][band], seed=seed))

    for band in BANDS:
        display = band.replace('_', ' ').title()
        rdms, avg = all_rdms[band], np.mean(all_rdms[band], axis=0)

        # Descriptive distal-proximal model and clustering statistics
        model = np.ones((8, 8))
        model[np.ix_(DISTAL_IDX, DISTAL_IDX)] = 0
        model[np.ix_(PROXIMAL_IDX, PROXIMAL_IDX)] = 0
        np.fill_diagonal(model, 0)
        tri, mv = np.triu_indices(8, 1), model[np.triu_indices(8, 1)]

        corrs = np.array([spearmanr(r[tri], mv).statistic for r in rdms])
        _, p_model = ttest_1samp(corrs, 0, alternative='greater')
        within = np.array([pair_mean(r, ALL_WITHIN_PAIRS) for r in rdms])
        between = np.array([pair_mean(r, DP_PAIRS) for r in rdms])
        _, p_cat = ttest_rel(between, within, alternative='greater')

        rng = np.random.RandomState(42 + sum(map(ord, band)))
        actual_rho = spearmanr(avg[tri], mv).statistic
        null = []
        for _ in range(5000):
            idx = rng.permutation(8)
            pm = model[idx][:, idx]
            null.append(spearmanr(avg[tri], pm[tri]).statistic)
        p_perm = np.mean(np.asarray(null) >= actual_rho)

        plot_rdm = np.maximum(avg, 0)
        np.fill_diagonal(plot_rdm, 0)
        linkage = hierarchy.linkage(squareform(plot_rdm), method='average')
        labels = fcluster(linkage, t=2, criterion='maxclust')
        sil = silhouette_score(plot_rdm, labels, metric='precomputed')

        rng = np.random.RandomState(42 + sum(map(ord, band)))
        null_sil = []
        for _ in range(5000):
            shuffled = []
            for r in rdms:
                idx = rng.permutation(8)
                shuffled.append(r[idx][:, idx])
            nr = np.maximum(np.mean(shuffled, axis=0), 0)
            np.fill_diagonal(nr, 0)
            nl = hierarchy.linkage(squareform(nr), method='average')
            nlab = fcluster(nl, t=2, criterion='maxclust')
            try:
                null_sil.append(silhouette_score(nr, nlab, metric='precomputed'))
            except ValueError:
                null_sil.append(-1)
        p_sil = np.mean(np.asarray(null_sil) >= sil)

        print(f"\n[{display}] mean r={corrs.mean():.3f}, p={p_model:.4f}; "
              f"within={within.mean():.3f}, between={between.mean():.3f}, p={p_cat:.4f}; "
              f"perm p={p_perm:.4f}; silhouette={sil:.3f}, p={p_sil:.4f}")

        # 3.1 RDM
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(plot_rdm, cmap='viridis', vmin=0)
        ax.set_title(f"Crossnobis Distance Matrix ({display} Band)")
        ax.set_xticks(range(8), TASK_NAMES, rotation=45)
        ax.set_yticks(range(8), TASK_NAMES)
        fig.colorbar(im, ax=ax, label="Crossnobis Distance")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"eeg_rsa_rdm_{band}.png"), dpi=300)
        plt.close(fig)

        # 3.2 Dendrogram
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        hierarchy.dendrogram(linkage, labels=TASK_NAMES, leaf_font_size=12, ax=ax)
        ax.set_title(f"Hierarchical Clustering ({display} Band)\n"
                     f"Silhouette Score (k=2): {sil:.3f} ($p_{{perm}}$ = {p_sil:.4f})", fontsize=12, pad=15)
        ax.set_ylabel("Crossnobis Distance")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"eeg_rsa_dendrogram_{band}.png"), dpi=300)
        plt.close(fig)

        # 3.3 MDS
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(plot_rdm)
        fig, ax = plt.subplots(figsize=(8, 7))

        def encircle(indices, color):
            pts = coords[indices]
            center = pts.mean(0)
            radius = np.linalg.norm(pts - center, axis=1).max()
            radius += max(0.01, radius * 0.1)
            ax.add_patch(patches.Circle(center, radius, color=color, alpha=0.08, zorder=1))
            ax.add_patch(patches.Circle(center, radius, edgecolor=color, fill=False,
                                        linestyle='--', linewidth=1.5, alpha=0.4, zorder=2))

        encircle(DISTAL_IDX, 'blue')
        encircle(PROXIMAL_IDX, 'red')
        for i, name in enumerate(TASK_NAMES):
            ax.scatter(coords[i, 0], coords[i, 1], c=TASK_COLORS[i], marker=TASK_MARKERS[i],
                       s=250, edgecolors='k', linewidths=1.2, zorder=5)
            ax.text(coords[i, 0] + 0.005, coords[i, 1] + 0.005, name, fontsize=10, fontweight='bold', zorder=6)

        legend = [
            Line2D([0], [0], color='red', lw=4, label='Joint: Hand'),
            Line2D([0], [0], color='blue', lw=4, label='Joint: Wrist'),
            Line2D([0], [0], color='green', lw=4, label='Joint: Elbow'),
            Line2D([0], [0], color='purple', lw=4, label='Joint: Shoulder'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Type: Open/Close'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Type: Flex/Ext'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Type: Abd/Add'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=10, label='Type: Pro/Sup'),
            patches.Patch(facecolor='blue', edgecolor='blue', alpha=0.2, label='Group: Distal'),
            patches.Patch(facecolor='red', edgecolor='red', alpha=0.2, label='Group: Proximal'),
        ]
        ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.02, 1), title="Coding Scheme", fontsize=9)
        ax.set(xlabel="MDS Dimension 1", ylabel="MDS Dimension 2", title=f"MDS of MI Tasks ({display} Band)")
        ax.grid(alpha=0.15, linestyle=':')
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"eeg_mds_optimized_{band}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return all_rdms


# =============================================================================
# 4. Theoretical models and participant-level model RSA
# =============================================================================
def categorical_rdm(labels):
    return np.array([[0. if a == b else 1. for b in labels] for a in labels], dtype=float)


def build_theoretical_rdms():
    level = np.array([0, 1, 1, 2, 2, 3, 3, 3], dtype=float)
    return {
        "Distal-Proximal": categorical_rdm(["Distal"] * 3 + ["Proximal"] * 5),
        "Anatomical Distance": np.abs(level[:, None] - level[None, :]),
        "Joint Category": categorical_rdm(["Hand", "Wrist", "Wrist", "Elbow", "Elbow", "Shoulder", "Shoulder", "Shoulder"]),
        "Movement Type": categorical_rdm(["Open-Close", "Flex-Extend", "Abduct-Adduct", "Pronate-Supinate",
                                          "Flex-Extend", "Pronate-Supinate", "Abduct-Adduct", "Flex-Extend"]),
    }


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order, m = np.argsort(p), len(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(adj, 1.0)
    return q


def run_theoretical_model_rsa(static_features_all):
    models = build_theoretical_rdms()
    model_vecs = {k: upper_triangle_vector(v) for k, v in models.items()}
    rows = []

    for subject_index, subj in enumerate(SUBJECTS):
        subject_id = subject_index + 1
        for band_display, band_key in BAND_DISPLAY.items():
            seed = 42 + subject_index * 100 + sum(map(ord, band_key))
            rdm = calc_crossnobis_rdm(static_features_all[subj][band_key], seed=seed)
            nv = upper_triangle_vector(rdm)
            for model in MODEL_ORDER:
                rows.append({"Subject": subject_id, "Band": band_display, "Model": model,
                             "Spearman_rho": spearmanr(nv, model_vecs[model]).statistic})

    participant_df = pd.DataFrame(rows)
    group_rows = []
    for band in BAND_DISPLAY:
        for model in MODEL_ORDER:
            x = participant_df.loc[(participant_df.Band == band) & (participant_df.Model == model), "Spearman_rho"].to_numpy(float)
            n, mean, sd = len(x), x.mean(), x.std(ddof=1)
            se, tc = sd / np.sqrt(n), stats.t.ppf(.975, n - 1)
            test = stats.ttest_1samp(x, 0, alternative='two-sided')
            group_rows.append({
                "Band": band, "Model": model, "N": n, "Mean": mean, "SD": sd,
                "CI95_Low": mean - tc * se, "CI95_High": mean + tc * se,
                "t": test.statistic, "df": n - 1, "p_raw": test.pvalue,
                "Cohen_dz": mean / sd if sd > 0 else np.nan,
            })

    group_df = pd.DataFrame(group_rows)
    group_df["q_FDR_within_band4"] = np.nan
    for band in BAND_DISPLAY:
        m = group_df.Band == band
        group_df.loc[m, "q_FDR_within_band4"] = bh_fdr(group_df.loc[m, "p_raw"].to_numpy())
    group_df.to_csv(os.path.join(OUTPUT_DIR, "theoretical models result.csv"), index=False)

    rng = np.random.default_rng(12345)
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.5), sharey=True)
    vals_all = participant_df.Spearman_rho.to_numpy(float)
    dmin, dmax = np.nanmin(vals_all), np.nanmax(vals_all)
    fr = dmax - dmin if dmax > dmin else 1.0
    ymin, ymax = dmin - .10 * fr, dmax + .32 * fr

    for ax, band in zip(axes, BAND_DISPLAY.keys()):
        for i, model in enumerate(MODEL_ORDER):
            vals = participant_df.loc[(participant_df.Band == band) & (participant_df.Model == model), "Spearman_rho"].to_numpy(float)
            ax.scatter(np.full(len(vals), i, dtype=float) + rng.normal(0, .045, len(vals)), vals, alpha=.50, s=80)
            row = group_df.loc[(group_df.Band == band) & (group_df.Model == model)].iloc[0]
            mean, low, high = float(row.Mean), float(row.CI95_Low), float(row.CI95_High)
            ax.errorbar(i, mean, yerr=[[mean - low], [high - mean]], fmt='o', capsize=10,
                        linewidth=3, elinewidth=3, markersize=14)
            q = float(row.q_FDR_within_band4)
            star = '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''
            if star:
                ax.text(i, max(np.nanmax(vals), high) + .08 * fr, star, ha='center', va='bottom', fontsize=28)
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=23, ha='right', fontsize=20)
        ax.set_title(band, fontsize=30)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(ymin, ymax)

    axes[0].set_ylabel("Neural–model Spearman's ρ", fontsize=23)
    fig.subplots_adjust(left=.07, right=.99, top=.90, bottom=.24, wspace=.12)
    fig.savefig(os.path.join(OUTPUT_DIR, "theoretical models figure.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\n>>> Theoretical-model RSA completed.")
    return group_df


# =============================================================================
# 5. Static distal-proximal contrast: DP minus AllWithin
# =============================================================================
def run_static_dp_allwithin(static_rdms):
    rows = []
    for si, subj in enumerate(SUBJECTS):
        for band in BANDS:
            r = static_rdms[band][si]
            value = pair_mean(r, DP_PAIRS) - pair_mean(r, ALL_WITHIN_PAIRS)
            rows.append({"Subject": subj, "Band": band, "Contrast": "DP_minus_AllWithin", "Value": value})

    participant_df = pd.DataFrame(rows)
    out = []
    for band in BANDS:
        x = participant_df.loc[participant_df.Band == band, "Value"].to_numpy(float)
        n, mean, sd = len(x), x.mean(), x.std(ddof=1)
        sem, tc = sd / np.sqrt(n), stats.t.ppf(.975, n - 1)
        tst = stats.ttest_1samp(x, 0, alternative='two-sided')
        out.append({
            "Band": band, "Contrast": "DP_minus_AllWithin", "N": n,
            "Mean": mean, "SD": sd, "Median": np.median(x),
            "CI95_Low": mean - tc * sem, "CI95_High": mean + tc * sem,
            "t": tst.statistic, "df": n - 1, "p_raw": tst.pvalue,
            "Cohens_dz": mean / sd if sd > 1e-15 else np.nan,
            "N_Positive": int(np.sum(x > 0)), "Proportion_Positive": float(np.mean(x > 0)),
        })

    group_df = pd.DataFrame(out)
    group_df.to_csv(os.path.join(OUTPUT_DIR, "GROUP_dp_contrasts_RAW_p.csv"), index=False, encoding='utf-8-sig')
    print("\n>>> Static distal-proximal contrast completed.")
    return group_df


# =============================================================================
# 6. Dynamic RSA
#    6.1 Dynamic RDMs and plots
#    6.2 Cluster permutation test
#    6.3 Dynamic distal-proximal contrasts
#    6.4 Band × representational-contrast interaction
# =============================================================================
def one_sample_t_timecourse(x):
    x = np.asarray(x, dtype=float)
    se = x.std(0, ddof=1) / np.sqrt(x.shape[0])
    return np.divide(x.mean(0), se, out=np.zeros(x.shape[1], dtype=float), where=se > 1e-15)


def find_signed_clusters(t_values, threshold):
    supra = np.abs(t_values) >= threshold
    clusters, start, sign = [], None, None
    for i, ok in enumerate(supra):
        if not ok:
            if start is not None:
                clusters.append((start, i - 1, sign))
                start = sign = None
            continue
        s = 1 if t_values[i] > 0 else -1
        if start is None:
            start, sign = i, s
        elif s != sign:
            clusters.append((start, i - 1, sign))
            start, sign = i, s
    if start is not None:
        clusters.append((start, len(t_values) - 1, sign))
    return clusters


def cluster_mass(t_values, start, end):
    return float(np.abs(t_values[start:end + 1]).sum())


def cluster_signflip_time_only(x, n_perm=N_PERM, seed=42):
    """Two-sided participant-level sign-flip cluster permutation with time-wise FWER correction."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must be finite participant × time array")

    threshold = stats.t.ppf(1 - CLUSTER_ALPHA / 2, x.shape[0] - 1)
    obs_t = one_sample_t_timecourse(x)
    clusters = find_signed_clusters(obs_t, threshold)
    masses = [cluster_mass(obs_t, a, b) for a, b, _ in clusters]

    rng, null_max = np.random.default_rng(seed), np.zeros(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1., 1.], size=(x.shape[0], 1))
        pt = one_sample_t_timecourse(x * signs)
        pc = find_signed_clusters(pt, threshold)
        null_max[p] = max((cluster_mass(pt, a, b) for a, b, _ in pc), default=0.0)

    pvals = [(1 + np.sum(null_max >= m)) / (n_perm + 1) for m in masses]
    return {"threshold": threshold, "observed_t": obs_t, "clusters": clusters,
            "cluster_masses": masses, "cluster_p_values": pvals}


def run_dynamic_rsa(dynamic_features_all):
    print("\n>>> [Dynamic RSA] Computing sliding-window RDMs...")
    band_rdms = {b: [] for b in BANDS}
    for subj in SUBJECTS:
        for band in BANDS:
            band_rdms[band].append([
                calc_crossnobis_rdm(dynamic_features_all[subj][band][w]) for w in range(N_WINDOWS)
            ])

    # 6.1 Dynamic plots and original cluster masks
    for band in BANDS:
        display = band.replace('_', ' ').title()
        within = np.zeros((18, N_WINDOWS)); between = np.zeros_like(within)
        dd = np.zeros_like(within); pp = np.zeros_like(within); dp = np.zeros_like(within)

        for s in range(18):
            for w in range(N_WINDOWS):
                r = band_rdms[band][s][w]
                within[s, w], between[s, w] = pair_mean(r, ALL_WITHIN_PAIRS), pair_mean(r, DP_PAIRS)
                dd[s, w], pp[s, w], dp[s, w] = pair_mean(r, DD_PAIRS), pair_mean(r, PP_PAIRS), pair_mean(r, DP_PAIRS)

        def mne_cluster(diff, tag):
            threshold = scipy_t.ppf(1 - .05 / 2, diff.shape[0] - 1)
            np.random.seed(42 + sum(map(ord, band + tag)))
            _, clusters, pvals, _ = permutation_cluster_1samp_test(
                diff, n_permutations=N_PERM, threshold=threshold, tail=0, n_jobs=1, verbose=False
            )
            mask = np.zeros(N_WINDOWS, dtype=bool)
            for c, p in zip(clusters, pvals):
                if p < .05:
                    mask[c[0]] = True
            return mask, clusters, pvals

        sig, clusters, pvals = mne_cluster(between - within, "W_vs_B")
        mask_dd, _, _ = mne_cluster(dp - dd, "DP_DD")
        mask_pp, _, _ = mne_cluster(dp - pp, "DP_PP")

        print(f"\n[{display}] significant global clusters:")
        found = False
        for c, p in zip(clusters, pvals):
            if p < .05:
                print(f"  {time_points[c[0][0]]:.2f}–{time_points[c[0][-1]]:.2f}s, p={p:.4f}")
                found = True
        if not found:
            print("  none")

        # Global dynamic plot
        mw, sw = within.mean(0), within.std(0) / np.sqrt(18)
        mb, sb = between.mean(0), between.std(0) / np.sqrt(18)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(time_points, mw, 'r-', label="Within Category", lw=2)
        ax.fill_between(time_points, mw - sw, mw + sw, color='red', alpha=.2)
        ax.plot(time_points, mb, 'b-', label="Between Category", lw=2)
        ax.fill_between(time_points, mb - sb, mb + sb, color='blue', alpha=.2)
        y0, y1 = ax.get_ylim(); ypos = y1 - (y1 - y0) * .03
        for i in range(N_WINDOWS - 1):
            if sig[i] and sig[i + 1]:
                ax.plot(time_points[i:i + 2], [ypos, ypos], color='black', lw=4, solid_capstyle='round')
        handles, _ = ax.get_legend_handles_labels()
        if sig.any():
            handles.append(Line2D([0], [0], color='black', lw=4, label='Cluster p<0.05'))
        ax.legend(handles=handles, loc='upper right')
        ax.set(xlim=(0, 4), ylim=(y0, y1), xlabel="Time (s)", ylabel="Crossnobis Distance",
               title=f"Dynamic RSA: Within vs Between Category ({display} Band)")
        ax.grid(axis='y', linestyle='--', alpha=.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{band}_cluster_rsa.png"), dpi=300)
        plt.close(fig)

        # DD / PP / DP dynamic plot
        mdd, sdd = dd.mean(0), dd.std(0) / np.sqrt(18)
        mpp, spp = pp.mean(0), pp.std(0) / np.sqrt(18)
        mdp, sdp = dp.mean(0), dp.std(0) / np.sqrt(18)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(time_points, mdd, color='royalblue', label="Within-Distal (Hand & Wrist)", lw=2.5)
        ax.fill_between(time_points, mdd - sdd, mdd + sdd, color='royalblue', alpha=.15)
        ax.plot(time_points, mpp, color='mediumseagreen', label="Within-Proximal (Elbow & Shoulder)", lw=2.5)
        ax.fill_between(time_points, mpp - spp, mpp + spp, color='mediumseagreen', alpha=.15)
        ax.plot(time_points, mdp, color='crimson', label="Between (Distal vs Proximal)", lw=2.5)
        ax.fill_between(time_points, mdp - sdp, mdp + sdp, color='crimson', alpha=.15)
        y0, y1 = ax.get_ylim(); ydd, ypp = y1 - (y1 - y0) * .03, y1 - (y1 - y0) * .07
        for i in range(N_WINDOWS - 1):
            if mask_dd[i] and mask_dd[i + 1]:
                ax.plot(time_points[i:i + 2], [ydd, ydd], color='black', lw=4)
            if mask_pp[i] and mask_pp[i + 1]:
                ax.plot(time_points[i:i + 2], [ypp, ypp], color='dimgray', lw=4)
        handles, _ = ax.get_legend_handles_labels()
        if mask_dd.any():
            handles.append(Line2D([0], [0], color='black', lw=4, label='Between > Within-Distal (p<0.05)'))
        if mask_pp.any():
            handles.append(Line2D([0], [0], color='dimgray', lw=4, label='Between > Within-Proximal (p<0.05)'))
        ax.legend(handles=handles, loc='upper right')
        ax.set(xlim=(0, 4), ylim=(y0, y1 + (y1 - y0) * .05), xlabel="Time (s)", ylabel="Crossnobis Distance",
               title=f"Dynamic RSA: Category Structure ({display} Band)")
        ax.grid(axis='y', linestyle='--', alpha=.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{band}_3categories_rsa.png"), dpi=300)
        plt.close(fig)

    # 6.2-6.3 Dynamic cluster permutation and distal-proximal contrasts
    contrast_names = ["DP_minus_AllWithin", "DP_minus_DD", "DP_minus_PP"]
    contrasts = {band: {name: np.zeros((18, N_WINDOWS)) for name in contrast_names} for band in BANDS}

    for band in BANDS:
        for s in range(18):
            for w in range(N_WINDOWS):
                r = band_rdms[band][s][w]
                dd, pp, dp = pair_mean(r, DD_PAIRS), pair_mean(r, PP_PAIRS), pair_mean(r, DP_PAIRS)
                contrasts[band]["DP_minus_AllWithin"][s, w] = dp - pair_mean(r, ALL_WITHIN_PAIRS)
                contrasts[band]["DP_minus_DD"][s, w] = dp - dd
                contrasts[band]["DP_minus_PP"][s, w] = dp - pp

    seed_base = {"DP_minus_AllWithin": 14001, "DP_minus_DD": 24001, "DP_minus_PP": 34001}
    band_offset = {"alpha": 101, "beta_low": 202, "beta_high": 303}
    rows = []
    for band in BANDS:
        for contrast in contrast_names:
            res = cluster_signflip_time_only(contrasts[band][contrast], seed=seed_base[contrast] + band_offset[band])
            ps = np.asarray(res["cluster_p_values"], dtype=float)
            rows.append({
                "Band": band, "Contrast": contrast, "N_Subjects": 18, "N_Time_Windows": N_WINDOWS,
                "Cluster_Forming_t_Threshold": res["threshold"], "N_Permutations": N_PERM,
                "N_Observed_Clusters": len(res["clusters"]),
                "Min_p_cluster_FWER_time": float(ps.min()) if len(ps) else 1.0,
                "Any_Significant_Cluster_FWER05": bool(np.any(ps < .05)) if len(ps) else False,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "DYNAMIC_family_summary_time_FWER.csv"), index=False, encoding='utf-8-sig')

    # 6.4 Band × representational-contrast interaction
    interaction = contrasts["beta_low"]["DP_minus_AllWithin"] - contrasts["alpha"]["DP_minus_AllWithin"]
    res = cluster_signflip_time_only(interaction, seed=44001)
    mean, sd = interaction.mean(0), interaction.std(0, ddof=1)
    ci = stats.t.ppf(.975, 17) * sd / np.sqrt(18)

    timecourse = pd.DataFrame({
        "Window_Index": np.arange(N_WINDOWS),
        "Window_Start": np.array(TIME_WINDOWS)[:, 0],
        "Window_End": np.array(TIME_WINDOWS)[:, 1],
        "Time_Center": time_points,
        "Mean_Interaction_LowBeta_minus_MuAlpha": mean,
        "CI95_Low": mean - ci,
        "CI95_High": mean + ci,
        "t": res["observed_t"],
    })
    timecourse.to_csv(os.path.join(OUTPUT_DIR, "DYNAMIC_band_x_contrast_timecourse.csv"), index=False, encoding='utf-8-sig')

    cluster_rows = []
    if not res["clusters"]:
        cluster_rows.append({
            "Interaction": "(DP-AllWithin)_LowBeta_minus_(DP-AllWithin)_MuAlpha",
            "Cluster_ID": 0, "Direction": "none", "Start_Index": np.nan, "End_Index": np.nan,
            "Start_Time": np.nan, "End_Time": np.nan, "Cluster_Mass": 0.0,
            "p_cluster_FWER_time": 1.0, "Significant_FWER05": False,
        })
    else:
        for cid, ((a, b, sign), mass, p) in enumerate(zip(res["clusters"], res["cluster_masses"], res["cluster_p_values"]), 1):
            cluster_rows.append({
                "Interaction": "(DP-AllWithin)_LowBeta_minus_(DP-AllWithin)_MuAlpha",
                "Cluster_ID": cid, "Direction": "LowBeta > MuAlpha" if sign > 0 else "MuAlpha > LowBeta",
                "Start_Index": a, "End_Index": b, "Start_Time": TIME_WINDOWS[a][0], "End_Time": TIME_WINDOWS[b][1],
                "Cluster_Mass": mass, "p_cluster_FWER_time": p, "Significant_FWER05": bool(p < .05),
            })

    clusters_df = pd.DataFrame(cluster_rows)
    clusters_df.to_csv(os.path.join(OUTPUT_DIR, "DYNAMIC_band_x_contrast_clusters.csv"), index=False, encoding='utf-8-sig')

    print("\n>>> Direct band × representational-contrast interaction")
    print(clusters_df.to_string(index=False))
    if not clusters_df.Significant_FWER05.any():
        print("No significant direct band × representational-contrast interaction after time-wise FWER correction.")

    return band_rdms, summary, timecourse, clusters_df


# =============================================================================
# 7. Main
# =============================================================================
if __name__ == "__main__":
    print(">>> Loading EEG and extracting static + dynamic features...")
    static_features_all, dynamic_features_all = {}, {}
    for subj in SUBJECTS:
        print(f"  {subj}")
        sf, df = extract_subject_features(subj)
        static_features_all[subj], dynamic_features_all[subj] = sf, df

    # Static RSA: RDM -> Dendrogram -> MDS
    static_rdms = run_static_rsa(static_features_all)

    # Theoretical models
    run_theoretical_model_rsa(static_features_all)

    # Static distal-proximal contrast: AllWithin only
    run_static_dp_allwithin(static_rdms)

    # Dynamic RSA: plots -> cluster permutation -> three DP contrasts -> band × contrast interaction
    random.seed(42)
    np.random.seed(42)
    run_dynamic_rsa(dynamic_features_all)

    print("\n>>> All analyses completed.")
    print(f"All results saved to: {OUTPUT_DIR}")
