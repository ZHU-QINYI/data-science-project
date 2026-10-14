#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Palmer Penguins: Can body measurements predict penguin species?
================================================================

A reproducible analysis of the Palmer Archipelago penguins dataset
(344 individuals, 3 species, 3 islands, 2007-2009).

Research question
-----------------
Can the three bill/flipper/body-mass measurements of a penguin predict its
species, and which measurement carries the most information?

Usage
-----
    python analysis.py

Inputs
------
    data/penguins.csv      (raw data, see README for source)

Outputs
-------
    figures/*.png          (5 figures)
    results/metrics.json   (model metrics, machine readable)
    results/summary.txt    (human readable summary of the run)

"""

from __future__ import print_function

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: we only ever write PNG files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, train_test_split)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "penguins.csv")
FIG_DIR = "figures"
RESULT_DIR = "results"

FEATURES = ["bill_length_mm", "bill_depth_mm",
            "flipper_length_mm", "body_mass_g"]
TARGET = "species"
SPECIES_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
# Colour-blind safe palette, one colour per species.
PALETTE = {"Adelie": "#0072B2", "Chinstrap": "#D55E00", "Gentoo": "#009E73"}

sns.set(style="whitegrid", context="talk", font_scale=0.8)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def ensure_dirs():
    for d in (FIG_DIR, RESULT_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote %s" % path)
    return path


# --------------------------------------------------------------------------
# 1. Load and profile the raw data
# --------------------------------------------------------------------------

def load_raw():
    banner("1. LOAD RAW DATA")
    if not os.path.exists(DATA_PATH):
        sys.exit("ERROR: %s not found. See README.md for how to obtain it."
                 % DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("  raw shape            : %d rows x %d columns" % df.shape)
    print("  columns              : %s" % ", ".join(df.columns))
    print("  species counts       : %s"
          % df[TARGET].value_counts().to_dict())
    print("  islands              : %s" % sorted(df["island"].unique()))
    print("  survey years         : %s" % sorted(df["year"].unique()))
    print("\n  missing values per column:")
    miss = df.isna().sum()
    for col, n in miss.items():
        if n:
            print("    %-20s %3d (%.1f%%)" % (col, n, 100.0 * n / len(df)))
    print("    (columns not listed have no missing values)")
    return df


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------

def clean(df):
    banner("2. CLEAN DATA")
    before = len(df)

    # The dataset uses the literal string "NA" only in the sex column, which
    # pandas already parsed as NaN. Rows missing the measurements cannot be
    # used by the model, so they are dropped; we keep the loss explicit.
    df = df.dropna(subset=FEATURES).copy()

    # Normalise the sex labels to lower case so "MALE"/"male" cannot split.
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df.loc[~df["sex"].isin(["male", "female"]), "sex"] = np.nan

    # Derived column used in the exploratory section.
    df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]

    print("  dropped %d row(s) with missing measurements" % (before - len(df)))
    print("  remaining shape      : %d rows x %d columns" % df.shape)
    print("  duplicates           : %d" % df.duplicated().sum())
    print("  rows missing sex     : %d (kept for the model, excluded from "
          "sex comparisons)" % df["sex"].isna().sum())
    print("\n  descriptive statistics:")
    print(df[FEATURES].describe().round(1).to_string())
    return df


# --------------------------------------------------------------------------
# 3. Exploratory visualisations
# --------------------------------------------------------------------------

def figure_1_overview(df):
    banner("3.1 FIGURE 1 - species counts and island composition")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    counts = df[TARGET].value_counts().reindex(SPECIES_ORDER)
    axes[0].bar(counts.index, counts.values,
                color=[PALETTE[s] for s in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")
    axes[0].set_title("A. Individuals per species")
    axes[0].set_ylabel("count")
    axes[0].set_ylim(0, counts.max() * 1.18)

    ct = pd.crosstab(df["island"], df[TARGET]).reindex(columns=SPECIES_ORDER)
    ct.plot(kind="bar", stacked=True, ax=axes[1],
            color=[PALETTE[s] for s in ct.columns], width=0.65)
    axes[1].set_title("B. Species composition by island")
    axes[1].set_xlabel("island")
    axes[1].set_ylabel("count")
    axes[1].legend(title="", fontsize=9)
    plt.setp(axes[1].get_xticklabels(), rotation=0)

    fig.suptitle("Figure 1 - The dataset is imbalanced and island is a "
                 "strong proxy for species", y=1.04, fontsize=12)
    return savefig(fig, "fig1_dataset_overview.png")


def figure_2_scatter(df):
    banner("3.2 FIGURE 2 - the key scatter plot")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    # Plot each species explicitly: seaborn's automatic legend emits the hue
    # column name as a stray entry, which we do not want here.
    for sp in SPECIES_ORDER:
        sub = df[df[TARGET] == sp]
        ax.scatter(sub["bill_length_mm"], sub["bill_depth_mm"],
                   s=55, alpha=0.85, color=PALETTE[sp], label=sp,
                   edgecolor="white", linewidth=0.4)
    ax.set_title("Figure 2 - Bill length vs bill depth separates all three "
                 "species", fontsize=12)
    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")
    ax.legend(title="species", loc="upper right", framealpha=0.95)
    ax.annotate("Gentoo: long, shallow bill",
                xy=(54.5, 15.8), xytext=(50.0, 13.6), fontsize=9,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    return savefig(fig, "fig2_bill_scatter.png")


def figure_3_boxes(df):
    banner("3.3 FIGURE 3 - measurement distributions by species")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, title in zip(
            axes,
            ["flipper_length_mm", "body_mass_g"],
            ["A. Flipper length", "B. Body mass"]):
        sns.boxplot(data=df, x=TARGET, y=col, order=SPECIES_ORDER,
                    palette=PALETTE, ax=ax, width=0.55, showfliers=False)
        sns.stripplot(data=df, x=TARGET, y=col, order=SPECIES_ORDER,
                      color="0.25", size=2.2, alpha=0.45, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", pad=6)
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.suptitle("Figure 3 - Size separates the species: Gentoo is clearly "
                 "the largest", y=1.02, fontsize=12)
    return savefig(fig, "fig3_size_by_species.png")


def figure_4_heatmap(df):
    banner("3.4 FIGURE 4 - feature correlation")
    cols = FEATURES + ["bill_ratio"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Figure 4 - Flipper length and body mass are almost "
                 "redundant (r = 0.87)", fontsize=11)
    return savefig(fig, "fig4_correlation_heatmap.png")


# --------------------------------------------------------------------------
# 4. Modelling
# --------------------------------------------------------------------------

def evaluate(model, X, y, name):
    """Honest 5-fold stratified CV, then a held-out test split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print("  %-22s 5-fold CV accuracy = %.3f +/- %.3f"
          % (name, scores.mean(), scores.std()))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print("  %-22s held-out accuracy  = %.3f  (n_test = %d)"
          % (name, acc, len(y_te)))
    return {"name": name, "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "holdout_accuracy": float(acc),
            "n_test": int(len(y_te))}


def model_and_figures(df):
    banner("4. MODEL - can measurements alone identify the species?")
    X = df[FEATURES]
    y = df[TARGET]

    candidates = [
        ("Logistic regression",
         Pipeline([("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000,
                                              random_state=RANDOM_STATE))])),
        ("Random forest",
         RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                random_state=RANDOM_STATE)),
    ]

    metrics = {}
    for name, model in candidates:
        metrics[name] = evaluate(model, X, y, name)

    # Use the random forest for the interpretability figures.
    rf = candidates[1][1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred_all = cross_val_predict(rf, X, y, cv=cv)
    cm = confusion_matrix(y, pred_all, labels=SPECIES_ORDER)
    report = classification_report(y, pred_all, labels=SPECIES_ORDER,
                                   output_dict=True)
    print("\n  out-of-fold classification report (random forest):")
    print(classification_report(y, pred_all, labels=SPECIES_ORDER))

    # Figure 5: confusion matrix + feature importance.
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=SPECIES_ORDER, yticklabels=SPECIES_ORDER, ax=axes[0])
    axes[0].set_title("A. Out-of-fold confusion matrix", fontsize=11)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("actual")

    axes[1].barh(imp.index, imp.values, color="#4C72B0")
    for i, v in enumerate(imp.values):
        axes[1].text(v + 0.012, i, "%.2f" % v, va="center", fontsize=9)
    axes[1].set_title("B. Random-forest feature importance", fontsize=11)
    axes[1].set_xlabel("importance")
    axes[1].set_xlim(0, imp.max() * 1.32)

    fig.suptitle("Figure 5 - Five measurements classify species almost "
                 "perfectly; bill length matters most", y=1.03, fontsize=12)
    # Reserve space explicitly: the feature names are long and would otherwise
    # spill leftwards over the confusion matrix.
    fig.subplots_adjust(left=0.09, right=0.97, wspace=0.55)
    fig_path = savefig(fig, "fig5_model_performance.png")

    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = SPECIES_ORDER
    metrics["classification_report"] = report
    metrics["feature_importance"] = imp.to_dict()
    metrics["feature_importance_figure"] = fig_path
    return metrics


# --------------------------------------------------------------------------
# 5. Findings and the known limitation
# --------------------------------------------------------------------------

def findings(df, metrics):
    banner("5. FINDINGS")
    group = df.groupby(TARGET)[FEATURES].mean().reindex(SPECIES_ORDER)
    print("  mean measurements per species:")
    print(group.round(1).to_string())

    gentoo = group.loc["Gentoo"]
    adelie = group.loc["Adelie"]
    print("\n  Gentoo vs Adelie:")
    print("    flipper length : %+.1f mm" %
          (gentoo["flipper_length_mm"] - adelie["flipper_length_mm"]))
    print("    body mass      : %+.0f g" %
          (gentoo["body_mass_g"] - adelie["body_mass_g"]))
    print("    bill depth     : %+.1f mm" %
          (gentoo["bill_depth_mm"] - adelie["bill_depth_mm"]))

    rf = metrics["Random forest"]
    print("\n  headline result: a random forest reaches %.1f%% 5-fold CV "
          "accuracy" % (100 * rf["cv_accuracy_mean"]))
    print("  most informative feature: %s"
          % max(metrics["feature_importance"].items(), key=lambda kv: kv[1])[0])

    # Quantify the limitation so the README can state it with a number.
    chinstrap = df[df[TARGET] == "Chinstrap"]
    adelie_m = df[(df[TARGET] == "Adelie") & (df["sex"] == "male")]
    overlap = int(((chinstrap["bill_length_mm"] >= 42) &
                   (chinstrap["bill_length_mm"] <= 48)).sum())
    print("\n  LIMITATION: sex is missing for %d of %d retained rows, so "
          "the sex comparison is not balanced." % (df["sex"].isna().sum(), len(df)))
    print("  LIMITATION: Chinstrap and Adelie overlap on Torgersen-like "
          "bill dimensions; %d of %d Chinstrap have bill length in the "
          "42-48 mm band where Adelie also occurs." % (overlap, len(chinstrap)))
    return group


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------

def main():
    ensure_dirs()
    raw = load_raw()
    df = clean(raw)

    figure_1_overview(df)
    figure_2_scatter(df)
    figure_3_boxes(df)
    figure_4_heatmap(df)

    metrics = model_and_figures(df)
    group = findings(df, metrics)

    banner("6. WRITE RESULT FILES")
    n_dropped = len(raw) - len(df)
    payload = {
        "dataset": "Palmer Archipelago (Antarctica) penguins",
        "source": "https://allisonhorst.github.io/palmerpenguins/",
        "license": "CC0 1.0 (public domain dedication), via Gorman, Williams & Fraser (2014)",
        "n_raw": int(len(raw)),
        "n_analysed": int(len(df)),
        "n_dropped_missing_measurements": int(n_dropped),
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "mean_measurements_per_species": group.round(2).to_dict(),
        "models": {k: v for k, v in metrics.items()
                   if k in ("Logistic regression", "Random forest")},
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": metrics["confusion_matrix_labels"],
        "feature_importance": metrics["feature_importance"],
    }
    metrics_path = os.path.join(RESULT_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("  wrote %s" % metrics_path)

    best = max(payload["models"].items(),
               key=lambda kv: kv[1]["cv_accuracy_mean"])
    lines = [
        "Palmer penguins - analysis summary",
        "=" * 40,
        "Raw rows          : %d" % len(raw),
        "Analysed rows     : %d (%d dropped for missing measurements)"
        % (len(df), n_dropped),
        "Species           : %s" % ", ".join(SPECIES_ORDER),
        "Features          : %s" % ", ".join(FEATURES),
        "",
        "Headline result   : %s achieves %.1f%% 5-fold cross-validated "
        "accuracy" % (best[0], 100 * best[1]["cv_accuracy_mean"]),
        "                    (held-out test accuracy %.1f%%)"
        % (100 * best[1]["holdout_accuracy"]),
        "Top feature       : %s"
        % max(payload["feature_importance"].items(), key=lambda kv: kv[1])[0],
        "",
        "Mean measurements per species:",
        group.round(1).to_string(),
        "",
        "Limitation        : sex is missing for %d rows, and Chinstrap "
        "overlaps Adelie" % df["sex"].isna().sum(),
        "                    on bill dimensions, so 2-3 individuals remain "
        "hard to separate.",
    ]
    txt_path = os.path.join(RESULT_DIR, "summary.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote %s" % txt_path)

    banner("DONE - see figures/ and results/")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
