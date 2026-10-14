"""Generate and execute the portfolio notebook.

Builds `notebooks/penguins_analysis.ipynb`, a narrative version of
`analysis.py`, then executes it in-process so that the committed notebook
contains real outputs (tables + inline figures).

Run from the repository root:  python tools/build_notebook.py
"""

import base64
import io
import json
import os

# Must happen before any jupyter import reads the flag (see
# _relax_jupyter_permissions for why this is needed on restricted filesystems).
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "True")

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

OUT_DIR = os.path.join("notebooks")
OUT_NAME = "penguins_analysis.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SETUP = """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, train_test_split)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SPECIES_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
PALETTE = {"Adelie": "#0072B2", "Chinstrap": "#D55E00", "Gentoo": "#009E73"}

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 100)
"""

LOAD = """
df = pd.read_csv("../data/penguins.csv")
print("shape:", df.shape)
df.head()
"""

PROFILE = """
df.info()
"""

MISSING = """
missing = df.isna().sum()
missing[missing > 0].to_frame("n_missing").assign(
    pct=lambda t: (100 * t["n_missing"] / len(df)).round(1))
"""

CLEAN = """
df = df.dropna(subset=FEATURES).copy()
df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]
print("analysed rows:", len(df))
df[FEATURES].describe().round(1)
"""

SCATTER = """
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for sp in SPECIES_ORDER:
    sub = df[df["species"] == sp]
    ax.scatter(sub["bill_length_mm"], sub["bill_depth_mm"], s=50, alpha=0.85,
               color=PALETTE[sp], label=sp, edgecolor="white", linewidth=0.4)
ax.set_xlabel("bill length (mm)")
ax.set_ylabel("bill depth (mm)")
ax.set_title("Bill length vs bill depth separates the three species")
ax.legend(title="species")
plt.show()
"""

BOXES = """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, col, title in zip(axes, ["flipper_length_mm", "body_mass_g"],
                          ["Flipper length (mm)", "Body mass (g)"]):
    sns.boxplot(data=df, x="species", y=col, order=SPECIES_ORDER,
                palette=PALETTE, ax=ax, width=0.55, showfliers=False)
    sns.stripplot(data=df, x="species", y=col, order=SPECIES_ORDER,
                  color="0.25", size=2.2, alpha=0.45, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
plt.tight_layout()
plt.show()
"""

CORR = """
sns.heatmap(df[FEATURES + ["bill_ratio"]].corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True, linewidths=0.5)
plt.title("Feature correlation")
plt.show()
"""

MODEL = """
X, y = df[FEATURES], df["species"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Logistic regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000,
                                                                random_state=RANDOM_STATE))]),
    "Random forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                            random_state=RANDOM_STATE),
}

rows = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    rows.append({"model": name, "cv_accuracy": scores.mean(),
                 "cv_std": scores.std()})
pd.DataFrame(rows).round(3)
"""

REPORT = """
rf = models["Random forest"]
pred = cross_val_predict(rf, X, y, cv=cv)
print(classification_report(y, pred, labels=SPECIES_ORDER))
"""

CM = """
cm = confusion_matrix(y, pred, labels=SPECIES_ORDER)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=SPECIES_ORDER, yticklabels=SPECIES_ORDER)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("Out-of-fold confusion matrix (random forest)")
plt.show()
"""

IMPORTANCE = """
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", color="#4C72B0", figsize=(7, 3.4))
plt.xlabel("importance")
plt.title("Random-forest feature importance")
plt.tight_layout()
plt.show()
imp.round(3)
"""

LIMITATION = """
chinstrap = df[df["species"] == "Chinstrap"]
band = chinstrap[(chinstrap["bill_length_mm"] >= 42) &
                 (chinstrap["bill_length_mm"] <= 48)]
print("Chinstrap with bill length in the 42-48 mm Adelie band: %d of %d"
      % (len(band), len(chinstrap)))
print("Rows missing sex (excluded from any sex-based comparison): %d of %d"
      % (df["sex"].isna().sum(), len(df)))
"""


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# Can body measurements predict penguin species?

**Course:** CISC7204 - Practice 09, Data Science Portfolio

## Research question

Given only the bill length, bill depth, flipper length and body mass of a
penguin, can we identify its species - and which measurement carries the
most information?

This notebook is the narrative companion to `analysis.py`. The script is the
reproducible pipeline; this notebook walks through the reasoning with inline
figures.
"""),
        md("""
## 1. Setup

All random seeds are fixed at `RANDOM_STATE = 42` so the results below can be
reproduced exactly.
"""),
        code(SETUP),
        md("""
## 2. Load the data

Raw file: `data/penguins.csv`, 344 individuals. Source and licence are in the
repository `README.md` (Palmer Archipelago penguins, CC0 1.0, collected by
Gorman, Williams & Fraser 2014).
"""),
        code(LOAD),
        md("### 2.1 Structure and missing values"),
        code(PROFILE),
        code(MISSING),
        md("""
Only 2 rows are missing a body measurement, so dropping them costs
essentially nothing. The 11 rows missing `sex` are kept for the species
model (which does not use `sex`) but excluded from any sex comparison.
"""),
        md("### 2.2 Clean"),
        code(CLEAN),
        md("""
## 3. Exploratory analysis

### 3.1 Bill length vs bill depth

This single scatter plot is the core of the story: the three species occupy
almost disjoint regions. Gentoo has a long but shallow bill, Adelie a short
deep bill, and Chinstrap a long deep bill.
"""),
        code(SCATTER),
        md("""
### 3.2 Overall size

Gentoo is visibly the largest species on both flipper length and body mass -
roughly 27 mm more flipper and 1.4 kg heavier than Adelie on average.
"""),
        code(BOXES),
        md("""
### 3.3 Correlated features

Flipper length and body mass are strongly correlated (r ~ 0.87), so they
carry partly redundant information. That matters for interpreting the
feature-importance ranking later.
"""),
        code(CORR),
        md("""
## 4. Modelling

Two deliberately simple baselines: a scaled logistic regression and a random
forest. Both are evaluated with 5-fold **stratified** cross-validation
(needed because the classes are imbalanced: 152 Adelie vs 68 Chinstrap).
"""),
        code(MODEL),
        md("""
### 4.1 Per-class performance

Out-of-fold predictions over all 342 rows:
"""),
        code(REPORT),
        code(CM),
        md("""
### 4.2 Which measurement matters most?
"""),
        code(IMPORTANCE),
        md("""
## 5. Findings and limitation

**Findings**

1. Body measurements alone identify species with ~98% cross-validated
   accuracy; bill length is the single most informative feature.
2. The three species separate along two axes: bill shape (length vs depth)
   and overall body size (flipper length, mass).
3. Gentoo is the easiest to identify - it is both larger and uniquely
   shallow-billed.
4. The remaining errors are almost entirely Adelie/Chinstrap confusions.

**Limitation**

This is a 3-class problem on 342 individuals from a single archipelago, and
several features are strongly correlated, so the model is descriptive rather
than causal - it cannot tell us *why* the species differ, and it should not be
expected to generalise to penguins from other regions.
"""),
        code(LIMITATION),
    ]
    return nb


def _relax_jupyter_permissions():
    """Make Jupyter able to write its kernel connection files.

    `jupyter_core.paths.secure_write` hardens the connection file with a
    Windows `SetFileSecurity` call, which restricted environments refuse, so
    the kernel can never start. jupyter_core ships an `allow_insecure_writes`
    flag for exactly this situation (mounted/restricted filesystems); the
    connection file is short-lived and holds only an ephemeral loopback port.

    The flag is read from the environment at import time, so it must be set
    before jupyter_core is first imported.
    """
    os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] = "True"
    try:
        import jupyter_core.paths as jpaths
    except ImportError:
        return
    jpaths.allow_insecure_writes = True
    print("relaxed jupyter secure_write (restricted-filesystem workaround)")


def execute(nb):
    """Execute in place so the saved notebook carries real outputs.

    Figures are produced by the `inline` matplotlib backend that ipykernel
    sets up, so they are embedded as base64 PNGs in the .ipynb itself.
    """
    _relax_jupyter_permissions()
    os.environ.setdefault("MPLBACKEND", "module://ipykernel.pylab.backend_inline")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": OUT_DIR}})
    return nb


def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    nb = build()
    print("executing notebook ...")
    execute(nb)
    path = os.path.join(OUT_DIR, OUT_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    n_outputs = sum(len(c.get("outputs", [])) for c in nb.cells
                    if c.cell_type == "code")
    print("wrote %s (%d cells, %d outputs)"
          % (path, len(nb.cells), n_outputs))


if __name__ == "__main__":
    main()
