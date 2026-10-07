"""Utilitaires partagés par les 4 notebooks du TP4.

Centralise : chargement de PneumoniaMNIST (et des jeux auxiliaires MedMNIST),
métriques, simulation de bruit de labels, export des figures/tables vers
le dossier de l'article LaTeX (article/figures, article/tables) et des
résultats bruts (results/).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "article" / "figures"
TAB_DIR = ROOT / "article" / "tables"

SEED = 42
CLASS_NAMES = ["Normal", "Pneumonie"]          # 0 = Normal, 1 = Pneumonie (classe positive)
CLASS_NAMES_EN = ["Normal", "Pneumonia"]

# Palette cohérente entre notebooks (violet / cobalt du style HTML des notebooks)
COLORS = {
    "logreg": "#5e35b1", "sga": "#8e24aa", "nb": "#f9a825", "knn": "#00897b",
    "tree": "#6d4c41", "ada": "#ef6c00", "gb": "#e53935", "svm_lin": "#1e88e5",
    "svm_rbf": "#3949ab", "cnn": "#546e7a", "edl": "#d81b60",
    "normal": "#1e88e5", "pneumonia": "#e53935",
}


# --------------------------------------------------------------------------
# Reproductibilité et style
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def setup_plots() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": True,
        "pdf.fonttype": 42,
    })


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------
def load_medmnist(flag: str = "pneumoniamnist", size: int = 28):
    """Télécharge (une seule fois, dans data/) et renvoie un jeu MedMNIST.

    Retourne un dict {split: (images uint8 [n, H, W(, C)], labels int [n])}.
    """
    import medmnist
    from medmnist import INFO

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, INFO[flag]["python_class"])
    out = {}
    for split in ("train", "val", "test"):
        ds = cls(split=split, download=True, root=str(DATA_DIR), size=size)
        out[split] = (ds.imgs, ds.labels.reshape(-1).astype(int))
    return out


def flatten(imgs: np.ndarray) -> np.ndarray:
    """Images uint8 -> vecteurs float32 dans [0, 1]."""
    return imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0


def load_pneumonia():
    """PneumoniaMNIST aplati : X_train, y_train, X_val, y_val, X_test, y_test (+ images)."""
    d = load_medmnist("pneumoniamnist")
    (itr, ytr), (iva, yva), (ite, yte) = d["train"], d["val"], d["test"]
    return {
        "X_train": flatten(itr), "y_train": ytr,
        "X_val": flatten(iva), "y_val": yva,
        "X_test": flatten(ite), "y_test": yte,
        "img_train": itr, "img_val": iva, "img_test": ite,
    }


# --------------------------------------------------------------------------
# Régression logistique par SGA (même implémentation que dans la Partie 1)
# --------------------------------------------------------------------------
def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LogisticRegressionSGA:
    """Régression logistique binaire entraînée par Stochastic Gradient Ascent (cf. Partie 1, Task 1.2)."""

    def __init__(self, lr=0.01, decay=1e-4, l2=1e-4, epochs=20, seed=SEED):
        self.lr, self.decay, self.l2, self.epochs, self.seed = lr, decay, l2, epochs, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, m = X.shape
        self.theta, self.bias, t = np.zeros(m), 0.0, 0
        for _ in range(self.epochs):
            for i in rng.permutation(n):
                alpha = self.lr / (1 + self.decay * t)
                error = y[i] - sigmoid(X[i] @ self.theta + self.bias)
                self.theta += alpha * (error * X[i] - self.l2 * self.theta)
                self.bias += alpha * error
                t += 1
        return self

    def decision_function(self, X):
        return X @ self.theta + self.bias

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


# --------------------------------------------------------------------------
# Bruit de labels (Task 3.2)
# --------------------------------------------------------------------------
def minority_class(y: np.ndarray) -> int:
    values, counts = np.unique(y, return_counts=True)
    return int(values[np.argmin(counts)])


def flip_minority_labels(y: np.ndarray, rate: float, seed: int = SEED, minority: int | None = None):
    """Inverse une fraction `rate` des labels de la classe minoritaire (cas binaire).

    Retourne (y_bruite, indices_inverses).
    """
    minority = minority_class(y) if minority is None else minority
    rng = np.random.default_rng(seed)
    idx_min = np.flatnonzero(y == minority)
    n_flip = int(round(rate * len(idx_min)))
    flipped = rng.choice(idx_min, size=n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flipped] = 1 - y_noisy[flipped]
    return y_noisy, np.sort(flipped)


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def scores(model, X) -> np.ndarray:
    """Score continu de la classe positive (proba si disponible, sinon marge du SVM)."""
    if hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    """Métriques binaires, classe positive = 1 (Pneumonie)."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["ap"] = average_precision_score(y_true, y_score)
    return m


def bootstrap_ci(y_true, y_pred, y_score, metric: str, n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05):
    """Intervalle de confiance bootstrap (percentile) d'une métrique de `binary_metrics`."""
    fn = {"accuracy": lambda t, p, s: accuracy_score(t, p), "f1": lambda t, p, s: f1_score(t, p, zero_division=0),
          "roc_auc": lambda t, p, s: roc_auc_score(t, s), "ap": lambda t, p, s: average_precision_score(t, s)}.get(metric)
    if fn is None:
        fn = lambda t, p, s: binary_metrics(t, p, s)[metric]
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_score = None if y_score is None else np.asarray(y_score)
    n, vals = len(y_true), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y_true[i])) < 2:
            continue
        vals.append(fn(y_true[i], y_pred[i], None if y_score is None else y_score[i]))
    return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))


def expected_calibration_error(y_true, prob_pos, n_bins: int = 15) -> float:
    """ECE binaire calculée sur la confiance de la classe prédite."""
    y_true, prob_pos = np.asarray(y_true), np.asarray(prob_pos)
    pred = (prob_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, prob_pos, 1 - prob_pos)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


# --------------------------------------------------------------------------
# Export (figures, tables LaTeX, résultats bruts)
# --------------------------------------------------------------------------
def save_fig(fig, name: str) -> Path:
    """Sauvegarde une figure en PDF vectoriel dans article/figures/ (prête pour Overleaf)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    return path


def save_results(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path


def load_results(name: str):
    path = RESULTS_DIR / name
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_latex_table(df: pd.DataFrame, name: str, caption: str, label: str,
                     float_fmt: str = "{:.3f}", bold_max: list[str] | None = None,
                     column_format: str | None = None) -> Path:
    """Écrit un tableau booktabs dans article/tables/<name>.tex (inclus par \\input)."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    body = df.copy()
    for col in body.columns:
        if pd.api.types.is_float_dtype(body[col]):
            best = body[col].max() if bold_max and col in bold_max else None
            body[col] = [
                (r"\textbf{" + float_fmt.format(v) + "}") if best is not None and np.isclose(v, best)
                else float_fmt.format(v)
                for v in body[col]
            ]
    column_format = column_format or ("l" + "c" * (body.shape[1] - 1))
    latex = body.to_latex(index=False, escape=False, column_format=column_format).strip()
    # adjustbox : la table n'est réduite que si elle dépasse la largeur du texte
    tex = (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n{latex}\n\\end{{adjustbox}}\n\\end{{table}}\n"
    )
    path = TAB_DIR / f"{name}.tex"
    path.write_text(tex, encoding="utf-8")
    write_french_table(name)
    return path


# --------------------------------------------------------------------------
# Version française des tables (article/main_fr.tex lit article/tables_fr/)
# --------------------------------------------------------------------------
TAB_DIR_FR = ROOT / "article" / "tables_fr"

CAPTIONS_FR = {
    "tab_clean": "Performances en test de tous les modèles classiques entraînés sur des étiquettes propres (classe "
                 "positive~: pneumonie~; intervalle de confiance bootstrap à 95\\,\\% pour le F1). Meilleure valeur de "
                 "chaque colonne en gras.",
    "tab_noise5": "Robustesse à l'inversion de 5\\,\\% des étiquettes d'entraînement de la classe minoritaire (normale). "
                  "\\emph{référence}~: hyperparamètres choisis sur la validation~; \\emph{robuste}~: régularisation plus "
                  "forte~; \\emph{recalibrée}~: modèle de référence dont le seuil de décision est re-choisi sur la "
                  "validation propre. Propre~: une exécution~; bruité~: moyenne$\\pm$écart-type sur 10 tirages. "
                  "$\\Delta$ = bruité $-$ propre (le test est toujours propre).",
    "tab_edl": "Performances en test (propre) après entraînement avec des étiquettes minoritaires inversées "
               "(moyenne$\\pm$écart-type sur les graines~: 5 pour le modèle classique, 3 pour les CNN). \\emph{recal.}~: "
               "seuil de décision re-choisi sur la validation propre. ECE~: erreur de calibration attendue (plus faible "
               "= meilleure).",
    "tab_uncertainty": "AUROC des scores d'incertitude (moyenne$\\pm$écart-type sur 3 graines) pour détecter les images "
                       "de test mal classées, les images hors distribution (échographies BreastMNIST, scanners "
                       "OrganCMNIST) et les étiquettes d'entraînement inversées. \\emph{Désacc.}~: "
                       "$1-\\hat p(\\text{étiquette fournie})$.",
}
HEADERS_FR = {
    "Model": "Modèle", "Config.": "Config.", "Acc.": "Exact.", "Prec.": "Préc.", "Recall": "Rappel", "Spec.": "Spéc.",
    "F1 [95\\% CI]": "F1 [IC 95\\,\\%]", "ROC-AUC": "AUC ROC", "AP": "AP", "ECE": "ECE", "Noise": "Bruit",
    "Acc. (recal.)": "Exact. (recal.)", "Spec. (recal.)": "Spéc. (recal.)", "Train noise": "Bruit (entraîn.)",
    "Model (uncertainty)": "Modèle (incertitude)", "Misclass.": "Erreurs", "OOD breast": "OOD échographie",
    "OOD CT": "OOD scanner", "Flipped ($u$)": "Inversées ($u$)", "Flipped (disagr.)": "Inversées (désacc.)",
}
CELLS_FR = {
    "LogReg-SGA (ours)": "LogReg-SGA (nôtre)", "Gaussian NB": "Bayes naïf gaussien", "Decision tree": "Arbre de décision",
    "Linear SVM": "SVM linéaire", "RBF-SVM": "SVM RBF", "baseline": "référence", "robust": "robuste",
    "recalibrated": "recalibrée", "CNN softmax (entropy)": "CNN softmax (entropie)",
}


def _cell_fr(cell: str, header: bool) -> str:
    c = cell.strip()
    if header:
        for suffix_en, suffix_fr in [(" clean", " propre"), (" noisy", " bruité")]:
            if c.endswith(suffix_en):
                return HEADERS_FR.get(c[: -len(suffix_en)], c[: -len(suffix_en)]) + suffix_fr
        if c.startswith("$\\Delta$"):
            return "$\\Delta$" + HEADERS_FR.get(c[len("$\\Delta$"):], c[len("$\\Delta$"):])
        return HEADERS_FR.get(c, c)
    c = c.replace(" (classical)", " (classique)")
    if c not in CELLS_FR:                       # noms de modèles suivis d'un suffixe, p. ex. « RBF-SVM (classique) »
        for en, fr in CELLS_FR.items():
            if c.startswith(en + " "):
                c = fr + c[len(en):]
    c = CELLS_FR.get(c, c)
    c = re.sub(r"\[(\d\.\d+), (\d\.\d+)\]", r"[\1~; \2]", c)       # IC : séparateur « ; » avec virgule décimale
    return re.sub(r"(\d)\.(\d)", r"\1,\2", c)


def french_table(tex: str, name: str) -> str:
    """Traduit une table LaTeX produite par save_latex_table (légende, entêtes, cellules, virgule décimale)."""
    out, in_body, header_done = [], False, False
    for line in tex.split("\n"):
        if line.startswith("\\caption{") and name in CAPTIONS_FR:
            line = f"\\caption{{{CAPTIONS_FR[name]}}}"
        elif line.startswith("\\toprule"):
            in_body = True
        elif line.startswith("\\bottomrule"):
            in_body = False
        elif in_body and line.rstrip().endswith("\\\\") and not line.startswith("\\midrule"):
            cells = line.rstrip()[:-2].split(" & ")
            line = " & ".join(_cell_fr(c, header=not header_done) for c in cells) + " \\\\"
            header_done = True
        out.append(line)
    return "\n".join(out)


def write_french_table(name: str) -> Path:
    TAB_DIR_FR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR_FR / f"{name}.tex"
    path.write_text(french_table((TAB_DIR / f"{name}.tex").read_text(encoding="utf-8"), name), encoding="utf-8")
    return path
