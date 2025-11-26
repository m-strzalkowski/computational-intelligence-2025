import os
import json
import numpy as np
import pandas as pd
import optuna
import traceback

# --- 1. Suppress TensorFlow/OneDNN Warnings ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Configuration & Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
OPTUNA_DIR = os.path.join(DATA_DIR, "optuna")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(OPTUNA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# -----------------------
# Helper: JSON Serialization & Saving
# -----------------------
def make_serializable(obj):
    """Recursively converts numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    return obj


def save_trial_json(study_name, trial_number, params, score, history=None):
    study_dir = os.path.join(OPTUNA_DIR, study_name)
    os.makedirs(study_dir, exist_ok=True)

    data = {
        "trial_number": trial_number,
        "score": score,
        "params": make_serializable(params),
        "history": (
            make_serializable(history)
            if history
            else "Not available for this model type"
        ),
    }

    filename = f"trial_{trial_number}.json"
    with open(os.path.join(study_dir, filename), "w") as f:
        json.dump(data, f, indent=2)


# -----------------------
# Data Helpers
# -----------------------
def load_and_prep_data(classes: int):
    from data_prep import prepare_data

    id_path = os.path.join(DATA_DIR, "final_combined_results_2937495-7229337.json")
    homepage_path = os.path.join(DATA_DIR, "homepage_posts_data.json")
    misinfo_path = os.path.join(DATA_DIR, "mssinfo_wykop_posts.json")

    try:
        df, _, _ = prepare_data(
            id_file=id_path, homepage_file=homepage_path, misinfo_file=misinfo_path
        )
    except TypeError:
        df, _, _ = prepare_data(id_path, homepage_path, misinfo_path)

    rounding_conditions = [
        df["scores"] < 0.35,
        (df["scores"] >= 0.35) & (df["scores"] < 0.85),
        df["scores"] >= 0.85,
    ]
    df["scores_rounded"] = np.select(rounding_conditions, [0, 0.7, 1])

    if classes == 3:
        y = df["scores_rounded"].apply(
            lambda x: 0 if x < 0.35 else (2 if x > 0.85 else 1)
        )
    elif classes == 2:
        y = df["scores_rounded"].apply(lambda x: 0 if x == 0 else 1)
    else:
        raise ValueError("Classes must be 2 or 3")

    mask = ~df["clean_text"].duplicated()
    return df.loc[mask, "clean_text"].reset_index(drop=True), y.loc[mask].reset_index(
        drop=True
    )


def get_splits(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# -----------------------
# Pipelines
# -----------------------
def tune_tfidf(model_name: str, classes: int, n_trials: int = 20):
    study_name = f"tfidf_{model_name}_{classes}c"
    # Create study directory immediately to ensure it exists
    os.makedirs(os.path.join(OPTUNA_DIR, study_name), exist_ok=True)

    X, y = load_and_prep_data(classes)
    X_train, X_val, y_train, y_val = get_splits(X, y)

    def objective(trial):
        params = {
            "tfidf_max_feat": trial.suggest_int(
                "tfidf_max_feat", 5000, 20000, step=5000
            ),
            "tfidf_ngram": trial.suggest_int("tfidf_ngram", 1, 2),
            "tfidf_min_df": trial.suggest_int("tfidf_min_df", 1, 5),
        }
        print(f"\n[Trial {trial.number}] {study_name} Config: {params}")

        vect = TfidfVectorizer(
            max_features=params["tfidf_max_feat"],
            ngram_range=(1, params["tfidf_ngram"]),
            min_df=params["tfidf_min_df"],
        )
        Xtr_vec = vect.fit_transform(X_train)
        Xva_vec = vect.transform(X_val)

        history = {}

        if model_name == "logreg":
            c_val = trial.suggest_float("lr_C", 1e-3, 100.0, log=True)
            cw = trial.suggest_categorical("lr_class_weight", [None, "balanced"])
            params.update({"lr_C": c_val, "lr_class_weight": cw})

            clf = LogisticRegression(
                C=c_val, penalty="l2", solver="lbfgs", max_iter=1000, class_weight=cw
            )
            clf.fit(Xtr_vec, y_train)

            # LogReg doesn't have loss history, but we can save iteration count
            history = {
                "n_iter": clf.n_iter_,
                "classes": clf.classes_,
                "note": "LogisticRegression does not provide loss curves.",
            }

        elif model_name == "mlp":
            layers = trial.suggest_int("mlp_layers", 1, 3)
            width = trial.suggest_int("mlp_width", 64, 512, step=64)
            alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True)
            lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)
            params.update(
                {
                    "mlp_layers": layers,
                    "mlp_width": width,
                    "mlp_alpha": alpha,
                    "mlp_lr": lr,
                }
            )

            clf = MLPClassifier(
                hidden_layer_sizes=tuple([width] * layers),
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=50,
                early_stopping=True,
            )
            clf.fit(Xtr_vec, y_train)

            history = {
                "loss_curve": clf.loss_curve_,
                "validation_scores": getattr(clf, "validation_scores_", []),
            }

        elif model_name == "xgboost":
            if xgb is None:
                raise ImportError("XGBoost not installed")
            p_xgb = {
                "n_estimators": trial.suggest_int("xgb_n_est", 100, 500, step=50),
                "max_depth": trial.suggest_int("xgb_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_sub", 0.6, 1.0),
                "use_bal_weights": trial.suggest_categorical(
                    "xgb_bal_weights", [True, False]
                ),
            }
            params.update(p_xgb)

            clf = xgb.XGBClassifier(
                n_estimators=p_xgb["n_estimators"],
                max_depth=p_xgb["max_depth"],
                learning_rate=p_xgb["learning_rate"],
                subsample=p_xgb["subsample"],
                objective="multi:softmax" if classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if classes > 2 else "logloss",
                tree_method="hist",
            )
            sw = (
                compute_sample_weight("balanced", y=y_train)
                if p_xgb["use_bal_weights"]
                else None
            )

            clf.fit(
                Xtr_vec,
                y_train,
                sample_weight=sw,
                eval_set=[(Xtr_vec, y_train), (Xva_vec, y_val)],
                verbose=False,
            )
            history = clf.evals_result()

        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_pred = clf.predict(Xva_vec)
        score = f1_score(y_val, y_pred, average="macro")

        save_trial_json(study_name, trial.number, params, score, history)
        return score

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    _save_summary(study, "tfidf", model_name, classes)


def tune_embeddings(model_name: str, classes: int, n_trials: int = 20):
    study_name = f"pl_emb_{model_name}_{classes}c"  # modify this if the polish embeddings are used
    os.makedirs(os.path.join(OPTUNA_DIR, study_name), exist_ok=True)

    if SentenceTransformer is None:
        raise ImportError("sentence-transformers missing")
    X, y = load_and_prep_data(classes)
    X_train_txt, X_val_txt, y_train, y_val = get_splits(X, y)

    def get_embeddings(transformer_name, texts, split_name):
        safe_name = transformer_name.replace("/", "_")
        path = os.path.join(CACHE_DIR, f"emb_{safe_name}_{split_name}.npy")
        if os.path.exists(path):
            return np.load(path)
        print(f"Generating embeddings for {transformer_name}...")
        model = SentenceTransformer(transformer_name)
        embs = model.encode(texts.tolist(), show_progress_bar=True)
        np.save(path, embs)
        return embs

    def objective(trial):
        emb_model = trial.suggest_categorical(
            "emb_model",
            # ["thenlper/gte-small", "sentence-transformers/paraphrase-MiniLM-L6-v2"], # these are for english
            [
                # "sdadas/stella-pl-retrieval", # way too big
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],  # use for polish language, better than previous
        )
        params = {"emb_model": emb_model}
        print(f"\n[Trial {trial.number}] {study_name} Config: {params}")

        Xtr = get_embeddings(emb_model, X_train_txt, "train")
        Xva = get_embeddings(emb_model, X_val_txt, "val")

        history = {}

        if model_name == "logreg":
            c_val = trial.suggest_float("lr_C", 1e-3, 100.0, log=True)
            cw = trial.suggest_categorical("lr_class_weight", [None, "balanced"])
            params.update({"lr_C": c_val, "lr_class_weight": cw})

            clf = LogisticRegression(
                C=c_val, penalty="l2", solver="lbfgs", max_iter=1000, class_weight=cw
            )
            clf.fit(Xtr, y_train)

            history = {
                "n_iter": clf.n_iter_,
                "classes": clf.classes_,
                "note": "LogisticRegression does not provide loss curves.",
            }

        elif model_name == "mlp":
            layers = trial.suggest_int("mlp_layers", 1, 3)
            width = trial.suggest_int("mlp_width", 64, 512, step=64)
            alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True)
            lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)
            params.update(
                {
                    "mlp_layers": layers,
                    "mlp_width": width,
                    "mlp_alpha": alpha,
                    "mlp_lr": lr,
                }
            )

            clf = MLPClassifier(
                hidden_layer_sizes=tuple([width] * layers),
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=50,
                early_stopping=True,
            )
            clf.fit(Xtr, y_train)

            history = {
                "loss_curve": clf.loss_curve_,
                "validation_scores": getattr(clf, "validation_scores_", []),
            }

        elif model_name == "xgboost":
            p_xgb = {
                "n_estimators": trial.suggest_int("xgb_n_est", 100, 500, step=50),
                "max_depth": trial.suggest_int("xgb_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_sub", 0.6, 1.0),
                "use_bal_weights": trial.suggest_categorical(
                    "xgb_bal_weights", [True, False]
                ),
            }
            params.update(p_xgb)

            clf = xgb.XGBClassifier(
                n_estimators=p_xgb["n_estimators"],
                max_depth=p_xgb["max_depth"],
                learning_rate=p_xgb["learning_rate"],
                subsample=p_xgb["subsample"],
                objective="multi:softmax" if classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if classes > 2 else "logloss",
                tree_method="hist",
            )
            sw = (
                compute_sample_weight("balanced", y=y_train)
                if p_xgb["use_bal_weights"]
                else None
            )
            clf.fit(
                Xtr,
                y_train,
                sample_weight=sw,
                eval_set=[(Xtr, y_train), (Xva, y_val)],
                verbose=False,
            )
            history = clf.evals_result()

        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_pred = clf.predict(Xva)
        score = f1_score(y_val, y_pred, average="macro")

        save_trial_json(study_name, trial.number, params, score, history)
        return score

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    _save_summary(study, "embedding", model_name, classes)


def _save_summary(study, enc, model, classes):
    out_dir = os.path.join(OPTUNA_DIR, f"{enc}_{model}")
    os.makedirs(out_dir, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(out_dir, f"summary_trials_{classes}c.csv"), index=False)

    with open(os.path.join(out_dir, f"best_params_{classes}c.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    N_TRIALS = 5
    experiments = []

    # for model in ["xgboost"]:  # , "logreg", "mlp"]: these were trained before, skipping
    #     for cls in [2, 3]:
    #         experiments.append((tune_tfidf, model, cls))

    for model in ["logreg", "mlp", "xgboost"]:
        for cls in [2, 3]:
            experiments.append((tune_embeddings, model, cls))

    print(f"Starting {len(experiments)} experiments with {N_TRIALS} trials each...")

    pbar = tqdm(experiments, desc="Overall Progress", unit="exp")

    for func, model_name, n_classes in pbar:
        pbar.set_description(
            f"Running: {func.__name__.replace('tune_', '')} -> {model_name} ({n_classes}c)"
        )

        try:
            func(model_name=model_name, classes=n_classes, n_trials=N_TRIALS)
        except Exception as e:
            print(f"\n\n[!] CRASH in {func.__name__} {model_name} {n_classes}c")
            traceback.print_exc()
            continue

    print("\nAll experiments completed.")
