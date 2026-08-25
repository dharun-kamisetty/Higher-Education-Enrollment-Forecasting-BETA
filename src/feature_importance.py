import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from config import MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR
from modeling_utils import aggregate_state_importance, get_transformed_feature_names

FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")

TREE_MODELS = ["Random Forest", "XGBoost", "LightGBM"]


def load_fitted_model(name):
    path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.joblib")
    return joblib.load(path)


def get_importances(pipe):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = get_transformed_feature_names(pipe.named_steps["prep"])
    values = model.feature_importances_
    df = pd.DataFrame({"feature": names, "importance": values})
    df = df[df["importance"] > 0]
    return aggregate_state_importance(df)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    all_df = []
    for name in TREE_MODELS:
        pipe = load_fitted_model(name)
        imp = get_importances(pipe)
        if imp is None:
            print(f"{name}: no feature_importances_ available")
            continue
        imp = imp.copy()
        imp["model"] = name
        all_df.append(imp)
        print(f"\n{name} feature importance (top 5):")
        print(imp.head().to_string(index=False))

    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        combined.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, name in zip(axes, TREE_MODELS):
            sub = combined[combined["model"] == name]
            top = sub.head(10).iloc[::-1]
            ax.barh(top["feature"], top["importance"], color="steelblue")
            ax.set_title(f"{name}")
            ax.set_xlabel("Importance")
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Feature Importance (tree-based models)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nFeature importance saved to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
