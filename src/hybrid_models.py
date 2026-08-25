import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.svm import SVR

from config import MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR
from evaluate_models import MODEL_COMPARISON_PATH, PREDICTIONS_PATH, select_best_model
from modeling_utils import (
    FEATURES,
    TARGET,
    build_model_pipeline,
    compute_metrics,
    split_chronologically,
)

METAHEURISTIC_MODELS = ["WOASVR", "GWOSVR", "PSOSVR"]
HYBRID_MODELS = METAHEURISTIC_MODELS + ["ActualGRIDSVR"]

BOUNDS = [
    (-2.0, 3.0),    # log10(C):      SVR penalty in [0.01, 1000]
    (-3.0, -0.301), # log10(eps):    epsilon in [0.001, 0.5]
    (-3.0, 0.0),    # log10(gamma):  rbf gamma in [0.001, 1]
]

N_AGENTS = 10
N_ITER = 15
SEED = 42

HYBRID_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "hybrid_comparison.csv")
HYBRID_PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "hybrid_predictions.csv")
HYBRID_PARAMS_PATH = os.path.join(OUTPUTS_DIR, "hybrid_best_params.json")
HYBRID_OPTIMIZATION_PATH = os.path.join(OUTPUTS_DIR, "hybrid_optimization.csv")


def decode(params):
    c, eps, gamma = np.power(10.0, params)
    return float(c), float(eps), float(gamma)


def build_svr(params):
    c, eps, gamma = decode(params)
    return SVR(C=c, epsilon=eps, gamma=gamma, kernel="rbf")


def make_fitness(X, y, tscv):
    def fitness(params):
        pipe = build_model_pipeline(build_svr(params))
        scores = cross_val_score(
            pipe, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1
        )
        return float(-scores.mean())

    return fitness


def init_population(rng):
    lb = np.array([b[0] for b in BOUNDS])
    ub = np.array([b[1] for b in BOUNDS])
    return lb, ub, rng.uniform(lb, ub, size=(N_AGENTS, len(BOUNDS)))


def woa_optimize(fitness, rng):
    lb, ub, X = init_population(rng)
    fits = np.array([fitness(x) for x in X])
    best_idx = int(np.argmin(fits))
    best_pos, best_fit = X[best_idx].copy(), fits[best_idx]
    history = []
    for t in range(N_ITER):
        a = 2.0 - 2.0 * t / N_ITER
        for i in range(N_AGENTS):
            r1, r2 = rng.random(), rng.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            if rng.random() < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * best_pos - X[i])
                    X[i] = best_pos - A * D
                else:
                    k = rng.integers(0, N_AGENTS)
                    D = np.abs(C * X[k] - X[i])
                    X[i] = X[k] - A * D
            else:
                l = rng.uniform(-1.0, 1.0)
                D = np.abs(best_pos - X[i])
                X[i] = D * np.exp(l) * np.cos(2.0 * np.pi * l) + best_pos
            X[i] = np.clip(X[i], lb, ub)
            fi = fitness(X[i])
            if fi < best_fit:
                best_fit, best_pos = fi, X[i].copy()
        history.append((t + 1, best_fit))
    return best_pos, best_fit, history


def gwo_optimize(fitness, rng):
    lb, ub, X = init_population(rng)
    fits = np.array([fitness(x) for x in X])
    history = []
    for t in range(N_ITER):
        order = np.argsort(fits)
        alpha, beta, delta = X[order[0]], X[order[1]], X[order[2]]
        a = 2.0 - 2.0 * t / N_ITER
        for i in range(N_AGENTS):
            x1, x2, x3 = [], [], []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(), rng.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                D = np.abs(C * leader - X[i])
                x = leader - A * D
                if leader is alpha:
                    x1 = x
                elif leader is beta:
                    x2 = x
                else:
                    x3 = x
            X[i] = np.clip((x1 + x2 + x3) / 3.0, lb, ub)
            fits[i] = fitness(X[i])
        history.append((t + 1, float(fits.min())))
    best_idx = int(np.argmin(fits))
    return X[best_idx].copy(), float(fits[best_idx]), history


def pso_optimize(fitness, rng):
    lb, ub, X = init_population(rng)
    v_max = 0.2 * (ub - lb)
    V = rng.uniform(-v_max, v_max, size=X.shape)
    fits = np.array([fitness(x) for x in X])
    pbest = X.copy()
    pbest_fit = fits.copy()
    g_idx = int(np.argmin(fits))
    gbest, gbest_fit = pbest[g_idx].copy(), pbest_fit[g_idx]
    w_hi, w_lo, c1, c2 = 0.9, 0.4, 1.5, 1.5
    history = []
    for t in range(N_ITER):
        w = w_hi - (w_hi - w_lo) * t / N_ITER
        for i in range(N_AGENTS):
            r1 = rng.random(size=len(BOUNDS))
            r2 = rng.random(size=len(BOUNDS))
            V[i] = (
                w * V[i]
                + c1 * r1 * (pbest[i] - X[i])
                + c2 * r2 * (gbest - X[i])
            )
            V[i] = np.clip(V[i], -v_max, v_max)
            X[i] = np.clip(X[i] + V[i], lb, ub)
            fits[i] = fitness(X[i])
            if fits[i] < pbest_fit[i]:
                pbest_fit[i], pbest[i] = fits[i], X[i].copy()
            if fits[i] < gbest_fit:
                gbest_fit, gbest = fits[i], X[i].copy()
        history.append((t + 1, float(gbest_fit)))
    return gbest.copy(), float(gbest_fit), history


OPTIMIZERS = {
    "WOASVR": woa_optimize,
    "GWOSVR": gwo_optimize,
    "PSOSVR": pso_optimize,
}


def grid_search_svr(X, y, tscv):
    param_grid = {
        "model__C": np.logspace(-1, 3, 5).tolist(),
        "model__epsilon": [0.001, 0.01, 0.1, 0.5],
        "model__gamma": np.logspace(-3, 0, 4).tolist(),
    }
    pipe = build_model_pipeline(SVR(kernel="rbf"))
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs.best_params_, float(-gs.best_score_)


def plot_convergence(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, hist in history.items():
        it, best = zip(*hist)
        ax.plot(it, best, marker="o", label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best CV RMSE (train 2014-2017)")
    ax.set_title("Metaheuristic hyperparameter optimization convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hybrid_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(combined):
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = [
        "steelblue" if m not in HYBRID_MODELS else "darkorange"
        for m in combined["Model"]
    ]
    for ax, metric in zip(axes, metrics):
        ax.bar(combined["Model"], combined[metric], color=colors)
        ax.set_title(f"{metric} by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hybrid_metric_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "ml_dataset.csv"))
    train, test = split_chronologically(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    years = sorted(train["YearCode"].unique())
    tscv = TimeSeriesSplit(n_splits=max(2, len(years) - 1))
    fitness = make_fitness(X_train, y_train, tscv)

    rng = np.random.default_rng(SEED)
    best_params = {}
    history = {}
    optimization_rows = []
    for name in METAHEURISTIC_MODELS:
        pos, best_fit, hist = OPTIMIZERS[name](fitness, rng)
        c, eps, gamma = decode(pos)
        best_params[name] = {"C": c, "epsilon": eps, "gamma": gamma}
        history[name] = hist
        for it, val in hist:
            optimization_rows.append({"Optimizer": name, "Iteration": it, "Best_CV_RMSE": val})
        print(f"{name}: best CV RMSE = {best_fit:.2f} (C={c:.3g}, eps={eps:.3g}, gamma={gamma:.3g})")

    gs_params, gs_best = grid_search_svr(X_train, y_train, tscv)
    best_params["ActualGRIDSVR"] = {
        k.replace("model__", ""): v for k, v in gs_params.items()
    }
    print(f"ActualGRIDSVR: best CV RMSE = {gs_best:.2f} ({best_params['ActualGRIDSVR']})")

    pd.DataFrame(optimization_rows).to_csv(HYBRID_OPTIMIZATION_PATH, index=False)
    with open(HYBRID_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    predictions = {}
    hybrid_metrics = {}
    for name in HYBRID_MODELS:
        model = SVR(**best_params[name], kernel="rbf")
        pipe = build_model_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        hybrid_metrics[name] = compute_metrics(y_test, pred)
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{name.lower()}.joblib"))
        print(f"{name}: test {hybrid_metrics[name]}")

    hybrid_comp = pd.DataFrame(hybrid_metrics).T
    hybrid_comp = hybrid_comp[["MAE", "RMSE", "MAPE", "R2"]].reset_index().rename(
        columns={"index": "Model"}
    )
    hybrid_comp.to_csv(HYBRID_COMPARISON_PATH, index=False)

    pred_df = test[["State", "YearCode", TARGET]].copy()
    pred_df = pred_df.rename(columns={"YearCode": "Year", TARGET: "Actual Enrollment"})
    for name in HYBRID_MODELS:
        pred_df[f"{name} Prediction"] = predictions[name]
    pred_df.to_csv(HYBRID_PREDICTIONS_PATH, index=False)

    combined = hybrid_comp.copy()
    if os.path.exists(MODEL_COMPARISON_PATH):
        benchmark = pd.read_csv(MODEL_COMPARISON_PATH)
        combined = pd.concat([benchmark, hybrid_comp], ignore_index=True)
    combined.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nCombined model comparison ({len(combined)} models) saved to {MODEL_COMPARISON_PATH}")

    if os.path.exists(PREDICTIONS_PATH):
        all_pred = pd.read_csv(PREDICTIONS_PATH)
        for name in HYBRID_MODELS:
            all_pred[f"{name} Prediction"] = predictions[name]
        all_pred.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Hybrid predictions appended to {PREDICTIONS_PATH}")

    best_model, ranked = select_best_model(combined)
    row = combined[combined["Model"] == best_model].iloc[0]
    selection = (
        f"Best-performing model: {best_model}\n"
        f"Reason: Lowest average ranking across MAE, RMSE, MAPE and R2 on the "
        f"chronological test set (2018-2019). "
        f"Metrics: MAE={row['MAE']:.0f}, RMSE={row['RMSE']:.0f}, "
        f"MAPE={row['MAPE']:.2f}%, R2={row['R2']:.4f}.\n"
        f"Hybrid SVR variants (WOASVR, GWOSVR, PSOSVR) tune SVR hyperparameters "
        f"(C, epsilon, gamma) with metaheuristics (Whale, Grey-Wolf, Particle-Swarm) "
        f"minimizing TimeSeriesSplit CV RMSE on 2014-2017 training data; "
        f"ActualGRIDSVR is the equivalent exhaustive grid-search baseline.\n"
    )
    with open(os.path.join(OUTPUTS_DIR, "model_selection.txt"), "w") as f:
        f.write(selection)
    print("\n" + selection)

    plot_convergence(history)
    plot_metric_comparison(combined)
    print(f"Plots saved: hybrid_convergence.png, hybrid_metric_comparison.png")


if __name__ == "__main__":
    main()