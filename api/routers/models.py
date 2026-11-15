from fastapi import APIRouter
from api.services import data_service as ds

router = APIRouter()


@router.get("/models/comparison")
def model_comparison():
    df = ds.get_model_comparison()
    return df.to_dict(orient="records")


@router.get("/models/best")
def best_model():
    df = ds.get_model_comparison()
    # lowest avg rank across MAE, RMSE, MAPE, R2
    rank_df = df.copy()
    for col in ["MAE", "RMSE", "MAPE"]:
        rank_df[f"{col}_rank"] = rank_df[col].rank(ascending=True)
    rank_df["R2_rank"] = rank_df["R2"].rank(ascending=False)
    rank_df["avg_rank"] = rank_df[["MAE_rank", "RMSE_rank", "MAPE_rank", "R2_rank"]].mean(axis=1)
    best = rank_df.loc[rank_df["avg_rank"].idxmin()]
    return {
        "model": best["Model"],
        "metrics": {
            "MAE": float(best["MAE"]),
            "RMSE": float(best["RMSE"]),
            "MAPE": float(best["MAPE"]),
            "R2": float(best["R2"]),
        },
    }


@router.get("/feature-importance")
def feature_importance():
    df = ds.get_feature_importance()
    return df.to_dict(orient="records")


@router.get("/best-params")
def best_params():
    return ds.get_best_params()


@router.get("/hybrid-params")
def hybrid_params():
    return ds.get_hybrid_best_params()
