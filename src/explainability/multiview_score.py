import numpy as np


def compute_mv_score(shap_values, view_names, idx:int = None):
    shap_values_scores = []
    for shap_values_score in shap_values[0]:
        if idx is not None:
            shap_values_score = shap_values_score[idx]
        shap_values_scores.append(np.abs(shap_values_score).sum())
    mv_scores = {view_name : shap_values_score/sum(shap_values_scores)*100 for view_name, shap_values_score in zip(view_names, shap_values_scores)}
    return mv_scores
