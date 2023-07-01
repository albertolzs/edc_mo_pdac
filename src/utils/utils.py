import os
import dill


def transform_full_dataset(Xs: list, fit_pipelines: bool, results_folder: str, features_per_component: int = 0,
                           optimization_folder: str = ""):
    transformed_Xs = []
    for idx_pipeline, X in enumerate(Xs):
        if fit_pipelines:
            pipeline_name = f"pipeline{idx_pipeline}_fsplit0_ssplit0.pkl"
            with open(os.path.join(optimization_folder, pipeline_name), 'rb') as f:
                pipeline = dill.load(f)
            if "featureselectionnmf" in pipeline.named_steps.keys():
                pipeline.set_params(**{"featureselectionnmf__n_features_per_component": features_per_component})
                pipeline[-2].select_features()
            pipeline.fit(X)
            with open(os.path.join(results_folder, f"pipeline{idx_pipeline}.pkl"), 'wb') as f:
                dill.dump(pipeline, f)
        else:
            pipeline_name = f"pipeline{idx_pipeline}.pkl"
            with open(os.path.join(results_folder, pipeline_name), 'rb') as f:
                pipeline = dill.load(f)
        transformed_X = pipeline.transform(X)
        transformed_Xs.append(transformed_X)
    return transformed_Xs
