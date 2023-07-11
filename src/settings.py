import os

RANDOM_STATE = 42

CODE_FOLDER = "src"
OUTPUT_FOLDER = "outputs"
RESULTS_FOLDER = "results"
results_path = os.path.join(OUTPUT_FOLDER, RESULTS_FOLDER)
OPTIMIZATION_FOLDER = "optimization"
optimization_path = os.path.join(OUTPUT_FOLDER, OPTIMIZATION_FOLDER)
FEATURE_IMPORTANCE_FOLDER = "feature_importance"
feature_importance_path = os.path.join(OUTPUT_FOLDER, FEATURE_IMPORTANCE_FOLDER)
WHATIF_FOLDER = "whatif"
whatif_path = os.path.join(OUTPUT_FOLDER, WHATIF_FOLDER)
DATA_FOLDER = "data"
PROCESSED_DATA_FOLDER = "processed"
CLINICAL_FILANEME = "paad_tcga_clinical_data.tsv"
METHYLATION_FILANEME = "methylation_PDACTCGA.csv"
RNASEQ_FILANEME = "rnaseq_PDACTCGA.csv"
processed_data_path = os.path.join(DATA_FOLDER, PROCESSED_DATA_FOLDER)
clinical_data_path = os.path.join(processed_data_path, CLINICAL_FILANEME)
methylation_data_path = os.path.join(processed_data_path, METHYLATION_FILANEME)
rnaseq_data_path = os.path.join(processed_data_path, RNASEQ_FILANEME)
