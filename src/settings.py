import os

RANDOM_STATE = 42

DATA_FOLDER = "data"
PROCESSED_DATA_FOLDER = "processed"
METHYLATION_FILANEME = "methylation_PDACTCGA.csv"
RNASEQ_FILANEME = "rnaseq_PDACTCGA.csv"
processed_data_path = os.path.join(DATA_FOLDER, PROCESSED_DATA_FOLDER)
methylation_data_path = os.path.join(processed_data_path, METHYLATION_FILANEME)
rnaseq_data_path = os.path.join(processed_data_path, RNASEQ_FILANEME)
