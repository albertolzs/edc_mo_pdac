
#################
##  LIBRARIES  ##
#################

library(curatedTCGAData)
library(TCGAutils)
library(UpSetR)
library(Gmisc)

##############
##  SCRIPT  ##
##############

# data directory
folder_data <- pathJoin("..", "data")
folder_raw_data <- pathJoin(folder_data, "raw")
# list diseaseCodes
data('diseaseCodes', package = "TCGAutils")
diseaseCodes
# choose the code
TCGA_CODE <- "PAAD"
CANCER_CODE <- "PDAC"
PLATFORM_CODE <- "TCGA"
# check experiments
curatedTCGAData(diseaseCode = TCGA_CODE, version = "2.0.1")
# filter by omics
omics <- c("RNASeq2GeneNorm*", "RPPA*", "*Methylation*", "*miRNA*")
curatedTCGAData(diseaseCode = TCGA_CODE, assays = omics, version = "2.0.1")
# download data
cancer_data <- curatedTCGAData(diseaseCode = TCGA_CODE, assays = omics, dry.run = FALSE, version = "2.0.1")
cancer_data
colnames(cancer_data)
rownames(cancer_data)
# filter by primary tumor
sampleTables(cancer_data)
tums <- TCGAsampleSelect(barcodes = colnames(cancer_data), sampleCodes = "01")
cancer_data <- cancer_data[,tums,]
# visualize data
upsetSamples(cancer_data)
# clinical data
clinical_data <- colData(cancer_data)[, c('patient.patient_id', 'histological_type')]
dim(clinical_data)
# remove histological nan
table(colData(cancer_data)$histological_type)
clinical_data <- clinical_data[!is.na(clinical_data$histological_type),]
# select histology
histological_type <- names(sort(table(clinical_data$histological_type), decreasing = T))[1]
# select patients
patients <- rownames(clinical_data[clinical_data$histological_type == histological_type,])
# filter patients by histology
cancer_data <- cancer_data[,colnames(cancer_data)[substr(colnames(cancer_data), 1, 12) %in% patients],]
cancer_data
# visualize filtered data
upsetSamples(cancer_data)
# save object
filename_rds <- pathJoin(folder_raw_data, paste0("rnaseqnorm_meth_rppa_mirna_", CANCER_CODE, PLATFORM_CODE, ".rds"))
saveRDS(cancer_data, filename_rds)
exportClass(cancer_data, dir = folder_raw_data, fmt = "csv", ext = ".csv")
# split dataset in 60% as training set and remaining 40% as testing set
patients <- as.data.frame(patients)
dataset_type <- rep("training", nrow(patients))
dataset_type[1:as.integer(nrow(patients)*0.4)] <- "testing"
set.seed(42)
patients$dataset_type <- sample(dataset_type)
table(patients$dataset_type)
table(patients$dataset_type)/length(patients$dataset_type)
filename_samples <- pathJoin(folder_raw_data, paste0("patients_", CANCER_CODE, PLATFORM_CODE, ".csv"))
write.csv2(x = patients, file = filename_samples)
