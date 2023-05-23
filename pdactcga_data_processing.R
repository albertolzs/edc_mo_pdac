
#################
##  LIBRARIES  ##
#################

library(Gmisc)

##############
##  SCRIPT  ##
##############

# data directory
folder_data <- pathJoin("..", "data")
folder_raw_data <- pathJoin(folder_data, "raw")
folder_processed_data <- pathJoin(folder_data, "processed")
# choose the code
CANCER_CODE <- "PDAC"
PLATFORM_CODE <- "TCGA"
# read cancer data
filename_rds <- pathJoin(folder_raw_data, paste0("rnaseqnorm_meth_rppa_mirna_", CANCER_CODE, PLATFORM_CODE, ".rds"))
cancer_data <- readRDS(filename_rds)
experiments(cancer_data)
filename_samples <- pathJoin(folder_raw_data, paste0("patients_", CANCER_CODE, PLATFORM_CODE, ".csv"))
samples <- read.csv2(filename_samples)
head(samples)
training_patients <- samples[samples$dataset_type == "training", "patients"]
testing_patients <- samples[samples$dataset_type == "testing", "patients"]

# methylation
##############

# get array
array_id <- 4
omics_data <- assay(cancer_data, array_id)
colnames(omics_data) <- substr(colnames(omics_data), 1, 12)
print(dim(omics_data))
print(omics_data[1:6, 1:6])

print("select somatic chromosomes")
somatic_gene <- rowData(experiments(cancer_data)[[array_id]])$Chromosome %in% c(1:22)
print(sum(somatic_gene))
print("select cpgs with gene symbol")
cpg_with_symbol <- !is.na(rowData(experiments(cancer_data)[[array_id]])$Gene_Symbol)
print(sum(cpg_with_symbol))
filtered_cpg <- somatic_gene + cpg_with_symbol
print("filtering somatic cpgs with gene symbol")
filtered_cpg <- filtered_cpg == 2
print(sum(filtered_cpg))
rowData(experiments(cancer_data)[[array_id]])$filtered_cpg <- filtered_cpg
omics_data <- omics_data[rowData(experiments(cancer_data)[[array_id]])$filtered_cpg,]
print(dim(omics_data))
print(omics_data[1:6, 1:6])
#saving files
filename_data <- pathJoin(folder_processed_data, paste0("methylation_", CANCER_CODE, PLATFORM_CODE, ".csv"))
write.csv2(x = omics_data, file = filename_data)

# rnaseq
##############

# get array
array_id <- 2
omics_data <- assay(cancer_data, array_id)
colnames(omics_data) <- substr(colnames(omics_data), 1, 12)
print(dim(omics_data))
print(omics_data[1:6, 1:6])
#saving files
filename_data <- pathJoin(folder_processed_data, paste0("rnaseq_", CANCER_CODE, PLATFORM_CODE, ".csv"))
write.csv2(x = omics_data, file = filename_data)

# mirna
##############

# get array
array_id <- 1
omics_data <- assay(cancer_data, array_id)
colnames(omics_data) <- substr(colnames(omics_data), 1, 12)
print(dim(omics_data))
print(omics_data[1:6, 1:6])
#saving files
filename_data <- pathJoin(folder_processed_data, paste0("mirna_", CANCER_CODE, PLATFORM_CODE, ".csv"))
write.csv2(x = omics_data, file = filename_data)

# rppa
##############

# get array
array_id <- 3
omics_data <- assay(cancer_data, array_id)
colnames(omics_data) <- substr(colnames(omics_data), 1, 12)
print(dim(omics_data))
print(omics_data[1:6, 1:6])
#saving files
filename_data <- pathJoin(folder_processed_data, paste0("rppa_", CANCER_CODE, PLATFORM_CODE, ".csv"))
write.csv2(x = omics_data, file = filename_data)
