# Explainable multi-omics deep clustering reveals an important role of DNA methylation in pancreatic ductal adenocarcinoma

## Abstract

Patients with pancreatic ductal adenocarcinoma (PDAC) have the lowest survival rate among all cancer patients in Europe. Since western societies have the highest incidence of pancreatic cancer, it has been projected that PDAC will soon become the second leading cause of cancer-related deaths. The main challenge of PDAC treatment is that patients with similar somatic genotypes exhibit a wide range of disease phenotypes. Artificial Intelligence (AI) is currently transforming the field of healthcare and represents a promising technology for integrating various datasets and optimizing evidence-based decision making. However, the interpretability of most AI models is limited and it is challenging to understand how and why a decision is made. In this study, we developed a deep clustering model for PDAC patient stratification using integrated methylation and gene expression data. We placed a specific emphasis on model explainability, with the aim to understand hidden multi-modal patterns learned by the model. The model resulted in two subgroups of PDAC patients with different prognoses and biological factors. We performed several follow-up analyses to measure the relative contribution of each modality to the clustering solution. This multi-omics profile analysis revealed an important role of DNA methylation, partially supported by previous experimental studies. We also show how the model learned the underlying patterns in a multi-modal setting, where individual hidden neurons are specialized either in single data modalities or their combinations. We hope this study will help to promote more explainable AI in real-world clinical applications, where the knowledge of the decision factors is crucial. The code of this project is publicly available in GitHub (https://github.com/albertolzs/edc_mo_pdac).

## Results

![survival](https://github.com/albertolzs/edc_mo_pdac/assets/140154262/35acd9ed-2706-4a32-bd94-3f63476d68f4)

Kaplan-Meier estimation for the survival function between clusters. For each estimation, the 95% confidence interval is shown. The log-rank test was applied for obtaining the p-value. The coefficient of the Cox’s proportional hazard model returned a hazard ratio of 1.75 (1.24-2.49, 95% confidence interval).

![featureimportance_2](https://github.com/albertolzs/edc_mo_pdac/assets/140154262/b0ea530b-2c47-485f-b542-91a8da6dd482)
Top-25 most important feature across feature importance methods and weights estimated in the training data. The order corresponds to the consensus among all the methods. The feature importance is represented as the values of attributions x 10⁻⁴. The weights indicate the average of the weights of the neurons in the first layer for that specific feature.

![contribution_per_neuron_2](https://github.com/albertolzs/edc_mo_pdac/assets/140154262/52b6f40d-2db4-4238-92bf-c0fd4f5aea9f)
Stacked bars showing omics relative contribution to neurons in the embedding layer ordered by importance. The contribution was measured using neuron conductance. 8 out of the 50 neurons are activated by only a single modality; 6 by methylation and 2 by RNA-seq data.
