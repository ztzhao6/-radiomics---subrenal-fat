# Radiomics about Subrenal Fat
Characteristics Analysis Based on Perirenal Fat for the Classification of Essential Hypertension
* Medical research has found that perirenal fat is closely related to the occurrence and treatment of hypertension. This project uses machine learning to establish a new set of primary hypertension diagnosis technology based on the characteristics of subrenal fat imaging.
* Methods:
  * Segmented subrenal fat by processing image data and kidney label.
  * Extracted 1743 quantitative imaging features from the previously segmented subrenal fat region.
  * Data normalization, feature selection and hypertension classification model construction through Scikit-Learn.
* Achieved an area under the receiver operating characteristic curve (AUC) of 0.7 for hypertension classification. Proved correlation between subrenal fat and hypertension.
