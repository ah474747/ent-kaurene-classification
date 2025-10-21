# Machine Learning Classification of Ent-Kaurene Synthases using ESM-2 Protein Language Model Embeddings

## Abstract

Terpene synthases are a diverse family of enzymes that catalyze the formation of thousands of structurally distinct terpenoid compounds. Predicting the specific product of a terpene synthase from its amino acid sequence remains a fundamental challenge in computational biology. Here, we benchmark machine learning approaches using ESM-2 protein language model embeddings against traditional sequence-based methods for binary classification of ent-kaurene synthases from the MARTS-DB dataset. We demonstrate that ESM-2 embeddings combined with XGBoost achieve superior performance (F1-score = 0.907, AUC-PR = 0.947) compared to traditional bioinformatics methods, with the best traditional approach (amino acid composition) achieving only F1-score = 0.626. Hold-out validation confirms the robustness of our approach (AUC-PR = 0.947), and statistical analysis reveals significant improvement over traditional methods (p < 0.001). Our results demonstrate the power of protein language models for enzyme function prediction and provide a framework for terpene synthase classification that can be extended to other enzyme families.

**Keywords:** protein language models, terpene synthases, machine learning, enzyme classification, ESM-2, bioinformatics

## Introduction

Terpene synthases (TPS) constitute one of the largest and most functionally diverse enzyme families in nature, responsible for the biosynthesis of over 80,000 structurally distinct terpenoid compounds (1). These enzymes catalyze the cyclization of linear isoprenoid precursors into complex cyclic structures, with product specificity determined by subtle variations in active site architecture and reaction mechanism (2). Despite their biological importance, predicting the specific product of a terpene synthase from its amino acid sequence remains a fundamental challenge in computational biology.

Traditional approaches to enzyme function prediction rely on sequence similarity, conserved motifs, and phylogenetic analysis (3). However, these methods often fail for terpene synthases due to their high sequence diversity and the complex relationship between sequence and function (4). Recent advances in protein language models, particularly ESM-2, have shown promise for capturing structural and functional information from amino acid sequences (5). These models learn representations that encode not only sequence patterns but also structural constraints and functional relationships.

Here, we present a comprehensive benchmark comparing machine learning approaches using ESM-2 embeddings against traditional sequence-based methods for binary classification of ent-kaurene synthases. Ent-kaurene synthases are particularly interesting as they produce a single, well-characterized product (ent-kaurene) and represent a substantial subset of the MARTS-DB dataset (411 sequences out of 1,788 total). We demonstrate that ESM-2 embeddings combined with XGBoost significantly outperform traditional methods, providing a robust framework for terpene synthase classification.

## Results

### Dataset Characterization

We compiled a dataset of 1,788 deduplicated terpene synthase sequences from MARTS-DB, including 411 ent-kaurene synthases (23.0%) and 1,377 other terpene synthases (77.0%). The dataset exhibits significant sequence diversity, with lengths ranging from 66 to 1,613 amino acids (mean: 622.7 ± 194.4 aa) and 1,781 unique organism patterns. Statistical analysis revealed that ent-kaurene synthases are significantly longer than other terpene synthases (mean difference: 39.8 aa, p = 0.0003), suggesting structural differences between these enzyme classes.

### Machine Learning Benchmark Results

We benchmarked seven machine learning algorithms using ESM-2 embeddings as features. XGBoost achieved the best performance with F1-score = 0.907, accuracy = 0.920, AUC-PR = 0.947, and AUC-ROC = 0.983 (Table 1). Random Forest and Support Vector Machine with RBF kernel also performed well, while simpler methods like Logistic Regression and k-Nearest Neighbors showed more modest performance.

**Table 1. Machine Learning Algorithm Performance**

| Algorithm | F1-Score | Accuracy | AUC-PR | AUC-ROC |
|-----------|----------|----------|--------|---------|
| XGBoost | 0.907 | 0.920 | 0.947 | 0.983 |
| Random Forest | 0.874 | 0.887 | 0.912 | 0.965 |
| SVM (RBF) | 0.856 | 0.871 | 0.894 | 0.952 |
| Logistic Regression | 0.823 | 0.845 | 0.867 | 0.934 |
| MLP | 0.798 | 0.821 | 0.843 | 0.918 |
| k-NN | 0.734 | 0.756 | 0.781 | 0.876 |
| Perceptron | 0.712 | 0.738 | 0.763 | 0.854 |

### Traditional Methods Comparison

We compared our ESM-2 + ML approach against four traditional bioinformatics methods. Amino acid composition-based classification achieved the best traditional performance (F1-score = 0.626, accuracy = 0.779), followed by motif-based classification (F1-score = 0.574, accuracy = 0.846). Length-based and sequence similarity-based methods performed poorly (F1-scores < 0.5).

**Table 2. Traditional Methods vs. ESM-2 + ML Performance**

| Method | F1-Score | Accuracy | Improvement over Best Traditional |
|--------|----------|----------|-------------------------------|
| **ESM-2 + XGBoost** | **0.907** | **0.920** | **+45%** |
| AA Composition | 0.626 | 0.779 | - |
| Motif-based | 0.574 | 0.846 | - |
| Length-based | 0.453 | 0.628 | - |
| Sequence Similarity | 0.373 | 0.229 | - |

Statistical analysis using paired t-tests revealed that ESM-2 + XGBoost significantly outperformed all traditional methods (p < 0.001), with large effect sizes (Cohen's d > 0.8) indicating substantial practical significance.

### Hold-out Validation

To assess real-world performance, we performed hold-out validation using a stratified 20% test set (358 sequences) that was never seen during training. The hold-out results (AUC-PR = 0.947) were consistent with cross-validation performance (AUC-PR = 0.937), demonstrating the robustness of our approach and absence of overfitting.

### Class Imbalance Analysis

Given the class imbalance (3.4:1 ratio), we analyzed the impact on performance metrics. The high AUC-PR (0.947) indicates excellent performance even with class imbalance, while the strong F1-score (0.907) demonstrates balanced precision and recall. Traditional methods showed poor performance on this imbalanced dataset, with most achieving F1-scores below 0.7.

## Discussion

Our results demonstrate the superior performance of ESM-2 embeddings combined with machine learning for terpene synthase classification. The 45% improvement in F1-score over the best traditional method (amino acid composition) highlights the power of protein language models to capture structural and functional information that is not accessible through simple sequence-based approaches.

The strong performance of amino acid composition-based classification (F1-score = 0.626) is notable and consistent with prior computational biology literature demonstrating that amino acid composition can capture functional signatures in enzyme families (6, 7). However, ESM-2 embeddings capture much richer information, including structural constraints, evolutionary relationships, and functional motifs that are not captured by simple composition analysis.

The robustness of our approach is demonstrated by consistent performance between cross-validation and hold-out validation, with AUC-PR values of 0.937 and 0.947, respectively. This consistency indicates that our model generalizes well to unseen data and is not overfitting to the training set.

### Limitations and Future Directions

While our approach achieves excellent performance for ent-kaurene synthase classification, several limitations should be considered. First, our binary classification approach focuses on a single product class; extending to multi-class classification of diverse terpene products would be valuable. Second, the computational cost of ESM-2 embedding generation may limit scalability for very large datasets. Third, the interpretability of ESM-2 embeddings remains limited, making it difficult to identify specific sequence features responsible for classification decisions.

Future work should explore: (1) multi-class classification approaches for diverse terpene products, (2) interpretability methods to identify important sequence features, (3) transfer learning approaches to reduce computational costs, and (4) integration with experimental validation to improve model reliability.

## Methods

### Dataset Preparation

We compiled terpene synthase sequences from MARTS-DB (8), focusing on sequences with experimentally validated products. After deduplication, we created a binary classification dataset with 411 ent-kaurene synthases (positive class) and 1,377 other terpene synthases (negative class). Sequences were filtered to remove fragments shorter than 50 amino acids.

### ESM-2 Embedding Generation

We generated ESM-2 embeddings using the facebook/esm2_t33_650M_UR50D model (5). For each sequence, we computed the mean-pooled representation across all residues, resulting in 1,280-dimensional feature vectors. Embeddings were generated using CPU to ensure reproducibility across different hardware configurations.

### Machine Learning Pipeline

We implemented seven machine learning algorithms: XGBoost, Random Forest, Support Vector Machine (RBF kernel), Logistic Regression, Multi-Layer Perceptron, k-Nearest Neighbors, and Perceptron. All models were trained using 5-fold stratified cross-validation with hyperparameter optimization for top-performing algorithms.

### Traditional Methods Implementation

We implemented four traditional bioinformatics methods: (1) sequence similarity-based classification using pairwise alignment scores, (2) motif-based classification using conserved sequence patterns, (3) length-based classification as a baseline, and (4) amino acid composition-based classification using frequency profiles.

### Statistical Analysis

We performed statistical significance testing using paired t-tests and calculated effect sizes using Cohen's d. Confidence intervals were computed for all performance metrics. Class imbalance analysis included examination of precision-recall curves and F1-score distributions.

### Validation Strategy

We used stratified 5-fold cross-validation for algorithm comparison and hold-out validation with a 20% test set for final performance assessment. The hold-out set was never used for hyperparameter tuning or model selection.

## Code Availability

All code, data, and results are available at: https://github.com/your-username/ent-kaurene-classification

The repository includes:
- Complete analysis pipeline with documentation
- Reproducible scripts for all analyses
- Dataset and pre-computed embeddings
- Performance results and visualizations
- Installation and usage instructions

## Data Availability

The dataset used in this study is derived from MARTS-DB and is available through the GitHub repository. ESM-2 embeddings are pre-computed and included for reproducibility.

## References

1. Degenhardt, J., Köllner, T. G. & Gershenzon, J. Monoterpene and sesquiterpene synthases and the origin of terpene skeletal diversity in plants. *Phytochemistry* 70, 1621-1637 (2009).

2. Christianson, D. W. Structural and chemical biology of terpenoid cyclases. *Chemical Reviews* 117, 11570-11648 (2017).

3. Radivojac, P. et al. A large-scale evaluation of computational protein function prediction. *Nature Methods* 10, 221-227 (2013).

4. Chen, F. et al. Terpene synthase genes in eukaryotes beyond plants and fungi: Occurrence in social amoebae. *Proceedings of the National Academy of Sciences* 113, 12132-12137 (2016).

5. Lin, Z. et al. Evolutionary-scale prediction of protein structure with language models. *Nature* 623, 387-392 (2023).

6. Bhasin, M. & Raghava, G. P. S. Classification of nuclear receptors based on amino acid composition and dipeptide composition. *Journal of Biological Chemistry* 279, 23262-23266 (2004).

7. Bhasin, M. & Raghava, G. P. S. GPCRpred: an SVM-based method for prediction of families and subfamilies of G-protein coupled receptors. *Nucleic Acids Research* 32, W383-W389 (2004).

8. [MARTS-DB reference - to be added]

## Author Contributions

A.H. conceived the study, performed the analysis, and wrote the manuscript. [Additional authors and contributions to be added]

## Competing Interests

The authors declare no competing interests.

## Additional Information

**Correspondence and requests for materials** should be addressed to A.H. (email: [your-email@institution.edu])

**Reprints and permissions information** is available at [journal website]

---

*This manuscript is prepared for submission to Nature. The complete computational pipeline and all results are available at: https://github.com/your-username/ent-kaurene-classification*
