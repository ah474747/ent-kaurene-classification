# Ent-Kaurene Synthase Classification using ESM-2 Embeddings

This repository contains the computational pipeline for predicting ent-kaurene synthase function from protein sequences using ESM-2 embeddings and machine learning algorithms.

## 📋 Overview

This study benchmarks machine learning approaches against traditional sequence-based methods for binary classification of ent-kaurene synthases from the MARTS-DB dataset. We demonstrate that ESM-2 embeddings combined with XGBoost significantly outperform traditional bioinformatics methods.

## 🎯 Key Results

- **ESM-2 + XGBoost**: F1-Score = 0.907, Accuracy = 0.920, AUC-PR = 0.947
- **Best traditional method**: Amino acid composition (F1-Score = 0.626)
- **Hold-out validation**: AUC-PR = 0.947 (consistent with cross-validation)
- **Improvement over best traditional**: 30% better F1-Score

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Running the Complete Pipeline

```bash
# 1. Generate ESM-2 embeddings (requires ~36 minutes on CPU)
python scripts/generate_embeddings.py

# 2. Run ML benchmark with 7 algorithms
python scripts/ent_kaurene_benchmark.py

# 3. Run hold-out validation
python scripts/holdout_validation.py

# 4. Run traditional methods comparison
python scripts/corrected_traditional_benchmark.py
```

## 📁 Repository Structure

```
Ent_Kaurene_Binary_Classifier/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/                               # Dataset and embeddings
│   ├── ent_kaurene_binary_dataset.csv  # Main dataset (1,788 sequences)
│   └── esm2_embeddings.npy            # ESM-2 embeddings (1,788 × 1,280)
├── scripts/                            # Analysis scripts
│   ├── generate_embeddings.py          # ESM-2 embedding generation
│   ├── ent_kaurene_benchmark.py        # ML algorithm benchmark
│   ├── holdout_validation.py           # Hold-out validation
│   ├── corrected_traditional_benchmark.py  # Traditional methods comparison
│   ├── statistical_analysis.py         # Statistical analysis
│   └── dataset_characterization.py     # Dataset analysis
└── results/                            # Output files
    ├── *.csv                          # Performance tables
    ├── *.json                         # Detailed results
    └── *.png                          # Visualization figures
```

## 📊 Dataset Information

- **Source**: MARTS-DB (deduplicated)
- **Total sequences**: 1,788
- **Ent-kaurene sequences**: 411 (23.0%)
- **Non-ent-kaurene sequences**: 1,377 (77.0%)
- **Sequence length**: 66-1,613 amino acids (mean: 622.7 ± 194.4)
- **Organism diversity**: 1,781 unique organism patterns

## 🔬 Methodology

### 1. Data Preparation
- Binary classification: ent-kaurene synthase (positive) vs. all other terpene synthases (negative)
- Deduplication while preserving product information
- Stratified hold-out split (80% train, 20% test)

### 2. Feature Generation
- ESM-2 embeddings (facebook/esm2_t33_650M_UR50D)
- 1,280-dimensional vectors per sequence
- Generated using CPU (compatible with most systems)

### 3. Machine Learning Benchmark
Seven algorithms tested:
- **XGBoost Classifier** (best performer)
- Random Forest Classifier
- Support Vector Machine (RBF kernel)
- Logistic Regression
- Multi-Layer Perceptron
- k-Nearest Neighbors
- Perceptron

### 4. Traditional Methods Comparison
- Sequence similarity-based classification
- Motif-based classification
- Length-based classification (baseline)
- Amino acid composition-based classification

### 5. Validation Strategy
- 5-fold stratified cross-validation
- Hold-out validation (20% unseen test set)
- Statistical significance testing
- Class imbalance analysis

## 📈 Results Summary

| Method | F1-Score | Accuracy | AUC-PR | AUC-ROC |
|--------|----------|----------|--------|---------|
| **ESM-2 + XGBoost** | **0.907** | **0.920** | **0.947** | **0.983** |
| AA Composition | 0.626 | 0.779 | N/A | N/A |
| Motif-based | 0.574 | 0.846 | N/A | N/A |
| Length-based | 0.453 | 0.628 | N/A | N/A |
| Sequence Similarity | 0.373 | 0.229 | N/A | N/A |

## 🔧 Computational Requirements

### Minimum System Requirements
- **RAM**: 8 GB (32 GB recommended)
- **Storage**: 2 GB free space
- **CPU**: Multi-core processor (embedding generation is CPU-intensive)

### Runtime Estimates
- ESM-2 embedding generation: ~36 minutes (1,788 sequences)
- ML benchmark: ~10 minutes
- Hold-out validation: ~2 minutes
- Traditional methods: ~5 minutes

## 📚 Citation

If you use this code or dataset, please cite:

```bibtex
@article{horwitz2024entkaurene,
  title={Machine Learning Classification of Ent-Kaurene Synthases using ESM-2 Protein Language Model Embeddings},
  author={Horwitz, Andrew and [Co-authors]},
  journal={Nature},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [ESM-2 Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)
- [MARTS-DB Database](https://www.marts-db.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your-email@institution.edu]

## 🏷️ Version

Current version: 1.0.0
Last updated: October 2024