# 🧬 VariantProject v2.0 - Computational Molecular Exploration Tool

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.09+-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

VariantProject v2.0 is an advanced AI-assisted molecular exploration tool for drug discovery and computational chemistry. It generates molecular variants, analyzes drug-likeness, and provides comprehensive molecular property predictions.

### ✨ Key Features

- 🔬 **Molecular Variant Generation** - Create diverse molecular variants using SMILES manipulation
- 📊 **Comprehensive Property Analysis** - Calculate 30+ molecular descriptors
- 🎯 **Drug-likeness Assessment** - Lipinski's Rule of 5, QED scores, PAINS filtering
- 🧪 **ML-Based Predictions** - Solubility and toxicity predictions using Random Forest models
- 📈 **Multi-objective Optimization** - Pareto frontier analysis for optimal molecules
- 🖼️ **Visualization** - 2D/3D molecular structures and property distributions
- 🔍 **Duplicate Detection** - InChI-based deduplication
- ⚡ **Bioisosteric Replacements** - Smart molecular modifications

## 📦 Installation

### Option 1: Google Colab (Easiest)
```python
!pip install -q condacolab
import condacolab
condacolab.install()
!conda install -c conda-forge rdkit -y
!pip install py3Dmol selfies tqdm pandas numpy scikit-learn matplotlib seaborn
```

### Option 2: Conda (Recommended for Local)
```bash
# Create new environment
conda create -n variantproject python=3.8
conda activate variantproject

# Install RDKit from conda-forge
conda install -c conda-forge rdkit

# Install other dependencies
pip install py3Dmol selfies tqdm pandas numpy scikit-learn matplotlib seaborn
```

### Option 3: Pip Only
```bash
pip install rdkit-pypi  # May not work on all systems
pip install py3Dmol selfies tqdm pandas numpy scikit-learn matplotlib seaborn
```

### Option 4: No RDKit Installation (Simplified Demo)
If you cannot install RDKit, use the simplified demo:
```bash
python molecular_explorer_demo.py
```

## 🎯 Quick Start

### Basic Usage
```python
from VariantProject import VariantProject, quick_explore

# Quick exploration with example molecules
molecules = [
    'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
]

explorer, results = quick_explore(molecules, n_variants=30)
```

### Interactive Mode
```bash
python VariantProject_v2.py
```

### Jupyter Notebook
Open `VariantProject_Complete_Notebook.ipynb` in Jupyter or Google Colab for step-by-step exploration.

## 🧪 Example Molecules

```python
examples = {
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'Acetaminophen': 'CC(=O)NC1=CC=C(C=C1)O',
    'Penicillin': 'CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O',
    'Metformin': 'CN(C)C(=N)NC(=N)N',
}
```

## 📊 Features in Detail

### 1. Molecular Property Calculation
- Molecular Weight, LogP, TPSA
- H-bond donors/acceptors
- Rotatable bonds, aromatic rings
- Lipinski's Rule of 5 compliance
- QED (Quantitative Estimate of Drug-likeness)

### 2. Variant Generation Methods
- Atom substitutions (bioisosteric replacements)
- Functional group modifications
- Scaffold preservation options
- SELFIES-based mutations

### 3. Filtering Systems
- PAINS (Pan-Assay Interference Compounds)
- Brenk structural alerts
- Reactive group detection
- InChI-based duplicate removal

### 4. Machine Learning Models
- Random Forest for property prediction
- Cross-validation and uncertainty quantification
- Model performance metrics (R², MAE, RMSE)

## 🔧 Troubleshooting

### RDKit Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'rdkit'`

**Solutions**:
1. Use Anaconda with conda-forge channel (most reliable)
2. For Google Colab, use condacolab installation method
3. Try rdkit-pypi package: `pip install rdkit-pypi`
4. Use the simplified demo if RDKit won't install

### Other Common Issues

| Issue | Solution |
|-------|----------|
| py3Dmol not working | Optional for 3D viz, code works without it |
| SELFIES import error | Optional, uses SMILES as fallback |
| Memory errors with large datasets | Reduce n_variants or batch size |
| Visualization not showing | Check matplotlib backend settings |

## 📁 Project Structure

```
VariantProject/
├── VariantProject_v2.py              # Main script
├── molecular_explorer_demo.py        # Simplified demo (no RDKit)
├── VariantProject_Complete_Notebook.ipynb  # Jupyter notebook
├── requirements.txt                  # Dependencies
├── README.md                         # Documentation
├── examples/                         # Example scripts
│   ├── quick_start.py
│   └── batch_analysis.py
└── data/                            # Sample data
    └── example_molecules.csv
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Variant generation speed | ~100 molecules/second |
| Property calculation | ~1000 molecules/second |
| Typical exploration time | 30-60 seconds for 1000 variants |
| Memory usage | ~500MB for 10,000 molecules |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use VariantProject in your research, please cite:

```bibtex
@software{variantproject2024,
  title = {VariantProject: AI-Assisted Molecular Exploration Tool},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/Hosuay/VariantProject}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- RDKit community for the excellent cheminformatics toolkit
- Anthropic for AI assistance in development
- Contributors and users of the project

## 📮 Contact

- GitHub: [@Hosuay](https://github.com/Hosuay)
- Issues: [GitHub Issues](https://github.com/Hosuay/VariantProject/issues)

---
**Version**: 2.0 | **Last Updated**: October 2025
