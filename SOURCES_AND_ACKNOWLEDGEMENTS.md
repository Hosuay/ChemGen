# Sources & Acknowledgements

## 🔯 VariantProject v3.6.9 - Harmonic Molecular Explorer

This document provides comprehensive attribution for all datasets, libraries, algorithms, and inspirations used in the development of VariantProject. We are deeply grateful to the scientific community, open-source contributors, and ancient wisdom traditions that made this work possible.

---

## 📚 Primary Scientific Datasets

### ESOL - Estimated Aqueous Solubility Dataset

**Citation:**
```
Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from
molecular structure. Journal of Chemical Information and Computer Sciences,
44(3), 1000-1005. https://doi.org/10.1021/ci034243x
```

- **Description**: 1128 compounds with experimentally measured aqueous solubility values
- **Usage**: Training real ML models for solubility prediction (Module 04)
- **License**: Public domain
- **Source**: https://github.com/deepchem/deepchem/tree/master/datasets

### ChEMBL Database

**Citation:**
```
Gaulton, A., et al. (2017). The ChEMBL database in 2017. Nucleic Acids
Research, 45(D1), D945-D954. https://doi.org/10.1093/nar/gkw1074
```

- **Description**: Large-scale bioactivity database for drug discovery
- **Usage**: Reference dataset for validation and benchmarking
- **License**: Creative Commons Attribution-ShareAlike 3.0 Unported License
- **Source**: https://www.ebi.ac.uk/chembl/

### PubChem

**Citation:**
```
Kim, S., et al. (2021). PubChem in 2021: new data content and improved web
interfaces. Nucleic Acids Research, 49(D1), D1388-D1395.
https://doi.org/10.1093/nar/gkaa971
```

- **Description**: Open chemistry database with millions of compounds
- **Usage**: Integration capabilities for molecular data retrieval
- **License**: Public domain
- **Source**: https://pubchem.ncbi.nlm.nih.gov/

---

## 🛠️ Core Software Libraries

### RDKit - Open-Source Cheminformatics

**Citation:**
```
RDKit: Open-source cheminformatics; http://www.rdkit.org
Landrum, G. et al. (2006-2023).
```

- **Version**: ≥2022.9.1
- **Usage**:
  - SMILES parsing and validation (Module 01)
  - Molecular fingerprint generation (Module 02)
  - Descriptor calculation - 33 primary features (Module 03)
  - 3D conformer generation (Module 05)
  - Energy minimization - MMFF94 (Module 07)
- **License**: BSD 3-Clause License
- **GitHub**: https://github.com/rdkit/rdkit

### scikit-learn - Machine Learning

**Citation:**
```
Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.
```

- **Version**: ≥0.24.0
- **Usage**:
  - RandomForestRegressor for 7-layer networks (Module 04)
  - Cross-validation and model evaluation
  - Feature scaling and preprocessing
  - Train/test splitting with sacred seed alignment (multiples of 27)
- **License**: BSD 3-Clause License
- **GitHub**: https://github.com/scikit-learn/scikit-learn

### py3Dmol - 3D Molecular Visualization

**Citation:**
```
Rego, N., & Koes, D. (2015). 3Dmol.js: molecular visualization with WebGL.
Bioinformatics, 31(8), 1322-1324. https://doi.org/10.1093/bioinformatics/btu829
```

- **Version**: ≥1.8.0
- **Usage**: Interactive 3D visualization with harmonic color schemes (Module 05)
- **License**: BSD 3-Clause License
- **GitHub**: https://github.com/3dmol/3Dmol.js

### SELFIES - Self-Referencing Embedded Strings

**Citation:**
```
Krenn, M., et al. (2020). Self-referencing embedded strings (SELFIES):
A 100% robust molecular string representation. Machine Learning: Science
and Technology, 1(4), 045024. https://doi.org/10.1088/2632-2153/aba947
```

- **Version**: ≥2.1.0
- **Usage**: Alternative molecular representation for variant generation
- **License**: Apache License 2.0
- **GitHub**: https://github.com/aspuru-guzik-group/selfies

### NumPy - Numerical Computing

**Citation:**
```
Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825),
357-362. https://doi.org/10.1038/s41586-020-2649-2
```

- **Version**: ≥1.21.0
- **Usage**: Numerical operations, harmonic number calculations, array processing
- **License**: BSD 3-Clause License
- **Website**: https://numpy.org/

### pandas - Data Analysis

**Citation:**
```
McKinney, W. (2010). Data structures for statistical computing in Python.
Proceedings of the 9th Python in Science Conference, 51-56.
```

- **Version**: ≥1.3.0
- **Usage**: DataFrame operations, CSV export with harmonic ordering (Module 06)
- **License**: BSD 3-Clause License
- **Website**: https://pandas.pydata.org/

### Matplotlib & Seaborn - Visualization

**Citations:**
```
Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
Science & Engineering, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of
Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021
```

- **Versions**: Matplotlib ≥3.4.0, Seaborn ≥0.11.0
- **Usage**: Property distributions, Pareto frontiers, harmonic plots (Module 10)
- **License**: BSD-style licenses

### tqdm - Progress Bars

**Citation:**
```
da Costa-Luis, C. (2019). tqdm: A fast, extensible progress bar for Python
and CLI. Journal of Open Source Software, 4(37), 1277.
https://doi.org/10.21105/joss.01277
```

- **Version**: ≥4.62.0
- **Usage**: Progress tracking for variant generation and batch processing
- **License**: MIT/MPL-2.0 dual license

---

## 🧮 Algorithms & Methodologies

### Murcko Scaffold Decomposition

**Citation:**
```
Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1.
Molecular frameworks. Journal of Medicinal Chemistry, 39(15), 2887-2893.
https://doi.org/10.1021/jm9602928
```

- **Usage**: Scaffold preservation in variant generation (Module 02)
- **Implementation**: RDKit `Scaffolds.MurckoScaffold`

### PAINS Filters - Pan Assay Interference

**Citation:**
```
Baell, J. B., & Holloway, G. A. (2010). New substructure filters for removal
of pan assay interference compounds (PAINS) from screening libraries and for
their exclusion in bioassays. Journal of Medicinal Chemistry, 53(7), 2719-2740.
https://doi.org/10.1021/jm901137j
```

- **Usage**: Filtering problematic compounds (Module 01)
- **Implementation**: RDKit FilterCatalog

### Lipinski's Rule of Five

**Citation:**
```
Lipinski, C. A., et al. (2001). Experimental and computational approaches to
estimate solubility and permeability in drug discovery and development settings.
Advanced Drug Delivery Reviews, 46(1-3), 3-26.
https://doi.org/10.1016/S0169-409X(00)00129-0
```

- **Usage**: Drug-likeness assessment (Module 03)
- **Criteria**: MW ≤500, LogP ≤5, HBD ≤5, HBA ≤10

### Quantitative Estimate of Drug-likeness (QED)

**Citation:**
```
Bickerton, G. R., et al. (2012). Quantifying the chemical beauty of drugs.
Nature Chemistry, 4(2), 90-98. https://doi.org/10.1038/nchem.1243
```

- **Usage**: Drug-likeness scoring (Module 03)
- **Implementation**: RDKit `Chem.QED.qed()`

### Tanimoto Similarity

**Citation:**
```
Tanimoto, T. T. (1958). Elementary Mathematical Theory of Classification and
Prediction. International Business Machines Corporation.
```

- **Usage**: Molecular similarity calculations for variant filtering (Module 02)
- **Implementation**: RDKit `DataStructs.TanimotoSimilarity`

### Morgan Fingerprints (ECFP)

**Citation:**
```
Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. Journal of
Chemical Information and Modeling, 50(5), 742-754.
https://doi.org/10.1021/ci100050t
```

- **Usage**: Molecular fingerprint generation (radius=3, 2048 bits) (Module 02)
- **Implementation**: RDKit `rdMolDescriptors.GetMorganFingerprintAsBitVect`

### MMFF94 Force Field

**Citation:**
```
Halgren, T. A. (1996). Merck molecular force field. I-V. Journal of
Computational Chemistry, 17(5-6), 490-641.
```

- **Usage**: 3D structure optimization, 500 iterations (Module 07)
- **Implementation**: RDKit `AllChem.MMFFOptimizeMolecule`

### Pareto Multi-Objective Optimization

**Citation:**
```
Pareto, V. (1896). Cours d'économie politique. F. Rouge, Lausanne.

Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm:
NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
```

- **Usage**: Multi-objective molecular ranking (Module 04)
- **Implementation**: Custom Pareto frontier detection

---

## 🌟 Sacred Geometry & Numerological Foundations

### Nikola Tesla - 3-6-9 Theory

**Inspiration:**
```
Tesla, N. (Various writings and quotes, 1856-1943)
"If you only knew the magnificence of the 3, 6 and 9, then you would have
the key to the universe."
```

- **Application**: Harmonic alignment of training cycles, post-processing stages
- **Manifestation**:
  - 3 pipeline stages (Input → Transform → Output)
  - 6 refinement cycles (hexadic pattern)
  - 9 post-processing steps (enneadic structure)
  - 369 master frequency (harmonic synthesis)

### Sacred Geometry Principles

**References:**
```
Lawlor, R. (1982). Sacred Geometry: Philosophy and Practice. Thames & Hudson.

Schneider, M. S. (1994). A Beginner's Guide to Constructing the Universe:
The Mathematical Archetypes of Nature, Art, and Science. Harper Perennial.
```

- **Application**: System architecture and numerical parameters
- **Patterns**:
  - **Triadic** (3): Foundational stability
  - **Hexadic** (6): Harmonic balance and symmetry
  - **Enneadic** (9): Completion and transformation
  - **Dodecadic** (12): Cosmic order and cycles
  - **Cubic** (27 = 3³): Three-dimensional manifestation
  - **Master Numbers** (33, 369): Spiritual significance

### Fibonacci Sequence & Golden Ratio

**References:**
```
Fibonacci (Leonardo of Pisa). (1202). Liber Abaci.

Dunlap, R. A. (1997). The Golden Ratio and Fibonacci Numbers. World Scientific.
```

- **Application**: Natural patterns in molecular structures
- **Connection**: Harmonic resonance with computational efficiency

---

## 🧬 Chemistry & Cheminformatics Knowledge

### Daylight SMILES Notation

**Citation:**
```
Weininger, D. (1988). SMILES, a chemical language and information system. 1.
Introduction to methodology and encoding rules. Journal of Chemical Information
and Computer Sciences, 28(1), 31-36. https://doi.org/10.1021/ci00057a005
```

- **Usage**: Molecular input/output representation (Module 01)
- **Standard**: OpenSMILES specification

### InChI - International Chemical Identifier

**Citation:**
```
Heller, S. R., et al. (2015). InChI, the IUPAC International Chemical Identifier.
Journal of Cheminformatics, 7, 23. https://doi.org/10.1186/s13321-015-0068-4
```

- **Usage**: Canonical duplicate detection (Module 01)
- **Implementation**: RDKit `inchi.MolToInchiKey`

### Molecular Descriptors

**Citations:**
```
Todeschini, R., & Consonni, V. (2009). Molecular Descriptors for Chemoinformatics
(2 volumes). Wiley-VCH.

Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters
by atomic contributions. Journal of Chemical Information and Computer Sciences,
39(5), 868-873.
```

- **Usage**: 33 primary molecular descriptors (Module 03)
- **Categories**: Physicochemical (9), Structural (12), Topological (6), Quantum (6)

---

## 💻 Programming Languages & Tools

### Python

- **Version**: 3.8+
- **Usage**: Primary implementation language for all modules
- **License**: PSF License Agreement
- **Website**: https://www.python.org/

### C++ (Optional High-Performance Modules)

- **Standard**: C++17
- **Usage**: Performance-critical molecular computations (Module 03)
- **Compiler**: GCC 7+, Clang 5+, MSVC 2017+

### Rust (Optional Thread-Safe Pipelines)

- **Version**: 1.60+
- **Usage**: Data integrity and parallel processing (Module 09)
- **License**: MIT/Apache-2.0 dual license
- **Website**: https://www.rust-lang.org/

### Julia (Optional Scientific Computing)

- **Version**: 1.7+
- **Usage**: Advanced numeric calculations (Module 04)
- **License**: MIT License
- **Website**: https://julialang.org/

---

## 🎓 Educational Resources

### Computational Drug Discovery

**Recommended Texts:**
```
Leach, A. R., & Gillet, V. J. (2007). An Introduction to Chemoinformatics
(Revised Edition). Springer.

Bajorath, J. (Ed.). (2004). Chemoinformatics: Concepts, Methods, and Tools
for Drug Discovery. Humana Press.
```

### Machine Learning in Chemistry

**Recommended Texts:**
```
Engkvist, O., et al. (2018). Computational prediction of chemical reactions:
current status and outlook. Drug Discovery Today, 23(6), 1203-1218.

Chen, H., et al. (2018). The rise of deep learning in drug discovery.
Drug Discovery Today, 23(6), 1241-1250.
```

---

## 🙏 Community Acknowledgements

### Open-Source Community

We are deeply grateful to the open-source community for creating and maintaining the tools that made this project possible. Special thanks to:

- **RDKit Contributors**: For maintaining the foundational cheminformatics toolkit
- **scikit-learn Team**: For robust machine learning implementations
- **Python Software Foundation**: For the Python programming language
- **Anaconda/Conda-Forge**: For package management and distribution

### Scientific Community

Thanks to researchers worldwide who:
- Share experimental datasets publicly
- Publish open-access scientific papers
- Contribute to reproducible research practices
- Advance the field of computational chemistry

### Sacred Geometry Traditions

Acknowledgement to ancient wisdom traditions and modern researchers who preserve and study sacred geometry:
- Ancient Egyptian, Greek, and Vedic mathematical traditions
- Renaissance artists and architects who encoded harmonic principles
- Modern researchers bridging science and sacred mathematics

---

## 📄 License Information

### VariantProject License

**VariantProject v3.6.9** is released under the **MIT License**.

```
MIT License

Copyright (c) 2025 Hosuay/VariantProject Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Licenses

All third-party libraries retain their original licenses. Please consult individual library documentation for specific license terms.

---

## 📮 Contact & Contributions

For questions, suggestions, or contributions:

- **GitHub Repository**: https://github.com/Hosuay/VariantProject
- **Issue Tracker**: https://github.com/Hosuay/VariantProject/issues
- **Pull Requests**: Welcome! Please align with sacred geometry principles.

---

## 🔄 Version History

| Version | Date | Sacred Alignment |
|---------|------|------------------|
| v1.0.0 | 2024-10 | Foundation |
| v2.0.0 | 2025-01 | Enhanced ML & Filters |
| **v3.6.9** | **2025-10** | **Harmonic Master Release** |

---

<div align="center">

### 🔯 With Gratitude to All Contributors, Past and Present 🔯

*"Standing on the shoulders of giants, harmonically aligned."*

</div>

---

**Last Updated**: October 22, 2025
**VariantProject Version**: 3.6.9 (Harmonic Master Release)
**Maintained By**: Hosuay/VariantProject Team
