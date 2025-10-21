# VariantProject v2.0 - Quick Start Guide

## 🚀 Installation

```bash
pip install rdkit py3Dmol pandas numpy scikit-learn selfies matplotlib seaborn tqdm
```

---

## ⚡ 30-Second Quick Start

```python
from VariantProject_Improved import quick_explore

# Single molecule
results = quick_explore(['CC(=O)Oc1ccccc1C(=O)O'])

# Multiple molecules
results = quick_explore([
    'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # Ibuprofen
])
```

**That's it!** Results automatically include:
- ✅ PAINS filtering
- ✅ Real ML predictions
- ✅ Pareto optimization
- ✅ Comprehensive descriptors
- ✅ Visualizations
- ✅ CSV export

---

## 🎛️ Advanced Usage

### Full Control

```python
from VariantProject_Improved import VariantProject

explorer = VariantProject()

results = explorer.explore(
    # Input
    smiles_list=['CC(=O)Oc1ccccc1C(=O)O'],
    
    # Variant generation
    n_variants=50,                  # Generate 50 variants per molecule
    similarity_threshold=0.6,       # Tanimoto similarity ≥ 0.6
    preserve_scaffold=True,         # Keep core structure
    use_bioisosteric=True,          # Use bioisosteric replacements
    
    # Quality filtering
    enable_pains_filter=True,       # Remove PAINS compounds
    enable_brenk_filter=True,       # Remove reactive groups
    
    # Optimization & analysis
    use_pareto=True,                # Pareto multi-objective ranking
    show_visualizations=True,       # Generate plots
    export_csv=True                 # Save results to CSV
)
```

---

## 📊 Understanding Your Results

### Result Table Columns

```python
results.head(10)
```

| Column | Description | Range |
|--------|-------------|-------|
| `SMILES` | Molecular structure | String |
| `MolWt` | Molecular weight | 0-1000 Da |
| `MolLogP` | Lipophilicity | -5 to +5 |
| `TPSA` | Polar surface area | 0-200 Ų |
| `QED` | Drug-likeness | 0-1 (higher better) |
| `RO5_Violations` | Lipinski violations | 0-4 (lower better) |
| `Pred_Solubility` | Predicted logS | Real (higher better) |
| `Pred_Solubility_Std` | Uncertainty | Real (lower better) |
| `Pred_Toxicity` | Predicted toxicity | Real (lower better) |
| `Lab_Feasibility` | Synthesis ease | 0-1 (higher better) |
| `Pareto_Rank` | Optimization rank | 1,2,3... (lower better) |
| `Composite_Score` | Overall score | 0-1 (higher better) |

---

## 🎯 Common Use Cases

### 1. Lead Optimization

```python
# Find similar molecules with better properties
results = explorer.explore(
    smiles_list=['CCO'],  # Ethanol
    n_variants=100,
    similarity_threshold=0.7,  # Very similar
    preserve_scaffold=True,
    use_bioisosteric=True
)

# Get top 5 with lowest toxicity
best = results.nsmallest(5, 'Pred_Toxicity')
print(best[['SMILES', 'Pred_Toxicity', 'Pred_Solubility']])
```

### 2. Diversity Exploration

```python
# Generate diverse variants
results = explorer.explore(
    smiles_list=['c1ccccc1'],  # Benzene
    n_variants=200,
    similarity_threshold=0.3,  # More diverse
    preserve_scaffold=False,
    use_bioisosteric=True
)
```

### 3. Scaffold Hopping

```python
# Keep scaffold, modify periphery
results = explorer.explore(
    smiles_list=['c1ccc2c(c1)ccc(n2)C(=O)O'],
    n_variants=50,
    preserve_scaffold=True,  # Keep core
    similarity_threshold=0.5
)
```

### 4. Export Top Candidates

```python
# Get Pareto-optimal molecules only
pareto_optimal = results[results['Pareto_Efficient'] == True]
pareto_optimal.to_csv('pareto_optimal_candidates.csv', index=False)

# Or top 20 by composite score
top_20 = results.head(20)
top_20.to_csv('top_20_candidates.csv', index=False)
```

---

## 📈 Interpreting Metrics

### Model Performance (Shown during training)

```
Solubility Model Performance:
  R² = 0.7854        # 78.5% variance explained (good!)
  RMSE = 0.6421      # Average error ±0.64 log units
  MAE = 0.4983       # Median error ~0.5 log units
  CV R² = 0.76±0.04  # Consistent across folds
```

**What this means:**
- R² > 0.7: Good predictive model
- RMSE < 1.0: Acceptable error for logS predictions
- Low CV std: Model is stable, not overfitting

### Prediction Uncertainty

```python
# High confidence prediction
Pred_Solubility: -2.34 ± 0.08  ✅ Trust this

# Low confidence prediction  
Pred_Solubility: -1.56 ± 0.45  ⚠️ Needs experimental validation
```

**Rule of thumb:**
- Std < 0.2: High confidence
- Std 0.2-0.5: Medium confidence
- Std > 0.5: Low confidence (verify experimentally)

### Pareto Ranking

```
Pareto_Rank = 1: Optimal (not dominated by any other molecule)
Pareto_Rank = 2: Second-best frontier
Pareto_Rank = 3+: Suboptimal (better alternatives exist)
```

**Strategy:**
- Focus on Rank 1 for lead candidates
- Rank 2-3 for backup options
- Rank 4+ usually discard

---

## 🎨 Visualizations Generated

### 1. Property Distributions
Histograms of MolWt, LogP, TPSA across all variants

### 2. Score Comparison
Bar chart comparing top 10 molecules across metrics

### 3. Pareto Frontier
Scatter plot showing optimal trade-offs

### 4. 2D Structures
Grid of top 5 molecular structures

### 5. 3D Visualizations
Interactive 3D conformers for top 3 molecules

---

## 🔧 Troubleshooting

### "No molecules passed filtering"

```python
# Relax filters
results = explorer.explore(
    smiles_list=your_smiles,
    enable_pains_filter=False,  # Disable PAINS
    enable_brenk_filter=False   # Disable reactive filter
)
```

### "Not enough variants generated"

```python
# Lower similarity threshold or increase attempts
results = explorer.explore(
    smiles_list=your_smiles,
    n_variants=50,
    similarity_threshold=0.3,  # More permissive
    preserve_scaffold=False    # Allow more changes
)
```

### "ESOL.csv not found"

```python
# File should be in same directory
# If missing, copy from repository or model will use synthetic data
import os
print(os.getcwd())  # Check current directory
```

---

## 📚 Example Workflows

### Complete Aspirin Optimization Pipeline

```python
from VariantProject_Improved import VariantProject

# Initialize
explorer = VariantProject()

# Generate and evaluate variants
results = explorer.explore(
    smiles_list=['CC(=O)Oc1ccccc1C(=O)O'],  # Aspirin
    n_variants=100,
    similarity_threshold=0.6,
    preserve_scaffold=True,
    use_bioisosteric=True,
    enable_pains_filter=True,
    enable_brenk_filter=True,
    use_pareto=True,
    show_visualizations=True,
    export_csv=True
)

# Filter by criteria
high_quality = results[
    (results['Pareto_Rank'] <= 2) &
    (results['RO5_Violations'] == 0) &
    (results['Pred_Solubility_Std'] < 0.2)
]

print(f"Found {len(high_quality)} high-quality candidates")

# Export top candidates
high_quality.to_csv('aspirin_optimization_candidates.csv', index=False)

# Get specific molecule
best = results.iloc[0]
print(f"\nBest candidate:")
print(f"  SMILES: {best['SMILES']}")
print(f"  Solubility: {best['Pred_Solubility']:.2f} ± {best['Pred_Solubility_Std']:.2f}")
print(f"  Toxicity: {best['Pred_Toxicity']:.2f}")
print(f"  QED: {best['QED']:.2f}")
```

---

## 🎓 Best Practices

1. **Start Small**: Test with 1-2 molecules and 30 variants
2. **Check Metrics**: Ensure R² > 0.7 for solubility model
3. **Filter Wisely**: Keep PAINS filtering enabled
4. **Trust Uncertainty**: Low std = reliable predictions
5. **Use Pareto**: Better than arbitrary weights
6. **Validate Top 5**: Always experimentally validate top candidates

---

## 📞 Need Help?

- **Documentation**: Read `IMPROVEMENTS_SUMMARY.md`
- **Changelog**: See `CHANGELOG.md` for version history
- **Code Comments**: Check `VariantProject_Improved.py` for inline docs

---

**Version:** 2.0
**Last Updated:** October 2025
