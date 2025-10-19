# VariantProject
🧪 AI-Driven Molecular Explorer

This project is a hands-on molecular discovery notebook that combines artificial intelligence techniques with cheminformatics to explore chemical variants, compute descriptors, and visualize molecules in 3D. It is designed for researchers, students, and enthusiasts in chemistry and AI.

🚀 Features

Input SMILES

Enter one or more SMILES strings (comma-separated) as the starting molecules.

Supports simple molecules like CCO, CCC, CC(=O)O as seeds.

Similarity-Guided Variant Generation

Generates 20–40 variants per molecule.

Maintains scaffold similarity using Murcko scaffolds.

Filters variants based on Tanimoto similarity to the input molecule.

Molecular Descriptors & Scoring

Computes 6 standard descriptors:

Molecular Weight (MolWt)

LogP (Crippen)

H-Bond Donors & Acceptors

Rotatable Bonds

Topological Polar Surface Area (TPSA)

Scores molecules using dummy AI models for:

Predicted Solubility

Predicted Toxicity

Lab Feasibility

Computes composite score to rank top molecules.

3D Structure Visualization

Generates 3D structures for top 3 molecules.

Interactive visualization with py3Dmol.

Can be exported for downstream computational analysis.

Export Results

All evaluated molecules exported to top_molecules_export.csv.

Includes descriptors, predicted scores, and composite ranking.

💡 How It Works

SMILES Validation

Invalid molecules are automatically skipped.

Unsupported characters or malformed SMILES are flagged with guidance.

Variant Generation

Uses RDKit RWMol mutations to randomly change atoms while maintaining scaffold.

Applies Tanimoto similarity threshold to filter meaningful variants.

Descriptor Computation

RDKit computes standard descriptors for all molecules.

Data prepared for scoring and ranking.

AI-Based Evaluation

Dummy Random Forest models simulate solubility and toxicity predictions.

Feasibility is randomly assigned for demonstration purposes.

3D Embedding

Adds hydrogens and optimizes geometry using MMFF forcefield.

Interactive 3D models allow exploration of molecular shape and conformation.

🛠️ How to Use

Clone or open the notebook in Google Colab or Jupyter Notebook.

Run all cells sequentially.

Dependencies will automatically install.

Input molecules as comma-separated SMILES strings when prompted.

View top molecules with descriptors, scores, and 3D structures.

Download CSV for further analysis.

Example Input:

CCO, CCC, CC(=O)O

🔬 Screenshots / Example Outputs

(Include screenshots from your notebook here for a polished look.)

Variant Table: Shows SMILES, descriptors, and composite scores.

Top 3 3D Structures: Interactive visualizations.

⚡ Project Applications

Demonstrates integration of AI, cheminformatics, and molecular modeling.

Useful for:

Teaching molecular similarity and cheminformatics concepts.

Rapid prototyping of molecule variants.

AI-driven early-stage molecule exploration.

Can be extended to:

Real ML models trained on experimental data.

Drug discovery or material design pipelines.

Integration with databases like ChEMBL, PubChem.

📝 Notes

Current scoring models are dummy placeholders; replace with real models for production.

Designed to handle simple molecules; complex or exotic molecules may fail scaffold/similarity filtering.

3D visualization works best for small to medium molecules.
