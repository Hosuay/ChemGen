#!/usr/bin/env python3
"""
Simplified Molecular Explorer Demo
Works without RDKit - demonstrates the concept using basic chemistry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from typing import List, Dict, Tuple

print("🧬 MOLECULAR EXPLORER - SIMPLIFIED DEMO")
print("=" * 60)
print("This version works without RDKit installation!")
print("=" * 60)

class SimpleMolecule:
    """Simple molecule representation without RDKit"""
    
    def __init__(self, smiles: str, name: str = "Unknown"):
        self.smiles = smiles
        self.name = name
        self.atoms = self._parse_atoms()
        self.properties = self._calculate_properties()
    
    def _parse_atoms(self) -> Dict[str, int]:
        """Parse atoms from SMILES (simplified)"""
        atoms = {
            'C': len(re.findall(r'[Cc]', self.smiles)),
            'O': len(re.findall(r'[Oo]', self.smiles)),
            'N': len(re.findall(r'[Nn]', self.smiles)),
            'S': self.smiles.count('S'),
            'F': self.smiles.count('F'),
            'Cl': self.smiles.count('Cl'),
            'Br': self.smiles.count('Br'),
            'P': self.smiles.count('P'),
        }
        return atoms
    
    def _calculate_properties(self) -> Dict:
        """Calculate molecular properties (simplified estimates)"""
        # Atomic weights
        weights = {'C': 12.01, 'O': 16.00, 'N': 14.01, 'S': 32.07, 
                  'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'P': 30.97, 'H': 1.008}
        
        # Estimate molecular weight
        mw = sum(self.atoms[atom] * weights.get(atom, 0) for atom in self.atoms)
        
        # Estimate hydrogen count (very simplified)
        h_count = max(0, 2 * self.atoms['C'] + 2 - self.atoms.get('=', 0) * 2)
        mw += h_count * weights['H']
        
        # Estimate LogP (simplified Wildman-Crippen method approximation)
        logp = 0.0
        logp += self.atoms['C'] * 0.15  # Carbons contribute to lipophilicity
        logp -= self.atoms['O'] * 0.5   # Oxygens decrease lipophilicity
        logp -= self.atoms['N'] * 0.3   # Nitrogens decrease lipophilicity
        logp += self.atoms['F'] * 0.1   # Fluorines slightly increase
        logp += self.atoms['Cl'] * 0.3  # Chlorines increase
        logp += self.atoms['Br'] * 0.5  # Bromines increase more
        
        # Estimate TPSA (simplified)
        tpsa = self.atoms['O'] * 20.0 + self.atoms['N'] * 15.0
        
        # H-bond donors and acceptors (simplified)
        hbd = self.atoms['O'] + self.atoms['N']  # Simplified
        hba = self.atoms['O'] * 2 + self.atoms['N']  # Simplified
        
        # Aromatic rings (count benzene patterns)
        aromatic_rings = self.smiles.count('c1ccccc1') + self.smiles.count('c1cccc')
        
        return {
            'MW': round(mw, 2),
            'LogP': round(logp, 2),
            'TPSA': round(tpsa, 2),
            'HBD': hbd,
            'HBA': hba,
            'AromaticRings': aromatic_rings,
            'RotatableBonds': self.smiles.count('CC') + self.smiles.count('CO'),  # Simplified
            'Atoms': sum(self.atoms.values())
        }
    
    def get_lipinski_violations(self) -> int:
        """Calculate Lipinski's Rule of 5 violations"""
        violations = 0
        if self.properties['MW'] > 500:
            violations += 1
        if self.properties['LogP'] > 5:
            violations += 1
        if self.properties['HBD'] > 5:
            violations += 1
        if self.properties['HBA'] > 10:
            violations += 1
        return violations
    
    def get_qed_score(self) -> float:
        """Simplified QED (drug-likeness) score"""
        score = 1.0
        
        # Penalize for Lipinski violations
        score -= self.get_lipinski_violations() * 0.2
        
        # Ideal ranges (simplified)
        if 200 < self.properties['MW'] < 500:
            score += 0.1
        if 0 < self.properties['LogP'] < 3:
            score += 0.1
        if 40 < self.properties['TPSA'] < 140:
            score += 0.1
        
        return max(0, min(1, score))

class MolecularExplorer:
    """Main molecular exploration class"""
    
    def __init__(self):
        self.molecules = []
        self.results = None
    
    def add_molecule(self, smiles: str, name: str = None) -> SimpleMolecule:
        """Add a molecule for analysis"""
        mol = SimpleMolecule(smiles, name or f"Mol_{len(self.molecules)+1}")
        self.molecules.append(mol)
        return mol
    
    def generate_variants(self, smiles: str, n_variants: int = 5) -> List[str]:
        """Generate simple molecular variants"""
        variants = []
        
        # Common substitutions
        substitutions = [
            ('F', 'Cl'), ('Cl', 'Br'), ('Br', 'I'),
            ('O', 'S'), ('N', 'O'),
            ('c', 'n'),  # Aromatic carbon to nitrogen
            ('CC', 'C(C)C'),  # Add branching
            ('C(=O)', 'C(=S)'),  # Carbonyl to thiocarbonyl
        ]
        
        for old, new in substitutions[:n_variants]:
            if old in smiles:
                variant = smiles.replace(old, new, 1)
                if variant != smiles:
                    variants.append(variant)
        
        # Add methyl groups
        if len(variants) < n_variants and 'c' in smiles:
            methylated = smiles.replace('c', 'c(C)', 1)
            variants.append(methylated)
        
        # Add functional groups
        functional_groups = ['F', 'Cl', 'OH', 'NH2', 'C(=O)O']
        for fg in functional_groups:
            if len(variants) >= n_variants:
                break
            if 'c1' in smiles:
                variant = smiles.replace('c1', f'c({fg})1', 1)
                if variant not in variants:
                    variants.append(variant)
        
        return variants[:n_variants]
    
    def analyze_molecules(self) -> pd.DataFrame:
        """Analyze all molecules and create results DataFrame"""
        if not self.molecules:
            print("No molecules to analyze!")
            return None
        
        data = []
        for mol in self.molecules:
            row = {
                'Name': mol.name,
                'SMILES': mol.smiles,
                **mol.properties,
                'Lipinski_Violations': mol.get_lipinski_violations(),
                'QED_Score': mol.get_qed_score()
            }
            data.append(row)
        
        self.results = pd.DataFrame(data)
        self.results = self.results.sort_values('QED_Score', ascending=False)
        return self.results
    
    def visualize_results(self):
        """Create visualizations of molecular properties"""
        if self.results is None:
            print("No results to visualize. Run analyze_molecules() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. MW vs LogP scatter
        scatter = axes[0, 0].scatter(
            self.results['MW'], 
            self.results['LogP'],
            c=self.results['QED_Score'],
            s=100,
            cmap='viridis',
            alpha=0.7,
            edgecolors='black'
        )
        axes[0, 0].set_xlabel('Molecular Weight (Da)')
        axes[0, 0].set_ylabel('LogP')
        axes[0, 0].set_title('Molecular Property Space')
        axes[0, 0].axvline(500, color='r', linestyle='--', alpha=0.5, label='MW limit')
        axes[0, 0].axhline(5, color='r', linestyle='--', alpha=0.5, label='LogP limit')
        axes[0, 0].legend()
        plt.colorbar(scatter, ax=axes[0, 0], label='QED Score')
        
        # Annotate points
        for idx, row in self.results.iterrows():
            axes[0, 0].annotate(
                row['Name'][:10], 
                (row['MW'], row['LogP']),
                fontsize=8,
                alpha=0.7
            )
        
        # 2. QED Score bar chart
        axes[0, 1].bar(
            range(len(self.results)),
            self.results['QED_Score'],
            color='skyblue',
            edgecolor='black'
        )
        axes[0, 1].set_xlabel('Molecule')
        axes[0, 1].set_ylabel('QED Score')
        axes[0, 1].set_title('Drug-likeness Scores')
        axes[0, 1].set_xticks(range(len(self.results)))
        axes[0, 1].set_xticklabels(self.results['Name'], rotation=45, ha='right')
        axes[0, 1].axhline(0.5, color='orange', linestyle='--', alpha=0.5)
        
        # 3. Lipinski violations
        violations_count = self.results['Lipinski_Violations'].value_counts().sort_index()
        axes[1, 0].bar(
            violations_count.index,
            violations_count.values,
            color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(violations_count)],
            edgecolor='black'
        )
        axes[1, 0].set_xlabel('Number of Violations')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title("Lipinski's Rule of 5 Violations")
        axes[1, 0].set_xticks(violations_count.index)
        
        # 4. Property distributions
        properties = ['MW', 'LogP', 'TPSA']
        for i, prop in enumerate(properties):
            axes[1, 1].hist(
                self.results[prop],
                bins=20,
                alpha=0.5,
                label=prop,
                edgecolor='black'
            )
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Property Distributions')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a text report of the analysis"""
        if self.results is None:
            return "No results available. Run analyze_molecules() first."
        
        report = []
        report.append("="*60)
        report.append("MOLECULAR ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total molecules analyzed: {len(self.results)}")
        
        report.append("\n" + "="*60)
        report.append("SUMMARY STATISTICS")
        report.append("="*60)
        
        stats = self.results.describe()
        report.append(f"\nMolecular Weight:")
        report.append(f"  Mean: {stats.loc['mean', 'MW']:.2f} Da")
        report.append(f"  Range: {stats.loc['min', 'MW']:.2f} - {stats.loc['max', 'MW']:.2f} Da")
        
        report.append(f"\nLogP:")
        report.append(f"  Mean: {stats.loc['mean', 'LogP']:.2f}")
        report.append(f"  Range: {stats.loc['min', 'LogP']:.2f} - {stats.loc['max', 'LogP']:.2f}")
        
        report.append(f"\nDrug-likeness (QED):")
        report.append(f"  Mean: {stats.loc['mean', 'QED_Score']:.3f}")
        report.append(f"  Best: {stats.loc['max', 'QED_Score']:.3f}")
        
        lipinski_pass = (self.results['Lipinski_Violations'] == 0).sum()
        report.append(f"\nLipinski's Rule of 5:")
        report.append(f"  Passing: {lipinski_pass}/{len(self.results)} molecules")
        
        report.append("\n" + "="*60)
        report.append("TOP 5 MOLECULES BY QED SCORE")
        report.append("="*60)
        
        top5 = self.results.head(5)
        for idx, row in top5.iterrows():
            report.append(f"\n{row['Name']}:")
            report.append(f"  SMILES: {row['SMILES']}")
            report.append(f"  QED Score: {row['QED_Score']:.3f}")
            report.append(f"  MW: {row['MW']:.2f} Da, LogP: {row['LogP']:.2f}")
            report.append(f"  Lipinski violations: {row['Lipinski_Violations']}")
        
        return "\n".join(report)

# Example usage and demonstration
def demo():
    """Run a demonstration of the molecular explorer"""
    
    print("\n🚀 Running Molecular Explorer Demo")
    print("-" * 60)
    
    # Create explorer instance
    explorer = MolecularExplorer()
    
    # Add some well-known drug molecules
    drugs = [
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
        ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'Ibuprofen'),
        ('CC(C)(C)NCC(O)c1ccc(O)c(O)c1', 'Albuterol'),
        ('CN1CCC[C@H]1c2cccnc2', 'Nicotine'),
        ('CC(=O)NC1=CC=C(C=C1)O', 'Acetaminophen'),
        ('CN(C)C(=N)NC(=N)N', 'Metformin'),
    ]
    
    print("\n📊 Adding molecules for analysis...")
    for smiles, name in drugs:
        mol = explorer.add_molecule(smiles, name)
        print(f"  ✓ {name}: MW={mol.properties['MW']:.1f}, LogP={mol.properties['LogP']:.2f}")
    
    # Generate variants for aspirin
    print("\n🔬 Generating variants for Aspirin...")
    aspirin_variants = explorer.generate_variants(drugs[0][0], n_variants=3)
    for i, variant in enumerate(aspirin_variants, 1):
        mol = explorer.add_molecule(variant, f"Aspirin_v{i}")
        print(f"  Variant {i}: {variant[:30]}...")
    
    # Analyze all molecules
    print("\n📈 Analyzing molecular properties...")
    results = explorer.analyze_molecules()
    
    print("\n📋 Analysis Results:")
    print(results[['Name', 'MW', 'LogP', 'QED_Score', 'Lipinski_Violations']].to_string())
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    explorer.visualize_results()
    
    # Generate report
    report = explorer.generate_report()
    print("\n" + report)
    
    # Save results
    filename = f"molecular_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    
    return explorer, results

# Interactive mode
def interactive_mode():
    """Run interactive molecular exploration"""
    
    print("\n🧬 INTERACTIVE MOLECULAR EXPLORER")
    print("="*60)
    print("Enter SMILES strings to analyze molecules")
    print("Commands: 'analyze', 'visualize', 'report', 'save', 'quit'")
    print("="*60)
    
    explorer = MolecularExplorer()
    
    while True:
        print("\nOptions:")
        print("  1. Add molecule (enter SMILES)")
        print("  2. Type 'analyze' to analyze all molecules")
        print("  3. Type 'visualize' to create plots")
        print("  4. Type 'report' to generate report")
        print("  5. Type 'save' to export results")
        print("  6. Type 'quit' to exit")
        
        user_input = input("\nEnter SMILES or command: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'analyze':
            if not explorer.molecules:
                print("No molecules added yet!")
                continue
            results = explorer.analyze_molecules()
            print("\n✅ Analysis complete!")
            print(results[['Name', 'MW', 'LogP', 'QED_Score']].to_string())
        elif user_input.lower() == 'visualize':
            explorer.visualize_results()
        elif user_input.lower() == 'report':
            print(explorer.generate_report())
        elif user_input.lower() == 'save':
            if explorer.results is not None:
                filename = f"molecular_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                explorer.results.to_csv(filename, index=False)
                print(f"✅ Saved to {filename}")
            else:
                print("No results to save!")
        else:
            # Assume it's a SMILES string
            name = input("Enter molecule name (optional): ").strip()
            try:
                mol = explorer.add_molecule(user_input, name or None)
                print(f"✅ Added: {mol.name}")
                print(f"   Properties: MW={mol.properties['MW']:.1f}, LogP={mol.properties['LogP']:.2f}")
                
                # Ask about variants
                gen_variants = input("Generate variants? (y/n): ").lower()
                if gen_variants == 'y':
                    n = int(input("Number of variants (1-10): ") or "3")
                    variants = explorer.generate_variants(user_input, min(n, 10))
                    for i, variant in enumerate(variants, 1):
                        var_mol = explorer.add_molecule(variant, f"{mol.name}_v{i}")
                        print(f"   Added variant {i}: MW={var_mol.properties['MW']:.1f}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    print("\nChoose mode:")
    print("  1. Run demo with example molecules")
    print("  2. Interactive mode")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo()
    elif choice == "2":
        interactive_mode()
    else:
        print("Running demo by default...")
        demo()
