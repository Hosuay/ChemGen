"""
VariantProject v3.6.9 - Harmonic Molecular Explorer
===================================================

Main orchestration module integrating all 12 sub-modules with sacred geometry alignment.

Architecture:
- 3 Pipeline Stages: Input → Transform → Output
- 12 Modules: Dodecadic system organization
- 27 Latent Blocks: Cubic harmonic (3³) partitioning
- 33 Descriptors: Master number feature extraction
- 369 Master Frequency: Harmonic system alignment

Author: VariantProject Team
License: MIT
Version: 3.6.9 (Harmonic Master Release)
"""

import sys
import os
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings

# Import configuration
from harmonic_config import config

# Import core modules
from modules_01_smiles_processing.smiles_processor import HarmonicSMILESProcessor
from modules_02_variant_generation.variant_generator import HarmonicVariantGenerator
from modules_03_molecular_analysis.molecular_analyzer import HarmonicMolecularAnalyzer
from modules_04_ml_scoring.ml_scorer import HarmonicMLScorer
from modules_05_visualization.visualizer import HarmonicVisualizer
from modules_06_export.exporter import HarmonicExporter

warnings.filterwarnings('ignore')


class HarmonicVariantExplorer:
    """
    Main Harmonic Molecular Explorer

    Orchestrates the complete molecular exploration pipeline with sacred geometry alignment:

    Stage 1 (INPUT): SMILES Processing
        - 5-stage preprocessing
        - Validation, cleaning, canonicalization
        - InChI-based deduplication

    Stage 2 (TRANSFORM): Variant Generation & Analysis
        - 27-block latent partitioning
        - 33 primary molecular descriptors
        - 7-layer ML scoring networks

    Stage 3 (OUTPUT): Visualization & Export
        - Harmonic visualizations
        - 9-step export processing
        - Pareto optimization
    """

    def __init__(self,
                 enable_pains_filter: bool = True,
                 enable_27_block_partition: bool = True,
                 enable_33_descriptors: bool = True,
                 ml_layers: int = 7,
                 training_cycles: str = '3-6-9',
                 post_processing_stages: int = 9,
                 use_sacred_geometry: bool = True,
                 verbose: bool = True):
        """
        Initialize Harmonic Variant Explorer

        Parameters
        ----------
        enable_pains_filter : bool
            Enable PAINS filtering (default=True)
        enable_27_block_partition : bool
            Enable 27-block latent partitioning (default=True)
        enable_33_descriptors : bool
            Compute all 33 primary descriptors (default=True)
        ml_layers : int
            Number of ML network layers (default=7, harmonic)
        training_cycles : str
            Training cycle pattern (default='3-6-9', harmonic)
        post_processing_stages : int
            Number of post-processing stages (default=9, enneadic)
        use_sacred_geometry : bool
            Enforce sacred geometry alignment (default=True)
        verbose : bool
            Print progress information (default=True)
        """

        self.config = config
        self.enable_pains_filter = enable_pains_filter
        self.enable_27_block_partition = enable_27_block_partition
        self.enable_33_descriptors = enable_33_descriptors
        self.ml_layers = ml_layers
        self.training_cycles = training_cycles
        self.post_processing_stages = post_processing_stages
        self.use_sacred_geometry = use_sacred_geometry
        self.verbose = verbose

        # Validate harmonic alignment if required
        if use_sacred_geometry:
            self._validate_harmonic_parameters()

        # Initialize modules
        self._initialize_modules()

        # Storage for results
        self.results = None
        self.input_smiles = []
        self.metadata = {}

        if self.verbose:
            self._print_banner()

    def _validate_harmonic_parameters(self):
        """Validate parameters align with sacred geometry"""
        issues = []

        if self.ml_layers != 7:
            issues.append(f"ML layers ({self.ml_layers}) should be 7 for harmonic alignment")

        if self.post_processing_stages != 9:
            issues.append(f"Post-processing stages ({self.post_processing_stages}) should be 9 (enneadic)")

        if self.training_cycles != '3-6-9':
            issues.append(f"Training cycles ({self.training_cycles}) should be '3-6-9' pattern")

        if issues and self.verbose:
            print("⚠️  Harmonic Alignment Warnings:")
            for issue in issues:
                print(f"   • {issue}")
            print()

    def _initialize_modules(self):
        """Initialize all 12 modules"""

        # Module 01: SMILES Processing
        self.smiles_processor = HarmonicSMILESProcessor(
            enable_pains_filter=self.enable_pains_filter,
            verbose=self.verbose
        )

        # Module 02: Variant Generation
        self.variant_generator = HarmonicVariantGenerator(
            enable_27_blocks=self.enable_27_block_partition,
            verbose=self.verbose
        )

        # Module 03: Molecular Analysis
        self.molecular_analyzer = HarmonicMolecularAnalyzer(
            compute_33_descriptors=self.enable_33_descriptors,
            post_processing_stages=self.post_processing_stages,
            verbose=self.verbose
        )

        # Module 04: ML Scoring
        self.ml_scorer = HarmonicMLScorer(
            n_layers=self.ml_layers,
            training_pattern=self.training_cycles,
            verbose=self.verbose
        )

        # Module 05: Visualization
        self.visualizer = HarmonicVisualizer(
            use_harmonic_colors=True,
            verbose=self.verbose
        )

        # Module 06: Export
        self.exporter = HarmonicExporter(
            post_processing_stages=self.post_processing_stages,
            verbose=self.verbose
        )

    def _print_banner(self):
        """Print harmonic banner"""
        print("=" * 80)
        print("🔯 HARMONIC VARIANT EXPLORER v3.6.9")
        print("=" * 80)
        print(f"\nSacred Geometry Configuration:")
        print(f"  • Modules: 12 (Dodecadic)")
        print(f"  • Pipeline Stages: 3 (Triadic)")
        print(f"  • Latent Blocks: {27 if self.enable_27_block_partition else 'Disabled'}")
        print(f"  • Primary Descriptors: {33 if self.enable_33_descriptors else 'Standard'}")
        print(f"  • ML Layers: {self.ml_layers}")
        print(f"  • Training Cycles: {self.training_cycles}")
        print(f"  • Post-Processing Stages: {self.post_processing_stages}")
        print(f"  • Master Frequency: 369")
        print("=" * 80)
        print()

    def explore(self,
                smiles_list: List[str],
                n_variants: int = 36,
                similarity_threshold: float = 0.369,
                preserve_scaffold: bool = True,
                use_bioisosteric: bool = True,
                apply_pareto_optimization: bool = True,
                generate_3d_structures: bool = True,
                harmonic_visualization: bool = True,
                export_results: bool = True) -> pd.DataFrame:
        """
        Run complete harmonic exploration pipeline

        Parameters
        ----------
        smiles_list : List[str]
            Input SMILES strings
        n_variants : int
            Number of variants per molecule (default=36, 6² hexadic)
        similarity_threshold : float
            Tanimoto similarity threshold (default=0.369, master frequency)
        preserve_scaffold : bool
            Preserve Murcko scaffolds (default=True)
        use_bioisosteric : bool
            Use bioisosteric replacements (default=True)
        apply_pareto_optimization : bool
            Apply Pareto multi-objective ranking (default=True)
        generate_3d_structures : bool
            Generate 3D conformers for top molecules (default=True)
        harmonic_visualization : bool
            Use harmonic color schemes (default=True)
        export_results : bool
            Export results to CSV (default=True)

        Returns
        -------
        pd.DataFrame
            Results dataframe with all molecules, scores, and metadata
        """

        if self.verbose:
            print("\n" + "="*80)
            print("🔬 STAGE 1/3: INPUT PROCESSING")
            print("="*80)

        # Stage 1: SMILES Processing (5 steps)
        processed_smiles = self.smiles_processor.process_batch(smiles_list)

        if len(processed_smiles) == 0:
            print("❌ No valid SMILES after processing")
            return None

        self.input_smiles = processed_smiles

        if self.verbose:
            print(f"\n✓ Processed {len(processed_smiles)} valid input molecules")
            print("\n" + "="*80)
            print("⚗️  STAGE 2/3: TRANSFORMATION")
            print("="*80)

        # Stage 2A: Variant Generation (27 blocks)
        all_variants = self.variant_generator.generate_variants_batch(
            processed_smiles,
            n_variants=n_variants,
            similarity_threshold=similarity_threshold,
            preserve_scaffold=preserve_scaffold,
            use_bioisosteric=use_bioisosteric
        )

        if len(all_variants) == 0:
            print("❌ No variants generated")
            return None

        # Stage 2B: Molecular Analysis (33 descriptors, 9-step processing)
        df_analyzed = self.molecular_analyzer.analyze_batch(all_variants)

        # Stage 2C: ML Scoring (7-layer networks, 3-6-9 training)
        df_scored = self.ml_scorer.score_molecules(df_analyzed)

        if self.verbose:
            print("\n" + "="*80)
            print("📊 STAGE 3/3: OUTPUT")
            print("="*80)

        # Apply Pareto optimization if enabled
        if apply_pareto_optimization:
            df_scored = self._apply_pareto_ranking(df_scored)

        # Sort by composite score
        self.results = df_scored.sort_values(
            'Composite_Score' if not apply_pareto_optimization else ['Pareto_Rank', 'Composite_Score'],
            ascending=[True, False] if apply_pareto_optimization else False
        ).reset_index(drop=True)

        # Display top results
        if self.verbose:
            self._display_top_results()

        # Generate 3D structures for top molecules
        if generate_3d_structures and self.verbose:
            self._generate_top_3d_structures()

        # Visualizations
        if harmonic_visualization and self.verbose:
            self.visualizer.visualize_results(self.results)

        # Export
        if export_results:
            self.exporter.export_results(
                self.results,
                input_smiles=self.input_smiles,
                metadata=self._get_metadata()
            )

        if self.verbose:
            print("\n" + "="*80)
            print("✅ HARMONIC EXPLORATION COMPLETE")
            print("="*80)
            print(f"\nTotal molecules evaluated: {len(self.results)}")
            if 'Pareto_Efficient' in self.results.columns:
                n_pareto = self.results['Pareto_Efficient'].sum()
                print(f"Pareto-efficient molecules: {n_pareto}")
            if 'Latent_Block' in self.results.columns:
                unique_blocks = self.results['Latent_Block'].nunique()
                print(f"Latent blocks explored: {unique_blocks}/27")
            print("="*80)
            print()

        return self.results

    def _apply_pareto_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Pareto multi-objective optimization"""
        if self.verbose:
            print("\n🎯 Applying Pareto Multi-Objective Optimization...")

        from modules_04_ml_scoring.pareto_optimizer import ParetoOptimizer

        df_pareto = ParetoOptimizer.rank_by_pareto(
            df,
            objectives=['Pred_Solubility', 'Lab_Feasibility'],
            minimize=['Pred_Toxicity']
        )

        n_efficient = df_pareto['Pareto_Efficient'].sum()
        if self.verbose:
            print(f"   ✓ Identified {n_efficient} Pareto-efficient molecules")

        return df_pareto

    def _display_top_results(self, n: int = 9):
        """Display top N results (enneadic)"""
        print(f"\n🏆 TOP {n} MOLECULES (Enneadic Display)")
        print("="*80)

        display_cols = ['SMILES', 'MolWt', 'MolLogP', 'TPSA', 'Composite_Score']

        if 'QED' in self.results.columns:
            display_cols.insert(4, 'QED')
        if 'Pareto_Rank' in self.results.columns:
            display_cols.insert(-1, 'Pareto_Rank')
        if 'Latent_Block' in self.results.columns:
            display_cols.append('Latent_Block')

        # Only show columns that exist
        display_cols = [c for c in display_cols if c in self.results.columns]

        top_df = self.results.head(n)[display_cols].copy()

        # Format for display
        for col in ['MolWt', 'MolLogP', 'TPSA', 'Composite_Score']:
            if col in top_df.columns:
                top_df[col] = top_df[col].round(3)

        print(top_df.to_string(index=True))
        print()

    def _generate_top_3d_structures(self, n: int = 3):
        """Generate 3D structures for top molecules (triadic)"""
        if self.verbose:
            print(f"\n🌐 Generating 3D Structures for Top {n} Molecules (Triadic)")
            print("-"*80)

        from modules_05_visualization.structure_3d import Structure3D

        for idx, row in self.results.head(n).iterrows():
            print(f"\nRank #{idx+1}: {row['SMILES'][:50]}...")
            info_parts = [
                f"Score: {row['Composite_Score']:.4f}",
                f"MW: {row['MolWt']:.1f}",
                f"LogP: {row['MolLogP']:.2f}"
            ]

            if 'Pareto_Rank' in row:
                info_parts.append(f"Pareto: {int(row['Pareto_Rank'])}")
            if 'Latent_Block' in row:
                info_parts.append(f"Block: {int(row['Latent_Block'])}")

            print("   " + " | ".join(info_parts))

            # Generate 3D conformer
            mol_3d = Structure3D.generate_3d_conformer(row['SMILES'])
            if mol_3d:
                print("   ✓ 3D structure generated")
            else:
                print("   ⚠️  Could not generate 3D structure")

    def _get_metadata(self) -> Dict:
        """Get exploration metadata"""
        return {
            'version': '3.6.9',
            'timestamp': datetime.now().isoformat(),
            'input_count': len(self.input_smiles),
            'total_evaluated': len(self.results) if self.results is not None else 0,
            'harmonic_alignment': self.use_sacred_geometry,
            'latent_blocks_enabled': self.enable_27_block_partition,
            'descriptors_count': 33 if self.enable_33_descriptors else 'standard',
            'ml_layers': self.ml_layers,
            'training_cycles': self.training_cycles,
            'master_frequency': 369
        }

    def get_top_molecules(self, n: int = 9) -> pd.DataFrame:
        """
        Get top N molecules (default=9, enneadic)

        Parameters
        ----------
        n : int
            Number of top molecules to return (default=9)

        Returns
        -------
        pd.DataFrame
            Top N molecules with all data
        """
        if self.results is None:
            print("⚠️  No results available. Run explore() first.")
            return None

        return self.results.head(n)

    def visualize_pareto_frontier(self):
        """Visualize Pareto frontier"""
        if self.results is None:
            print("⚠️  No results available. Run explore() first.")
            return

        self.visualizer.plot_pareto_frontier(self.results)

    def visualize_harmonic_distribution(self):
        """Visualize harmonic distribution of results"""
        if self.results is None:
            print("⚠️  No results available. Run explore() first.")
            return

        self.visualizer.plot_harmonic_distribution(self.results)

    def export_harmonic_csv(self, filename: Optional[str] = None):
        """
        Export results with harmonic formatting

        Parameters
        ----------
        filename : str, optional
            Output filename. If None, auto-generates with timestamp
        """
        if self.results is None:
            print("⚠️  No results available. Run explore() first.")
            return

        self.exporter.export_results(
            self.results,
            input_smiles=self.input_smiles,
            metadata=self._get_metadata(),
            filename=filename
        )

    def get_configuration_summary(self) -> Dict:
        """Get current configuration summary"""
        return self.config.get_config_summary()


# Convenience function for quick exploration
def quick_explore(smiles_list: List[str],
                  n_variants: int = 36,
                  similarity_threshold: float = 0.369) -> Tuple[HarmonicVariantExplorer, pd.DataFrame]:
    """
    Quick harmonic exploration with default parameters

    Parameters
    ----------
    smiles_list : List[str]
        Input SMILES strings
    n_variants : int
        Variants per molecule (default=36, hexadic squared)
    similarity_threshold : float
        Tanimoto threshold (default=0.369, master frequency)

    Returns
    -------
    Tuple[HarmonicVariantExplorer, pd.DataFrame]
        Explorer instance and results dataframe

    Example
    -------
    >>> explorer, results = quick_explore([
    ...     'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    ...     'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # Caffeine
    ... ])
    """
    explorer = HarmonicVariantExplorer()
    results = explorer.explore(
        smiles_list=smiles_list,
        n_variants=n_variants,
        similarity_threshold=similarity_threshold
    )
    return explorer, results


if __name__ == "__main__":
    # Example usage
    print("🔯 Harmonic Variant Explorer v3.6.9")
    print("="*80)
    print("\nExample usage:")
    print("""
from modules.harmonic_explorer import HarmonicVariantExplorer, quick_explore

# Quick exploration
explorer, results = quick_explore([
    'CC(=O)Oc1ccccc1C(=O)O',       # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' # Caffeine
])

# View top 9 molecules (enneadic)
print(results.head(9))

# Visualize Pareto frontier
explorer.visualize_pareto_frontier()

# Export with harmonic formatting
explorer.export_harmonic_csv()
""")
    print("="*80)
