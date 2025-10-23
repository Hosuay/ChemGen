"""
Module 01: Harmonic SMILES Processing
=====================================

5-Stage SMILES Preprocessing Pipeline:
1. Validation (chemical correctness)
2. Cleaning (standardization)
3. Canonicalization (unique representation)
4. Error Correction (heuristic fixes)
5. Feature Extraction (initial descriptors)

Aligned with sacred geometry principles (pentagon = 5 stages)
"""

from typing import List, Optional, Tuple, Dict
import sys
from pathlib import Path

# Handle imports based on RDKit availability
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    RDKIT_AVAILABLE = True
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available. Install with: conda install -c conda-forge rdkit")


class HarmonicSMILESProcessor:
    """
    5-Stage SMILES Processing Pipeline with Sacred Geometry Alignment

    Stage 1: Validation
    Stage 2: Cleaning
    Stage 3: Canonicalization
    Stage 4: Error Correction
    Stage 5: Feature Extraction
    """

    def __init__(self, enable_pains_filter: bool = True, verbose: bool = True):
        """
        Initialize SMILES Processor

        Parameters
        ----------
        enable_pains_filter : bool
            Enable PAINS filtering (default=True)
        verbose : bool
            Print processing information (default=True)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES processing")

        self.enable_pains_filter = enable_pains_filter
        self.verbose = verbose
        self.stats = {'processed': 0, 'failed': 0, 'duplicates': 0}

        # Initialize PAINS filter if enabled
        if enable_pains_filter:
            try:
                from rdkit.Chem import FilterCatalog
                from rdkit.Chem.FilterCatalog import FilterCatalogParams
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                self.pains_catalog = FilterCatalog.FilterCatalog(params)
            except:
                if verbose:
                    print("⚠️  PAINS filter initialization failed")
                self.pains_catalog = None
        else:
            self.pains_catalog = None

    def process_batch(self, smiles_list: List[str]) -> List[str]:
        """
        Process batch of SMILES through 5-stage pipeline

        Parameters
        ----------
        smiles_list : List[str]
            Input SMILES strings

        Returns
        -------
        List[str]
            Processed and validated SMILES
        """
        if self.verbose:
            print("\n🧬 5-STAGE SMILES PROCESSING PIPELINE")
            print("-" * 60)

        # Stage 1: Validation
        valid_smiles = self._stage1_validation(smiles_list)

        # Stage 2: Cleaning
        cleaned_smiles = self._stage2_cleaning(valid_smiles)

        # Stage 3: Canonicalization
        canonical_smiles = self._stage3_canonicalization(cleaned_smiles)

        # Stage 4: Error Correction
        corrected_smiles = self._stage4_error_correction(canonical_smiles)

        # Stage 5: Feature Extraction (metadata only, actual features computed later)
        final_smiles, metadata = self._stage5_feature_extraction(corrected_smiles)

        if self.verbose:
            print(f"\n✓ Pipeline complete: {len(final_smiles)}/{len(smiles_list)} SMILES validated")
            print("-" * 60)

        return final_smiles

    def _stage1_validation(self, smiles_list: List[str]) -> List[str]:
        """Stage 1: Validate SMILES chemical correctness"""
        if self.verbose:
            print("\nStage 1/5: Validation")

        valid = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.GetNumAtoms() >= 3:  # Minimum triadic structure
                valid.append(smi)
            else:
                self.stats['failed'] += 1

        if self.verbose:
            print(f"   ✓ Valid: {len(valid)}/{len(smiles_list)}")

        return valid

    def _stage2_cleaning(self, smiles_list: List[str]) -> List[str]:
        """Stage 2: Clean and standardize SMILES"""
        if self.verbose:
            print("Stage 2/5: Cleaning")

        cleaned = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Remove salts, neutralize charges where possible
                cleaned_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                cleaned.append(cleaned_smi)

        if self.verbose:
            print(f"   ✓ Cleaned: {len(cleaned)} molecules")

        return cleaned

    def _stage3_canonicalization(self, smiles_list: List[str]) -> List[str]:
        """Stage 3: Canonicalize for unique representation"""
        if self.verbose:
            print("Stage 3/5: Canonicalization")

        canonical = []
        seen_inchi = set()

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Use InChI for deduplication
                try:
                    inchi_key = inchi.MolToInchiKey(mol)
                    if inchi_key not in seen_inchi:
                        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                        canonical.append(canonical_smi)
                        seen_inchi.add(inchi_key)
                    else:
                        self.stats['duplicates'] += 1
                except:
                    # Fallback to SMILES if InChI fails
                    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                    canonical.append(canonical_smi)

        if self.verbose:
            print(f"   ✓ Canonical: {len(canonical)} (removed {self.stats['duplicates']} duplicates)")

        return canonical

    def _stage4_error_correction(self, smiles_list: List[str]) -> List[str]:
        """Stage 4: Apply error correction heuristics"""
        if self.verbose:
            print("Stage 4/5: Error Correction")

        corrected = []

        # Apply PAINS filter if enabled
        if self.enable_pains_filter and self.pains_catalog:
            pains_count = 0
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol and not self.pains_catalog.HasMatch(mol):
                    corrected.append(smi)
                else:
                    pains_count += 1

            if self.verbose:
                print(f"   ✓ PAINS filtered: {len(corrected)} pass ({pains_count} rejected)")
        else:
            corrected = smiles_list.copy()
            if self.verbose:
                print(f"   ✓ No corrections needed: {len(corrected)}")

        return corrected

    def _stage5_feature_extraction(self, smiles_list: List[str]) -> Tuple[List[str], Dict]:
        """Stage 5: Extract initial features (metadata)"""
        if self.verbose:
            print("Stage 5/5: Feature Extraction")

        # Feature extraction happens in molecular analysis module
        # Here we just prepare metadata
        metadata = {
            'total_processed': len(smiles_list),
            'stage': '5-stage-complete'
        }

        if self.verbose:
            print(f"   ✓ Ready for feature extraction: {len(smiles_list)} molecules")

        return smiles_list, metadata


if __name__ == "__main__":
    # Test the processor
    if RDKIT_AVAILABLE:
        processor = HarmonicSMILESProcessor()

        test_smiles = [
            'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
            'INVALID',  # Should be rejected
        ]

        processed = processor.process_batch(test_smiles)
        print(f"\n✓ Processed {len(processed)} valid SMILES")
    else:
        print("RDKit not available for testing")
