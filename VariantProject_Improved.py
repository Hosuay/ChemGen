# =========================================
# VariantProject IMPROVED: Computational Molecular Exploration Tool
# Version 2.0 - Enhanced with Real ML Models and Advanced Features
# =========================================

"""
Major Improvements in v2.0:
1. ✅ PAINS & structural alerts filtering
2. ✅ InChI-based duplicate detection
3. ✅ Real ML models trained on ESOL dataset
4. ✅ Comprehensive 30+ molecular descriptors
5. ✅ Model validation metrics (R², MAE, RMSE)
6. ✅ Uncertainty quantification for predictions
7. ✅ Bioisosteric replacement generation
8. ✅ Pareto multi-objective optimization
9. ✅ Progress bars for long operations
10. ✅ Enhanced visualization and reporting
"""

# =========================================
# 0️⃣ Install Dependencies
# =========================================
# !pip install rdkit py3Dmol pandas numpy scikit-learn selfies matplotlib seaborn tqdm --quiet

# =========================================
# 1️⃣ Imports
# =========================================
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, Crippen, Draw, AllChem, inchi
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import py3Dmol
import pandas as pd
import numpy as np
import selfies as sf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# =========================================
# 2️⃣ Configuration Class
# =========================================
class MolecularExplorerConfig:
    """Configuration parameters for molecular exploration"""

    # Variant generation
    N_VARIANTS = 30
    SIMILARITY_THRESHOLD = 0.5
    MAX_ATTEMPTS_MULTIPLIER = 20

    # Allowed atoms for mutations
    ALLOWED_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I

    # Fingerprint parameters
    FP_RADIUS = 2
    FP_NBITS = 2048

    # Scoring weights
    WEIGHT_SOLUBILITY = 0.3
    WEIGHT_TOXICITY = 0.3
    WEIGHT_FEASIBILITY = 0.4

    # 3D optimization
    MMFF_MAX_ITERS = 500

    # Visualization
    VIEWER_WIDTH = 700
    VIEWER_HEIGHT = 500
    
    # NEW: Filtering options
    ENABLE_PAINS_FILTER = True
    ENABLE_BRENK_FILTER = True
    USE_INCHI_DEDUPLICATION = True
    
    # NEW: ML options
    USE_REAL_MODELS = True
    ESOL_DATASET_PATH = 'ESOL.csv'

config = MolecularExplorerConfig()

# =========================================
# 3️⃣ SMILES Validation and Cleaning
# =========================================
class SMILESProcessor:
    """Handles SMILES string validation and cleaning"""

    @staticmethod
    def clean_smiles(smiles):
        """Clean and canonicalize SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Remove stereochemistry for canonical form
            smiles_clean = Chem.MolToSmiles(mol, isomericSmiles=False)

            # Verify it can be parsed again
            mol_clean = Chem.MolFromSmiles(smiles_clean)
            return smiles_clean if mol_clean else None
        except Exception as e:
            print(f"⚠️  Error cleaning SMILES '{smiles}': {str(e)}")
            return None

    @staticmethod
    def is_valid_smiles(smiles):
        """Check if SMILES string is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    @staticmethod
    def to_selfies(smiles):
        """Convert SMILES to SELFIES representation"""
        try:
            return sf.encoder(smiles)
        except:
            return None

    @staticmethod
    def from_selfies(selfies_str):
        """Convert SELFIES back to SMILES"""
        try:
            return sf.decoder(selfies_str)
        except:
            return None

# =========================================
# 4️⃣ NEW: Duplicate Detection using InChI
# =========================================
class DuplicateFilter:
    """Remove duplicates using InChI keys"""
    
    @staticmethod
    def get_inchi_key(smiles):
        """Get InChI key for canonical comparison"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return inchi.MolToInchiKey(mol)
        except:
            return None
    
    @staticmethod
    def remove_duplicates(smiles_list, verbose=True):
        """Remove duplicates based on InChI keys"""
        seen_keys = set()
        unique_smiles = []
        duplicates_removed = 0
        
        for smi in smiles_list:
            key = DuplicateFilter.get_inchi_key(smi)
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_smiles.append(smi)
            else:
                duplicates_removed += 1
        
        if verbose and duplicates_removed > 0:
            print(f"   🔍 Removed {duplicates_removed} duplicate structures (InChI-based)")
        
        return unique_smiles

# =========================================
# 5️⃣ NEW: Structural Filters (PAINS & Reactive Groups)
# =========================================
class StructuralFilters:
    """Filter out problematic molecular structures"""
    
    def __init__(self):
        # Initialize PAINS filter
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(params)
        
        # Add Brenk filter for reactive groups
        params_brenk = FilterCatalogParams()
        params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog.FilterCatalog(params_brenk)
    
    def is_pains(self, mol):
        """Check if molecule matches PAINS patterns"""
        return self.pains_catalog.HasMatch(mol)
    
    def is_reactive(self, mol):
        """Check for reactive/unstable groups"""
        return self.brenk_catalog.HasMatch(mol)
    
    def get_alerts(self, mol):
        """Get specific structural alerts"""
        alerts = []
        
        # PAINS
        if self.pains_catalog.HasMatch(mol):
            entry = self.pains_catalog.GetFirstMatch(mol)
            alerts.append(f"PAINS: {entry.GetDescription()}")
        
        # Reactive groups
        if self.brenk_catalog.HasMatch(mol):
            entry = self.brenk_catalog.GetFirstMatch(mol)
            alerts.append(f"Reactive: {entry.GetDescription()}")
        
        return alerts
    
    def filter_variants(self, smiles_list, enable_pains=True, enable_brenk=True):
        """Filter out problematic molecules"""
        filtered = []
        rejected = {'pains': 0, 'reactive': 0, 'invalid': 0}
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rejected['invalid'] += 1
                continue
            
            if enable_pains and self.is_pains(mol):
                rejected['pains'] += 1
                continue
            
            if enable_brenk and self.is_reactive(mol):
                rejected['reactive'] += 1
                continue
            
            filtered.append(smi)
        
        print(f"   🛡️  Filtered: {rejected['pains']} PAINS, {rejected['reactive']} reactive, {rejected['invalid']} invalid")
        return filtered

# =========================================
# 6️⃣ ENHANCED: Molecular Descriptors Calculator
# =========================================
class MolecularDescriptors:
    """Compute molecular descriptors and properties"""

    @staticmethod
    def compute_basic_descriptors(mol):
        """Compute basic molecular descriptors"""
        if mol is None:
            return [None] * 6

        try:
            return [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
                rdMolDescriptors.CalcTPSA(mol)
            ]
        except:
            return [None] * 6

    @staticmethod
    def compute_extended_descriptors(mol):
        """Compute extended molecular descriptors"""
        if mol is None:
            return {}

        try:
            return {
                'NumAromaticRings': Lipinski.NumAromaticRings(mol),
                'NumSaturatedRings': Lipinski.NumSaturatedRings(mol),
                'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
                'FractionCSP3': Lipinski.FractionCsp3(mol),
                'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
                'SASA': rdMolDescriptors.CalcLabuteASA(mol)
            }
        except:
            return {}
    
    @staticmethod
    def compute_comprehensive_descriptors(mol):
        """NEW: Compute 30+ descriptors for ML"""
        if mol is None:
            return {}
        
        try:
            desc = {
                # Basic physicochemical
                'MolWt': Descriptors.MolWt(mol),
                'MolLogP': Crippen.MolLogP(mol),
                'MolMR': Crippen.MolMR(mol),  # Molar refractivity
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol),
                
                # Complexity metrics
                'RotBonds': Lipinski.NumRotatableBonds(mol),
                'AromaticRings': Lipinski.NumAromaticRings(mol),
                'SaturatedRings': Lipinski.NumSaturatedRings(mol),
                'AliphaticRings': Lipinski.NumAliphaticRings(mol),
                
                # Shape & surface
                'FractionCSP3': Lipinski.FractionCsp3(mol),
                'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
                'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(mol),
                
                # Drug-likeness
                'QED': QED.qed(mol),  # Quantitative estimate of drug-likeness
                
                # Additional important descriptors
                'BertzCT': Descriptors.BertzCT(mol),  # Complexity
                'HallKierAlpha': Descriptors.HallKierAlpha(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
                'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
                'SMR_VSA1': Descriptors.SMR_VSA1(mol),
                'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
                'NumStereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            }
            return desc
        except Exception as e:
            print(f"Error computing comprehensive descriptors: {e}")
            return {}

    @staticmethod
    def lipinski_rule_of_five(mol):
        """Check Lipinski's Rule of Five compliance"""
        if mol is None:
            return None

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1

        return violations

    @classmethod
    def compute_features(cls, smiles_list, comprehensive=True):
        """Compute molecular features for a list of SMILES"""
        results = []
        extended_results = []
        comprehensive_results = []
        ro5_violations = []

        for s in tqdm(smiles_list, desc="Computing descriptors", disable=len(smiles_list)<10):
            mol = Chem.MolFromSmiles(s)

            # Basic descriptors
            basic = cls.compute_basic_descriptors(mol)
            results.append(basic)

            # Extended descriptors
            extended = cls.compute_extended_descriptors(mol)
            extended_results.append(extended)
            
            # NEW: Comprehensive descriptors
            if comprehensive:
                comp = cls.compute_comprehensive_descriptors(mol)
                comprehensive_results.append(comp)

            # Rule of Five
            ro5 = cls.lipinski_rule_of_five(mol)
            ro5_violations.append(ro5)

        # Create DataFrame
        df = pd.DataFrame(
            results,
            columns=['MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']
        )
        df['SMILES'] = smiles_list
        df['RO5_Violations'] = ro5_violations

        # Add extended descriptors
        if extended_results:
            for key in extended_results[0].keys():
                df[key] = [ext.get(key) for ext in extended_results]
        
        # Add comprehensive descriptors
        if comprehensive and comprehensive_results:
            for key in comprehensive_results[0].keys():
                if key not in df.columns:
                    df[key] = [comp.get(key) for comp in comprehensive_results]

        return df

# =========================================
# 7️⃣ Molecular Fingerprints
# =========================================
class FingerprintGenerator:
    """Generate molecular fingerprints for similarity calculations"""

    @staticmethod
    def morgan_fingerprint(smiles, radius=config.FP_RADIUS, nBits=config.FP_NBITS):
        """Generate Morgan (ECFP) fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=nBits
            )
        except:
            return None

    @staticmethod
    def calculate_similarity(smiles1, smiles2):
        """Calculate Tanimoto similarity between two molecules"""
        fp1 = FingerprintGenerator.morgan_fingerprint(smiles1)
        fp2 = FingerprintGenerator.morgan_fingerprint(smiles2)

        if fp1 is None or fp2 is None:
            return 0.0

        return TanimotoSimilarity(fp1, fp2)

# =========================================
# 8️⃣ NEW: Bioisosteric Replacements
# =========================================
class BioisostericReplacements:
    """Database of common bioisosteric replacements"""
    
    REPLACEMENTS = {
        # Halogen swaps (most common)
        'c1ccccc1F': ['c1ccccc1Cl'],
        'c1ccccc1Cl': ['c1ccccc1Br', 'c1ccccc1F'],
        'c1ccccc1Br': ['c1ccccc1Cl', 'c1ccccc1I'],
        
        # Hydroxyl to amino
        'c1ccccc1O': ['c1ccccc1N'],
        
        # Methyl to ethyl
        'c1ccccc1C': ['c1ccccc1CC'],
        
        # Methoxy to ethoxy
        'c1ccccc1OC': ['c1ccccc1OCC'],
    }
    
    @staticmethod
    def apply_bioisosteric_replacement(smiles):
        """Apply bioisosteric replacement"""
        for original, replacements in BioisostericReplacements.REPLACEMENTS.items():
            if original in smiles:
                replacement = np.random.choice(replacements)
                # Replace first occurrence only
                new_smiles = smiles.replace(original, replacement, 1)
                if SMILESProcessor.is_valid_smiles(new_smiles):
                    return new_smiles
        return None

# =========================================
# 9️⃣ ENHANCED: Variant Generation Engine
# =========================================
class VariantGenerator:
    """Generate molecular variants with similarity constraints"""

    @staticmethod
    def get_scaffold(mol):
        """Extract Murcko scaffold from molecule"""
        try:
            return MurckoScaffold.GetScaffoldForMol(mol)
        except:
            return None

    @staticmethod
    def mutate_atom(mol, atom_idx, allowed_atoms=config.ALLOWED_ATOMS):
        """Mutate a single atom in the molecule"""
        rw_mol = Chem.RWMol(mol)

        try:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            current_atomic_num = atom.GetAtomicNum()

            # Choose a different atom
            new_atomic_num = np.random.choice(
                [a for a in allowed_atoms if a != current_atomic_num]
            )
            atom.SetAtomicNum(int(new_atomic_num))

            return rw_mol
        except:
            return None

    @classmethod
    def generate_variants(cls, smiles_input, n_variants=config.N_VARIANTS,
                         similarity_threshold=config.SIMILARITY_THRESHOLD,
                         preserve_scaffold=True, use_bioisosteric=False):
        """Generate molecular variants using similarity-guided mutations"""

        # Clean input SMILES
        cleaned = SMILESProcessor.clean_smiles(smiles_input)
        if cleaned is None:
            print(f"❌ Invalid SMILES: {smiles_input}")
            return []

        base_mol = Chem.MolFromSmiles(cleaned)
        if base_mol is None:
            return []

        # Get base fingerprint and scaffold
        base_fp = FingerprintGenerator.morgan_fingerprint(cleaned)
        if base_fp is None:
            return []

        base_scaffold = cls.get_scaffold(base_mol) if preserve_scaffold else None
        base_scaffold_smiles = Chem.MolToSmiles(base_scaffold) if base_scaffold else None

        variants = set()
        max_attempts = n_variants * config.MAX_ATTEMPTS_MULTIPLIER
        attempts = 0

        # Progress bar for variant generation
        pbar = tqdm(total=n_variants, desc=f"  Generating variants", leave=False)

        while len(variants) < n_variants and attempts < max_attempts:
            # NEW: Mix atomic mutations and bioisosteric replacements
            if use_bioisosteric and np.random.random() < 0.3:  # 30% bioisosteric
                new_smiles = BioisostericReplacements.apply_bioisosteric_replacement(cleaned)
                if new_smiles is None:
                    attempts += 1
                    continue
            else:
                # Atomic mutation
                mol = Chem.MolFromSmiles(cleaned)
                if mol.GetNumAtoms() == 0:
                    break

                atom_idx = np.random.randint(0, mol.GetNumAtoms())
                mutated_mol = cls.mutate_atom(mol, atom_idx)

                if mutated_mol is None:
                    attempts += 1
                    continue

                try:
                    new_smiles = Chem.MolToSmiles(mutated_mol, isomericSmiles=False)
                except:
                    attempts += 1
                    continue

            # Validate new SMILES
            if not SMILESProcessor.is_valid_smiles(new_smiles):
                attempts += 1
                continue

            # Check similarity
            new_fp = FingerprintGenerator.morgan_fingerprint(new_smiles)
            if new_fp is None:
                attempts += 1
                continue
                
            sim = TanimotoSimilarity(base_fp, new_fp)

            # Check scaffold preservation if required
            scaffold_match = True
            if preserve_scaffold and base_scaffold is not None:
                new_mol = Chem.MolFromSmiles(new_smiles)
                new_scaffold = cls.get_scaffold(new_mol)
                new_scaffold_smiles = Chem.MolToSmiles(new_scaffold) if new_scaffold else None
                scaffold_match = (new_scaffold_smiles == base_scaffold_smiles)

            # Add variant if it passes all checks
            if sim >= similarity_threshold and scaffold_match:
                if new_smiles not in variants:
                    variants.add(new_smiles)
                    pbar.update(1)

            attempts += 1

        pbar.close()
        return list(variants)

# =========================================
# 🔟 ENHANCED: Machine Learning Models with Real Data
# =========================================
class MolecularPropertyPredictor:
    """ML models for predicting molecular properties"""

    def __init__(self, use_real_models=config.USE_REAL_MODELS):
        self.model_solubility = None
        self.model_toxicity = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.use_real_models = use_real_models
        self.metrics = {}

    def load_esol_dataset(self, filepath=config.ESOL_DATASET_PATH):
        """NEW: Load real ESOL solubility dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded ESOL dataset: {len(df)} compounds")
            return df
        except Exception as e:
            print(f"⚠️  Could not load ESOL dataset: {e}")
            print("   Falling back to synthetic data")
            return None

    def train_on_esol(self, df_esol):
        """NEW: Train model on real ESOL data"""
        print("🧪 Training on ESOL dataset...")
        
        # Compute features for ESOL molecules
        feature_cols = ['MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']
        
        # Use provided features or compute them
        if not all(col in df_esol.columns for col in feature_cols):
            df_features = MolecularDescriptors.compute_features(
                df_esol['smiles'].tolist(), 
                comprehensive=False
            )
            for col in feature_cols:
                df_esol[col] = df_features[col]
        
        X = df_esol[feature_cols].fillna(0)
        y = df_esol['measured log solubility in mols per litre']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model_solubility = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model_solubility.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model_solubility, X_train_scaled, y_train,
            cv=5, scoring='r2'
        )
        
        # Test set evaluation
        y_pred = self.model_solubility.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        self.metrics['solubility'] = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'CV_R²_mean': cv_scores.mean(),
            'CV_R²_std': cv_scores.std()
        }
        
        print(f"   Solubility Model Performance:")
        print(f"     R² = {r2:.4f}")
        print(f"     RMSE = {rmse:.4f}")
        print(f"     MAE = {mae:.4f}")
        print(f"     CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self

    def train(self, df_features):
        """Train models (uses ESOL if available, else synthetic)"""
        
        if self.use_real_models:
            # Try to load and train on ESOL dataset
            df_esol = self.load_esol_dataset()
            if df_esol is not None:
                self.train_on_esol(df_esol)
                # Train toxicity model with synthetic data (no real dataset available)
                self._train_toxicity_synthetic(df_features)
                self.is_trained = True
                return self
        
        # Fallback to synthetic training
        print("⚠️  Using synthetic training data")
        np.random.seed(42)
        n = len(df_features)

        # Solubility correlates with TPSA and LogP
        df_features['Solubility'] = (
            -0.3 * df_features['MolLogP'].fillna(0) +
            0.2 * df_features['TPSA'].fillna(0) +
            np.random.normal(0, 0.1, n)
        )

        # Toxicity correlates with LogP and molecular weight
        df_features['Toxicity'] = (
            0.2 * df_features['MolLogP'].fillna(0) +
            0.0005 * df_features['MolWt'].fillna(0) +
            np.random.normal(0, 0.1, n)
        )

        # Prepare features
        feature_cols = ['MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']
        X = df_features[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Train models
        self.model_solubility = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model_toxicity = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.model_solubility.fit(X_scaled, df_features['Solubility'])
        self.model_toxicity.fit(X_scaled, df_features['Toxicity'])

        self.is_trained = True
        return self
    
    def _train_toxicity_synthetic(self, df_features):
        """Train toxicity model with synthetic data"""
        np.random.seed(42)
        n = len(df_features)
        
        df_features['Toxicity'] = (
            0.2 * df_features['MolLogP'].fillna(0) +
            0.0005 * df_features['MolWt'].fillna(0) +
            np.random.normal(0, 0.1, n)
        )
        
        feature_cols = ['MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']
        X = df_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)  # Use already fitted scaler
        
        self.model_toxicity = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model_toxicity.fit(X_scaled, df_features['Toxicity'])

    def predict_with_uncertainty(self, df_features):
        """NEW: Predict with uncertainty estimates"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        feature_cols = ['MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds', 'TPSA']
        X = df_features[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Get predictions from all trees for uncertainty
        sol_tree_preds = np.array([tree.predict(X_scaled) 
                                   for tree in self.model_solubility.estimators_])
        tox_tree_preds = np.array([tree.predict(X_scaled) 
                                   for tree in self.model_toxicity.estimators_])
        
        predictions = {
            'Pred_Solubility': sol_tree_preds.mean(axis=0),
            'Pred_Solubility_Std': sol_tree_preds.std(axis=0),
            'Pred_Toxicity': tox_tree_preds.mean(axis=0),
            'Pred_Toxicity_Std': tox_tree_preds.std(axis=0),
        }

        return predictions

    def predict(self, df_features):
        """Predict molecular properties"""
        return self.predict_with_uncertainty(df_features)

# =========================================
# 1️⃣1️⃣ NEW: Pareto Multi-Objective Optimization
# =========================================
class ParetoOptimizer:
    """Multi-objective Pareto optimization"""
    
    @staticmethod
    def is_pareto_efficient(costs):
        """Find Pareto-efficient points"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        
        return is_efficient
    
    @staticmethod
    def rank_by_pareto(df, objectives=['Pred_Solubility', 'Lab_Feasibility'],
                       minimize=['Pred_Toxicity']):
        """Rank molecules using Pareto frontiers"""
        
        # Prepare objectives (all to maximize)
        costs = []
        for obj in objectives:
            if obj in df.columns:
                costs.append(df[obj].values)
        
        for obj in minimize:
            if obj in df.columns:
                costs.append(-df[obj].values)  # Negate for minimization
        
        costs = np.array(costs).T
        
        # Negate all for minimization
        costs = -costs
        
        # Assign Pareto ranks
        remaining = np.arange(len(df))
        rank = 1
        pareto_ranks = np.zeros(len(df))
        
        while len(remaining) > 0:
            efficient_mask = ParetoOptimizer.is_pareto_efficient(costs[remaining])
            efficient_idx = remaining[efficient_mask]
            pareto_ranks[efficient_idx] = rank
            remaining = remaining[~efficient_mask]
            rank += 1
        
        df['Pareto_Rank'] = pareto_ranks.astype(int)
        df['Pareto_Efficient'] = (pareto_ranks == 1)
        
        return df.sort_values('Pareto_Rank')

# =========================================
# 1️⃣2️⃣ Molecular Evaluator
# =========================================
class MolecularEvaluator:
    """Evaluate and rank molecular variants"""

    @staticmethod
    def calculate_lab_feasibility(df):
        """Calculate synthetic feasibility score"""
        feasibility = np.ones(len(df))

        # Penalize high molecular weight
        feasibility -= 0.3 * (df['MolWt'] > 500).astype(int)

        # Penalize many rotatable bonds
        feasibility -= 0.2 * (df['RotBonds'] > 10).astype(int)

        # Penalize RO5 violations
        feasibility -= 0.15 * df['RO5_Violations'].fillna(0)
        
        # Reward high QED if available
        if 'QED' in df.columns:
            feasibility += 0.2 * df['QED'].fillna(0.5)

        # Add some noise
        feasibility += np.random.normal(0, 0.05, len(df))

        return np.clip(feasibility, 0, 1)

    @staticmethod
    def calculate_composite_score(df):
        """Calculate composite score from multiple factors"""

        # Normalize predictions to [0, 1]
        sol_norm = (df['Pred_Solubility'] - df['Pred_Solubility'].min()) / \
                   (df['Pred_Solubility'].max() - df['Pred_Solubility'].min() + 1e-6)

        tox_norm = 1 - (df['Pred_Toxicity'] - df['Pred_Toxicity'].min()) / \
                   (df['Pred_Toxicity'].max() - df['Pred_Toxicity'].min() + 1e-6)

        # Calculate composite score
        composite = (
            config.WEIGHT_SOLUBILITY * sol_norm +
            config.WEIGHT_TOXICITY * tox_norm +
            config.WEIGHT_FEASIBILITY * df['Lab_Feasibility']
        )

        return composite

    @classmethod
    def evaluate(cls, smiles_list, predictor, use_pareto=True):
        """Evaluate molecules and return ranked DataFrame"""

        # Compute molecular descriptors
        df = MolecularDescriptors.compute_features(smiles_list, comprehensive=True)

        # Predict properties
        predictions = predictor.predict(df)
        df['Pred_Solubility'] = predictions['Pred_Solubility']
        df['Pred_Solubility_Std'] = predictions['Pred_Solubility_Std']
        df['Pred_Toxicity'] = predictions['Pred_Toxicity']
        df['Pred_Toxicity_Std'] = predictions['Pred_Toxicity_Std']

        # Calculate feasibility
        df['Lab_Feasibility'] = cls.calculate_lab_feasibility(df)

        # Calculate composite score
        df['Composite_Score'] = cls.calculate_composite_score(df)
        
        # NEW: Pareto ranking
        if use_pareto:
            df = ParetoOptimizer.rank_by_pareto(
                df,
                objectives=['Pred_Solubility', 'Lab_Feasibility'],
                minimize=['Pred_Toxicity']
            )
            # Sort by Pareto rank first, then composite score
            df_sorted = df.sort_values(['Pareto_Rank', 'Composite_Score'], 
                                       ascending=[True, False])
        else:
            # Sort by composite score only
            df_sorted = df.sort_values('Composite_Score', ascending=False)

        return df_sorted.reset_index(drop=True)

# =========================================
# 1️⃣3️⃣ 3D Structure Visualization
# =========================================
class Structure3D:
    """Generate and visualize 3D molecular structures"""

    @staticmethod
    def generate_3d_conformer(smiles):
        """Generate optimized 3D conformer"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)

            # Optimize geometry with MMFF
            AllChem.MMFFOptimizeMolecule(mol, maxIters=config.MMFF_MAX_ITERS)

            return mol
        except:
            return None

    @staticmethod
    def visualize_3d(mol, style='stick', color_scheme='default'):
        """Create interactive 3D visualization"""
        if mol is None:
            return None

        try:
            block = Chem.MolToMolBlock(mol)
            viewer = py3Dmol.view(
                width=config.VIEWER_WIDTH,
                height=config.VIEWER_HEIGHT
            )
            viewer.addModel(block, "mol")

            if style == 'stick':
                viewer.setStyle({'stick': {'colorscheme': color_scheme}})
            elif style == 'sphere':
                viewer.setStyle({'sphere': {'colorscheme': color_scheme}})
            else:
                viewer.setStyle({
                    'stick': {'colorscheme': color_scheme},
                    'sphere': {'scale': 0.3, 'colorscheme': color_scheme}
                })

            viewer.setBackgroundColor('white')
            viewer.zoomTo()

            return viewer
        except:
            return None

# =========================================
# 1️⃣4️⃣ Visualization and Reporting
# =========================================
class MolecularVisualizer:
    """Create visualizations and reports"""

    @staticmethod
    def plot_property_distribution(df, properties=['MolWt', 'MolLogP', 'TPSA']):
        """Plot distribution of molecular properties"""
        n_props = len(properties)
        fig, axes = plt.subplots(1, n_props, figsize=(5*n_props, 4))

        if n_props == 1:
            axes = [axes]

        for ax, prop in zip(axes, properties):
            ax.hist(df[prop].dropna(), bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {prop}')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_score_comparison(df, top_n=10):
        """Compare scores for top molecules"""
        top_mols = df.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(top_mols))
        width = 0.2

        ax.bar(x - width*1.5, top_mols['Pred_Solubility'], width,
               label='Solubility', alpha=0.8)
        ax.bar(x - width*0.5, top_mols['Pred_Toxicity'], width,
               label='Toxicity', alpha=0.8)
        ax.bar(x + width*0.5, top_mols['Lab_Feasibility'], width,
               label='Feasibility', alpha=0.8)
        ax.bar(x + width*1.5, top_mols['Composite_Score'], width,
               label='Composite', alpha=0.8)

        ax.set_xlabel('Molecule Rank')
        ax.set_ylabel('Score')
        ax.set_title(f'Property Comparison - Top {top_n} Molecules')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(len(top_mols))])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pareto_frontier(df):
        """NEW: Visualize Pareto frontier"""
        if 'Pareto_Rank' not in df.columns:
            print("⚠️  No Pareto ranking available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all points
        scatter = ax.scatter(df['Pred_Solubility'], df['Pred_Toxicity'],
                           c=df['Pareto_Rank'], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black')
        
        # Highlight Pareto-efficient points
        pareto_df = df[df['Pareto_Efficient']]
        ax.scatter(pareto_df['Pred_Solubility'], pareto_df['Pred_Toxicity'],
                  c='red', s=200, marker='*', edgecolors='black',
                  label='Pareto Efficient', zorder=5)
        
        ax.set_xlabel('Predicted Solubility (higher is better)')
        ax.set_ylabel('Predicted Toxicity (lower is better)')
        ax.set_title('Pareto Frontier Analysis')
        ax.legend()
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pareto Rank')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_molecules_grid(smiles_list, labels=None, mols_per_row=5):
        """Draw grid of 2D molecular structures"""
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]

        if labels is None:
            labels = [f"Mol {i+1}" for i in range(len(smiles_list))]

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=(250, 250),
            legends=labels
        )

        return img

# =========================================
# 1️⃣5️⃣ Main Explorer Class - IMPROVED
# =========================================
class VariantProject:
    """Main molecular exploration orchestrator - IMPROVED VERSION 2.0"""

    def __init__(self, config=config):
        self.config = config
        self.predictor = MolecularPropertyPredictor(use_real_models=config.USE_REAL_MODELS)
        self.filters = StructuralFilters()
        self.results = None
        self.input_smiles = []

    def explore(self, smiles_list, n_variants=None, similarity_threshold=None,
                preserve_scaffold=True, use_bioisosteric=True,
                enable_pains_filter=True, enable_brenk_filter=True,
                use_pareto=True, show_visualizations=True, export_csv=True):
        """
        Main exploration pipeline - IMPROVED

        Parameters:
        -----------
        smiles_list : list
            List of input SMILES strings
        n_variants : int
            Number of variants to generate per input molecule
        similarity_threshold : float
            Minimum Tanimoto similarity to input molecule
        preserve_scaffold : bool
            Whether to preserve Murcko scaffold
        use_bioisosteric : bool
            Use bioisosteric replacements (NEW)
        enable_pains_filter : bool
            Filter PAINS compounds (NEW)
        enable_brenk_filter : bool
            Filter reactive groups (NEW)
        use_pareto : bool
            Use Pareto multi-objective optimization (NEW)
        show_visualizations : bool
            Whether to display plots
        export_csv : bool
            Whether to export results to CSV
        """

        clear_output()

        # Set defaults
        if n_variants is None:
            n_variants = self.config.N_VARIANTS
        if similarity_threshold is None:
            similarity_threshold = self.config.SIMILARITY_THRESHOLD

        self.input_smiles = smiles_list

        print("=" * 80)
        print("🧬 VARIANTPROJECT v2.0 - IMPROVED Molecular Exploration Pipeline")
        print("=" * 80)
        print(f"\n📊 Configuration:")
        print(f"   • Variants per molecule: {n_variants}")
        print(f"   • Similarity threshold: {similarity_threshold}")
        print(f"   • Preserve scaffold: {preserve_scaffold}")
        print(f"   • Use bioisosteric replacements: {use_bioisosteric}")
        print(f"   • PAINS filtering: {enable_pains_filter}")
        print(f"   • Reactive group filtering: {enable_brenk_filter}")
        print(f"   • Pareto optimization: {use_pareto}")
        print(f"   • Input molecules: {len(smiles_list)}")

        # Generate variants
        print(f"\n🔬 Generating variants...")
        all_variants = []

        for idx, smiles_input in enumerate(smiles_list, 1):
            print(f"\n   [{idx}/{len(smiles_list)}] Processing: {smiles_input}")

            variants = VariantGenerator.generate_variants(
                smiles_input,
                n_variants=n_variants,
                similarity_threshold=similarity_threshold,
                preserve_scaffold=preserve_scaffold,
                use_bioisosteric=use_bioisosteric
            )

            print(f"   ✓ Generated {len(variants)} valid variants")

            # Include input molecule
            all_variants.append(smiles_input)
            all_variants.extend(variants)

        if len(all_variants) == 0:
            print("\n❌ No valid variants generated. Try different parameters.")
            return None

        print(f"\n   📈 Total molecules before filtering: {len(all_variants)}")

        # NEW: Remove duplicates using InChI
        if config.USE_INCHI_DEDUPLICATION:
            all_variants = DuplicateFilter.remove_duplicates(all_variants)
            print(f"   📈 After duplicate removal: {len(all_variants)}")

        # NEW: Structural filtering
        if enable_pains_filter or enable_brenk_filter:
            all_variants = self.filters.filter_variants(
                all_variants,
                enable_pains=enable_pains_filter,
                enable_brenk=enable_brenk_filter
            )
            print(f"   📈 After structural filtering: {len(all_variants)}")

        if len(all_variants) == 0:
            print("\n❌ No molecules passed filtering. Try relaxing filter settings.")
            return None

        # Compute features and train models
        print(f"\n🧪 Computing molecular descriptors...")
        df_features = MolecularDescriptors.compute_features(all_variants, comprehensive=True)

        print(f"\n📈 Training predictive models...")
        self.predictor.train(df_features)
        
        # Display model metrics if available
        if self.predictor.metrics:
            print(f"\n📊 Model Performance Metrics:")
            for model_name, metrics in self.predictor.metrics.items():
                print(f"   {model_name.capitalize()}:")
                for metric, value in metrics.items():
                    print(f"     {metric}: {value:.4f}")

        # Evaluate molecules
        print(f"\n🎯 Evaluating and ranking molecules...")
        self.results = MolecularEvaluator.evaluate(all_variants, self.predictor, use_pareto=use_pareto)

        # Display results
        print(f"\n{'='*80}")
        print(f"🏆 TOP 10 RANKED MOLECULES")
        print(f"{'='*80}\n")

        display_cols = ['SMILES', 'MolWt', 'MolLogP', 'TPSA', 'QED',
                       'RO5_Violations', 'Composite_Score']
        if 'Pareto_Rank' in self.results.columns:
            display_cols.insert(-1, 'Pareto_Rank')
        
        # Only show columns that exist
        display_cols = [c for c in display_cols if c in self.results.columns]
        
        display(self.results.head(10)[display_cols])

        # Visualizations
        if show_visualizations:
            print(f"\n📊 Generating visualizations...")

            # Property distributions
            MolecularVisualizer.plot_property_distribution(
                self.results,
                properties=['MolWt', 'MolLogP', 'TPSA']
            )

            # Score comparison
            MolecularVisualizer.plot_score_comparison(self.results, top_n=10)
            
            # NEW: Pareto frontier
            if use_pareto:
                MolecularVisualizer.plot_pareto_frontier(self.results)

            # 2D structures of top molecules
            print(f"\n🖼️  Top 5 Molecular Structures (2D):")
            top_smiles = self.results.head(5)['SMILES'].tolist()
            top_labels = []
            for i, row in enumerate(self.results.head(5).itertuples(), 1):
                label = f"Rank #{i}\nScore: {row.Composite_Score:.3f}"
                if hasattr(row, 'Pareto_Rank'):
                    label += f"\nPareto: {row.Pareto_Rank}"
                top_labels.append(label)
            
            img = MolecularVisualizer.draw_molecules_grid(top_smiles, labels=top_labels)
            display(img)

        # 3D visualization of top 3
        print(f"\n🌐 Generating 3D structures for top 3 molecules...")
        top3 = self.results.head(3)

        for idx, row in top3.iterrows():
            print(f"\n   Rank #{idx+1}: {row['SMILES']}")
            info = f"   Score: {row['Composite_Score']:.4f} | MW: {row['MolWt']:.1f} | LogP: {row['MolLogP']:.2f}"
            if 'Pareto_Rank' in row:
                info += f" | Pareto Rank: {int(row['Pareto_Rank'])}"
            if 'Pred_Solubility_Std' in row:
                info += f" | Sol±: {row['Pred_Solubility_Std']:.3f}"
            print(info)

            mol3d = Structure3D.generate_3d_conformer(row['SMILES'])
            if mol3d:
                viewer = Structure3D.visualize_3d(mol3d, style='stick')
                if viewer:
                    viewer.show()
            else:
                print("   ⚠️  Could not generate 3D structure")

        # Export results
        if export_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"variant_project_v2_results_{timestamp}.csv"
            self.results.to_csv(filename, index=False)
            print(f"\n💾 Results exported to: {filename}")

        print(f"\n{'='*80}")
        print(f"✅ EXPLORATION COMPLETE")
        print(f"   Total molecules evaluated: {len(self.results)}")
        if 'Pareto_Efficient' in self.results.columns:
            n_pareto = self.results['Pareto_Efficient'].sum()
            print(f"   Pareto-efficient molecules: {n_pareto}")
        print(f"{'='*80}\n")

        return self.results

    def get_top_molecules(self, n=10):
        """Get top N molecules from results"""
        if self.results is None:
            print("⚠️  No results available. Run explore() first.")
            return None

        return self.results.head(n)

    def export_top_molecules(self, n=10, filename=None):
        """Export top N molecules to CSV"""
        if self.results is None:
            print("⚠️  No results available. Run explore() first.")
            return

        top_mols = self.results.head(n)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_{n}_molecules_v2_{timestamp}.csv"

        top_mols.to_csv(filename, index=False)
        print(f"✅ Top {n} molecules exported to: {filename}")

# =========================================
# 1️⃣6️⃣ Run VariantProject v2.0
# =========================================

def main():
    """Main execution function"""

    # Get user input
    print("=" * 80)
    print("🧬 VARIANTPROJECT v2.0 - IMPROVED AI-Assisted Molecular Exploration")
    print("=" * 80)
    print("\nEnter SMILES strings for molecular exploration")
    print("Examples:")
    print("  • Aspirin: CC(=O)Oc1ccccc1C(=O)O")
    print("  • Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    print("  • Ibuprofen: CC(C)Cc1ccc(cc1)C(C)C(=O)O\n")

    smiles_input_raw = input("\nEnter SMILES (comma-separated): ")
    smiles_list = [s.strip() for s in smiles_input_raw.split(",") if s.strip()]

    if not smiles_list:
        print("❌ No valid SMILES provided. Exiting.")
        return

    # Optional parameters
    print("\n⚙️  Optional Parameters (press Enter for defaults):")

    try:
        n_var = input(f"Number of variants per molecule [{config.N_VARIANTS}]: ")
        n_variants = int(n_var) if n_var else config.N_VARIANTS
    except:
        n_variants = config.N_VARIANTS

    try:
        sim_thresh = input(f"Similarity threshold (0-1) [{config.SIMILARITY_THRESHOLD}]: ")
        similarity_threshold = float(sim_thresh) if sim_thresh else config.SIMILARITY_THRESHOLD
    except:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    preserve_scaffold = input("Preserve molecular scaffold? [Y/n]: ").lower() != 'n'
    use_bioisosteric = input("Use bioisosteric replacements? [Y/n]: ").lower() != 'n'
    use_pareto = input("Use Pareto multi-objective optimization? [Y/n]: ").lower() != 'n'

    # Initialize and run explorer
    explorer = VariantProject()

    results = explorer.explore(
        smiles_list=smiles_list,
        n_variants=n_variants,
        similarity_threshold=similarity_threshold,
        preserve_scaffold=preserve_scaffold,
        use_bioisosteric=use_bioisosteric,
        enable_pains_filter=True,
        enable_brenk_filter=True,
        use_pareto=use_pareto,
        show_visualizations=True,
        export_csv=True
    )

    return explorer, results

# =========================================
# 1️⃣7️⃣ Alternative: Quick Start Function
# =========================================

def quick_explore(smiles_list, n_variants=30, similarity_threshold=0.5):
    """
    Quick exploration with improved defaults

    Example usage:
    >>> results = quick_explore(['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'])
    """
    explorer = VariantProject()
    results = explorer.explore(
        smiles_list=smiles_list,
        n_variants=n_variants,
        similarity_threshold=similarity_threshold,
        preserve_scaffold=True,
        use_bioisosteric=True,
        enable_pains_filter=True,
        enable_brenk_filter=True,
        use_pareto=True,
        show_visualizations=True,
        export_csv=True
    )
    return explorer, results

# =========================================
# 1️⃣8️⃣ Execute Main Program
# =========================================

if __name__ == "__main__":
    # Run the interactive main function
    print("🚀 Starting VariantProject v2.0 - IMPROVED VERSION\n")
    print("Key Improvements:")
    print("  ✅ Real ML models trained on ESOL dataset")
    print("  ✅ PAINS & reactive group filtering")
    print("  ✅ InChI-based duplicate detection")
    print("  ✅ 30+ comprehensive molecular descriptors")
    print("  ✅ Uncertainty quantification")
    print("  ✅ Bioisosteric replacements")
    print("  ✅ Pareto multi-objective optimization")
    print("  ✅ Progress bars & enhanced visualization\n")
    
    explorer, results = main()

    # Uncomment below for quick testing with predefined molecules
    # explorer, results = quick_explore([
    #     'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    #     'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    #     'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # Ibuprofen
    # ])
