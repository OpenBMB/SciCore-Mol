from __future__ import annotations

from typing import Optional


def canonicalize_smiles(smiles: str) -> Optional[str]:
 """
 use RDKit SMILEScanonical + isomeric
 failreturns None
 """
 smiles = (smiles or "").strip()
 if not smiles:
 return None
 try:
 from rdkit import Chem

 mol = Chem.MolFromSmiles(smiles)
 if mol is None:
 return None
 return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
 except Exception:
 return None

