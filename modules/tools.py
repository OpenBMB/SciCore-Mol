# tools/chem_tools_combined.py
# -*- coding: utf-8 -*-
"""
merge
- 1extractchemdataextractormapping SMILESSQLiteandrecognition SMILES+RDKit
- 2modelgenerate RDKit Mol listconvert SMILES

dependencyoptional
- RDKitfor SMILES canonical 
- chemdataextractorfor
- SQLite compounds/synonyms data
"""

from typing import List, Dict, Optional, Tuple, Set
import re
import sqlite3

from rdkit import Chem
from rdkit.Chem import AllChem
_HAS_RDKIT = True


# Lazy import to avoid network issues during stanza initialization
cde = None
_HAS_CDE = False
def _ensure_cde():
 """Lazy import of chemdataextractor to avoid network issues"""
 global cde, _HAS_CDE
 if cde is None and not _HAS_CDE:
 try:
 # setenvironmentvariabledownloadif
 import os
 os.environ.setdefault('CDE_DISABLE_DOWNLOADS', '1')
 # stanza download
 os.environ.setdefault('STANZA_RESOURCES_DIR', '/tmp/stanza_resources')
 import chemdataextractor as cde_module
 cde = cde_module
 _HAS_CDE = True
 # successprintwarning
 except Exception as e:
 # processprintwarning
 if not _HAS_CDE:
 print(f"Warning: chemdataextractor import failed: {e}. Name-to-SMILES mapping will be disabled.")
 cde = None
 _HAS_CDE = True # mark
 return cde

# latencyinitializemodulei.e.execute
cde = None


# ======================================================================
# 1 → SMILES
# ======================================================================

# --- 1A. mappingSQLite ---

def create_high_freq_table(db_path: str) -> None:
 """
 create/high_freq_compounds(name TEXT PRIMARY KEY, smiles TEXT NOT NULL)
 """
 conn = sqlite3.connect(db_path)
 cur = conn.cursor()
 cur.execute(
 """
 CREATE TABLE IF NOT EXISTS high_freq_compounds (
 name TEXT PRIMARY KEY,
 smiles TEXT NOT NULL
 )
 """
 )
 conn.commit()
 conn.close()


def query_smiles_by_name(name: str, db_path: str) -> Optional[str]:
 """
 SMILES high_freq_compoundsotherwise compounds synonyms
 high_freq_compoundscache
 """
 name = (name or "").strip().lower()
 if not name:
 return None

 conn = sqlite3.connect(db_path)
 cur = conn.cursor()

 #
 cur.execute("SELECT smiles FROM high_freq_compounds WHERE name = ?", (name,))
 row = cur.fetchone()
 if row:
 conn.close()
 return row[0]

 # compounds match
 cur.execute("SELECT smiles FROM compounds WHERE name = ?", (name,))
 row = cur.fetchone()
 if row:
 cur.execute("INSERT OR REPLACE INTO high_freq_compounds(name, smiles) VALUES(?,?)", (name, row[0]))
 conn.commit()
 conn.close()
 return row[0]

 # synonyms match
 cur.execute(
 "SELECT c.smiles FROM compounds c JOIN synonyms s ON c.cid = s.cid WHERE s.synonym = ?",
 (name,)
 )
 row = cur.fetchone()
 if row:
 cur.execute("INSERT OR REPLACE INTO high_freq_compounds(name, smiles) VALUES(?,?)", (name, row[0]))
 conn.commit()
 conn.close()
 return row[0]

 conn.close()
 return None


def extract_entities_to_smiles(text: str, db_path: str) -> Dict[str, str]:
 """
 use chemdataextractor datamapping SMILES
 returns {: SMILES}contains SMILES 
 """
 out: Dict[str, str] = {}
 if not text:
 return out
 
 # Lazy import CDE
 _ensure_cde()
 if cde is None:
 return out

 try:
 doc = cde.Document.from_string(text)
 cems = getattr(doc, "cems", [])
 if not cems:
 return out

 seen: Set[str] = set()
 names: List[str] = []
 for cem in cems:
 t = cem.text.strip()
 if len(t) < 3:
 continue
 if t in seen:
 continue
 seen.add(t)
 names.append(t)

 for name in names:
 smi = query_smiles_by_name(name, db_path=db_path)
 if smi:
 out[name] = smi
 return out
 except Exception:
 return out


# --- 1B. recognition SMILES + optional RDKit ---

# SMILES allowsetsupports“.”
_SMILES_CHARS = r"A-Za-z0-9@\+\-\[\]\(\)\\\/=#%\."
_SMILES_CANDIDATE = re.compile(rf"([{_SMILES_CHARS}]+)")

#
# 1) “class” @ + - [ ] ( ) / \ = # %
# 2) “+”mode C1, N2
# 3) contains“”mode Cl, Brlength>=3
_SYMBOLY_RE = re.compile(r"[@\+\-\[\]\(\)\\\/=#%]")
_ELEMNUM_RE = re.compile(r"[A-Z][a-z]?\d")
_ELEMENT_RE = re.compile(r"(?:Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Sn|Ag|Zn|Cu|Fe|Mn|Co|Ni|Mo|Hf|Ta|Ti|Cr|Pt|Au|Hg|Pb|Bi|I|F|O|N|S|P|B|C)")


def _looks_like_smiles(token: str) -> bool:
 token = token.strip()
 if len(token) < 3:
 return False
 if _SYMBOLY_RE.search(token):
 return True
 if _ELEMNUM_RE.search(token):
 return True
 if _ELEMENT_RE.search(token):
 return True
 return False


def _canonical_if_valid_smiles(token: str) -> Optional[str]:
 # if not _HAS_RDKIT:
 return token if _looks_like_smiles(token) else None
 try:
 m = Chem.MolFromSmiles(token, sanitize=True)
 if m is None:
 return None
 return Chem.MolToSmiles(m, canonical=True)
 except Exception:
 return None


def find_smiles_in_text(text: str, max_hits: int = 16, unique: bool = True) -> List[str]:
 """
 original SMILES RDKit 
 returns canonical SMILES list
 """
 if not text:
 return []

 hits: List[str] = []
 seen: Set[str] = set()
 for cand in _SMILES_CANDIDATE.findall(text):
 token = cand.strip()
 if not _looks_like_smiles(token):
 continue
 smi = _canonical_if_valid_smiles(token)
 if not smi:
 continue
 if unique:
 if smi in seen:
 continue
 seen.add(smi)
 hits.append(smi)
 if len(hits) >= max_hits:
 break
 return hits

SENT_SPLIT_RE = re.compile(r'(?<=[!?\.])\s*|\n+') # 
def resolve_text_to_smiles(
 text: str,
 db_path: str,
 prefer_names: bool = True,
) -> Dict[str, object]:
 """
 → returns
 {
 'from_names': {name: smiles}, # chemdataextractor + DB
 'from_smiles': [smiles, ...], # match SMILES
 'union_smiles': [unique list] # merge SMILES list
 }
 """
 # 1) <mol> last
 parts = text.split("<mol>")
 focus_text = parts[-1].strip() if parts else text.strip()

 # 2) 
 sents = [s.strip() for s in SENT_SPLIT_RE.split(focus_text) if s.strip()]
 focus_sent = sents[-1] if sents else focus_text
 
 
 # 3) 
 from_names = extract_entities_to_smiles(focus_sent, db_path=db_path) if prefer_names else {}
 from_smiles = find_smiles_in_text(focus_sent) or []

 # 4) according tochoose“last”
 candidates = []

 # save (, smiles)
 for name, smi in from_names.items():
 for m in re.finditer(re.escape(name), focus_sent):
 candidates.append((m.start(), smi))

 # SMILES
 for smi in from_smiles:
 for m in re.finditer(re.escape(smi), focus_sent):
 candidates.append((m.start(), smi))

 if not candidates:
 return None

 # last start sort
 candidates.sort(key=lambda x: x[0])
 return candidates[-1][1]


def extract_and_convert_online(text: str, proxy: Optional[str] = None) -> Dict[str, str]:
 """
 NER -> SMILES
 returns {token: SMILES} find_smiles_in_text
 """
 hits = find_smiles_in_text(text)
 return {smi: smi for smi in hits}
# ======================================================================
# 2Mol(s) → SMILES
# ======================================================================

def mol_to_canonical_smiles(mol) -> Optional[str]:
 """
 RDKit Mol canonical SMILESfailreturns None
 """
 if not _HAS_RDKIT or mol is None:
 return None
 try:
 Chem.SanitizeMol(mol)
 mol = Chem.RemoveHs(mol)
 return Chem.MolToSmiles(mol, canonical=True)
 except Exception:
 return None


def best_smiles_from_generated_mols(
 gen_mols: List,
 pick_largest_fragment: bool = True
) -> Optional[str]:
 """
 modelgenerate RDKit Mol list“” SMILES
 - each mol → canonical smiles
 - smiles '.' 
 - returnsfirstsuccess smiles canonical otherwise None
 """
 if not gen_mols or not _HAS_RDKIT:
 return None

 def _largest_fragment_smiles(smi: str) -> str:
 if not pick_largest_fragment or "." not in smi:
 return smi
 frags = smi.split(".")
 scored = []
 for f in frags:
 m = Chem.MolFromSmiles(f)
 if m is not None:
 scored.append((m.GetNumAtoms(), f))
 if not scored:
 return frags[0]
 scored.sort(reverse=True)
 return scored[0][1]

 for m in gen_mols:
 smi = mol_to_canonical_smiles(m)
 if not smi:
 continue
 smi = _largest_fragment_smiles(smi)
 try:
 mm = Chem.MolFromSmiles(smi)
 if mm is None:
 continue
 return Chem.MolToSmiles(mm, canonical=True)
 except Exception:
 continue
 return None


# ======================================================================
# test
# ======================================================================

if __name__ == "__main__":
 # needsaccording to
 # - RDKitoptional
 # - chemdataextractoroptional
 # - SQLite datafile `compounds.db`contains tables: compounds(name, smiles), synonyms(cid, synonym)
 # ifdatamappingtestreturnsempty SMILES 
 proxy_url = "http://127.0.0.1:7899" # ifneedssetvariable
 DB_PATH = "compounds.db"
 try:
 create_high_freq_table(DB_PATH)
 except Exception:
 pass

 print("=== test1 —— parse+SMILES ===")
 samples = [
 # SMILES
 "We tested aspirin CC(C)CC1=CC=C(C=C1)C(C)C(=O)O, (CC(=O)OC1=CC=CC=C1C(=O)O) and ibuprofen in this experiment.",
 # SMILES
 "Mixture: CCO.CN was observed prior to reaction. <mol>",
 # needsDBmapping
 "",
 # embeddingSMILES
 "Chiral test: C[C@H](O)[C@@H](N)C(=O)O appeared in trace amounts.",
 #
 "It's just a plain sentence without chemicals."
 ]

 for i, text in enumerate(samples, 1):
 res = resolve_text_to_smiles(text, db_path=DB_PATH, prefer_names=True)
 print(f"\n[Case {i}] {text}")

 if isinstance(res, dict):
 print(" from_names :", res.get("from_names", []))
 print(" from_smiles:", res.get("from_smiles", []))
 print(" union :", res.get("union_smiles", []))
 elif isinstance(res, list):
 # ifreturnslistprinteach from_names/from_smiles
 for r in res:
 if isinstance(r, dict):
 print(" from_names :", r.get("from_names", []))
 print(" from_smiles:", r.get("from_smiles", []))
 print(" union :", r.get("union_smiles", []))
 else:
 # r string
 print(" from_names :", r)
 print(" from_smiles:", r)
 print(" union :", r)
 else:
 # string
 print(" from_names :", res)
 print(" from_smiles:", res)
 print(" union :", res)


"""
from tools.chem_tools_combined import resolve_text_to_smiles, create_high_freq_table

DB_PATH = "compounds.db"
create_high_freq_table(DB_PATH)

text = "We tested aspirin and ibuprofen. Also we saw CCC(C)Br and CCO.CN <mol>."
res = resolve_text_to_smiles(text, db_path=DB_PATH, prefer_names=True)
print(res["from_names"]) # {'aspirin': '...', 'ibuprofen': '...'} (dependencyDB)
print(res["from_smiles"]) # ['CCC(C)Br', 'CCO', 'CN'] (RDKit canonical )
print(res["union_smiles"]) # merge
"""

"""
from tools.chem_tools_combined import best_smiles_from_generated_mols

with torch.no_grad():
 gen_mols = diffusion.generate_mol_from_embedding(
 batch_size=len(emb1), embeddings=emb1, num_nodes_lig=None
 )

smi = best_smiles_from_generated_mols(gen_mols) # returns SMILES None
"""
