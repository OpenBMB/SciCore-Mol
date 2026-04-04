# ner_online.py
import re
import requests
import logging
from typing import Optional, List, Dict
import os

# configlog
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# latency chemdataextractornetwork
_Document = None
_HAS_CDE = False

def _get_document_class():
 """latency Document classnetworkconnection"""
 global _Document, _HAS_CDE
 if _Document is None and not _HAS_CDE:
 try:
 # setenvironmentvariabledownloadif
 os.environ.setdefault('CDE_DISABLE_DOWNLOADS', '1')
 from chemdataextractor.doc import Document as _Document
 _HAS_CDE = True
 except Exception as e:
 logging.warning(f"Failed to import chemdataextractor: {e}. Name-to-SMILES mapping will be disabled.")
 _HAS_CDE = False
 return _Document

# ======= function =======
def clean_text(text: str) -> str:
 """mark"""
 text = re.sub(r'\{\{.*?\}\}', '', text)
 text = re.sub(r'\$\{.*?\}\}', '', text)
 return text.strip()

def preprocess_cem(cem: str) -> str:
 """process PubChem request"""
 cleaned = re.sub(r'\s*\([^)]*\)\s*', '', cem)
 cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '_', cleaned)
 cleaned = cleaned.strip().replace(' ', '+')
 cleaned = cleaned.strip('_+')
 return cleaned

def is_likely_smiles(s: str) -> bool:
 """checkstringwhethervalidSMILES"""
 smiles_chars = set("BCNOPSFIKLHRecnops1234567890@-=#$()[]+\\/%")
 return all(c in smiles_chars for c in s) and ' ' not in s and len(s) > 0

def safe_document(text: str) -> Optional:
 """parse ChemDataExtractor Document"""
 try:
 Document = _get_document_class()
 if Document is None:
 return None
 if not text or not text.strip():
 return None
 return Document(text)
 except Exception as e:
 logging.warning(f"Failed to parse text with ChemDataExtractor. Error: {e}")
 return None

def get_smiles_from_pubchem(name: str, proxy: Optional[str] = None) -> Optional[str]:
 """
 PubChem API get SMILESsupports SMILES input
 """
 headers = {"User-Agent": "Mozilla/5.0"}
 proxies = {'http': proxy, 'https': proxy} if proxy else None

 # --- Case 1: ifinput SMILES ---
 if is_likely_smiles(name):
 url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{name}/property/CanonicalSMILES/TXT"
 else:
 # --- Case 2: input ---
 preprocessed_name = preprocess_cem(name).lower()
 preprocessed_name = preprocessed_name.replace("sulphuric", "sulfuric") # 
 url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{preprocessed_name}/property/CanonicalSMILES/TXT"

 try:
 response = requests.get(url, headers=headers, proxies=proxies, timeout=8)
 response.raise_for_status()
 text = response.text.strip()
 # print(f"[DEBUG] PubChem returns: {text}") # debug
 if text and not text.startswith("Page not found") and is_likely_smiles(text):
 return text
 else:
 return None
 except requests.exceptions.RequestException as e:
 logging.error(f"PubChem API request failed for '{name}': {e}")
 return None

def extract_and_convert_online(text: str, proxy: Optional[str] = None) -> Dict[str, str]:
 """
 extractconvert SMILES
 CDE failuse SMILES detection fallback
 """
 results = {}
 cleaned_text = clean_text(text)

 # Step 1: ChemDataExtractor
 doc = safe_document(cleaned_text)
 cems: List[str] = []
 if doc:
 try:
 cems = list(set(
 cem.text.strip()
 for cem in getattr(doc, "cems", []) or []
 if cem.text and cem.text.strip()
 ))
 except Exception as e:
 logging.warning(f"Error extracting CEMS from document: {e}")

 # Step 2: fallback 
 if not cems:
 logging.info("CDErecognitionmatch")
 regex_candidates = re.findall(
 r'\b[A-Za-z][a-z]{1,}(?:ic acid|ate|ene|one|ol|ide|ium|ane|yne)?\b',
 cleaned_text
 )
 cems = list(set(regex_candidates))

 # Step 3: fallback detection SMILES
 if not cems and is_likely_smiles(cleaned_text):
 cems = [cleaned_text]

 # Step 4: PubChem
 for cem_name in cems:
 smiles = get_smiles_from_pubchem(cem_name, proxy)
 if smiles:
 results[cem_name] = smiles

 # print(f"[DEBUG] extract: {cems}")
 return results

# ======= exampleLLM callmodule =======
def handle_mol_token(llm_context: str, proxy: Optional[str] = None) -> str:
 """
 function LLM callrecognition
 supportsrecognition SMILES fallback
 """
 # 1. recognitionconvertSMILES
 smiles_map = extract_and_convert_online(llm_context, proxy)

 # 2. llm_context extractneedsconverte.g.last
 last_cem = ""
 last_idx = -1
 for cem_name in smiles_map:
 idx = llm_context.rfind(cem_name)
 if idx > last_idx:
 last_idx = idx
 last_cem = cem_name

 # 3. ifrecognitionsuccess SMILES
 if last_cem and last_cem in smiles_map:
 smiles = smiles_map[last_cem]
 print(f"✅ success '{last_cem}' convert SMILES: '{smiles}'")
 return smiles # returns SMILES string

 # 4. fallback: check SMILES
 if is_likely_smiles(llm_context.strip()):
 print(f"⚡ detectioninput SMILES: '{llm_context.strip()}'")
 return llm_context.strip()

 # 5. otherwise Diffusion 
 print("❌ recognitionconvert Diffusion ")
 return "<mol_not_found>"

# ======= test =======
# if __name__ == "__main__":
# test_texts = [
# "A novel synthesis of Benzene was reported.",
# "The reaction uses Sulphuric acid.",
# "A mixture of ethyl acetate and isopropanol was used.",
# "The molecule C1=CC=CC=C1 has a unique structure."
# ]
 
# proxy_url = "http://127.0.0.1:7899" # ifneedssetvariable

# for text in test_texts:
# print("-" * 20)
# print(f"input: '{text}'")
 
# # LLMcall
# result_smiles_or_token = handle_mol_token(text, proxy=proxy_url)
# print(f"processresult: {result_smiles_or_token}")
