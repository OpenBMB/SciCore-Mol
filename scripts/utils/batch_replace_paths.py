#!/usr/bin/env python3
"""
replacefilepath
outputallpathfileconfirm
"""
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# pathmappingrule (original_path, target_path)
PATH_MAPPINGS = [
 # NOTEreplacepathmatch
 ("${DATA_DIR:-/path/to/data}/Project/SciCore-Mol", "${SCICORE_ROOT:-/path/to/scicore-mol}"),
 ("${DATA_DIR:-/path/to/data}/Project/MSMLM", "${DATA_DIR:-/path/to/data}/MSMLM"),
 ("${DATA_DIR:-/path/to/data}/Layer2", "${SCICORE_ROOT:-/path/to/scicore-mol}/Layer2"),
 ("${DATA_DIR:-/path/to/data}/cpt_data", "${DATA_DIR:-/path/to/data}/train_data/cpt_data"),
 ("${DATA_DIR:-/path/to/data}/SFT_DATA", "${DATA_DIR:-/path/to/data}/train_data/SFT_DATA"),
 ("${DATA_DIR:-/path/to/data}/base_model", "${DATA_DIR:-/path/to/data}/base_model"),
 ("${DATA_DIR:-/path/to/data}/checkpoint", "${CHECKPOINT_DIR:-/path/to/checkpoints}"),
 # valid_unreplaced_paths.json confirmmappingsuggested_path 
 ("${DATA_DIR:-/path/to/data}/checkpoint", "${CHECKPOINT_DIR:-/path/to/checkpoints}"),
 ("${DATA_DIR:-/path/to/data}/model", "${DATA_DIR:-/path/to/data}/base_model"),
 ("${DATA_DIR:-/path/to/data}/model", "${DATA_DIR:-/path/to/data}/base_model"),
 ("${EXT_DATA_DIR:-/path/to/external}", "${EXT_DATA_DIR:-/path/to/external}"),
 # Conda pathmapping
 ("${DATA_DIR:-/path/to/data}/miniconda3", "${CONDA_PREFIX:-/path/to/conda}"),
]

# needsprocessfile
INCLUDE_EXTENSIONS = {'.py', '.yaml', '.yml', '.sh', '.json', '.md', '.txt', '.jsonl'}

# directory
EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache', 'artifacts', 'scripts/utils'}

# filegenerateno need to
EXCLUDE_FILES = {
 'batch_replace_paths.py', # 
}

# createpathpathmappingdict
PATH_MAP_DICT = {old: new for old, new in PATH_MAPPINGS}


def should_process_file(file_path: Path) -> bool:
 """judgewhethershouldprocessfile"""
 # check
 if file_path.suffix not in INCLUDE_EXTENSIONS:
 return False
 
 # checkwhetherlist
 if file_path.name in EXCLUDE_FILES:
 return False
 
 # checkwhetherdirectory
 for part in file_path.parts:
 if part in EXCLUDE_DIRS:
 return False
 
 # scripts/utils directorydirectory
 try:
 parts = file_path.parts
 if 'scripts' in parts and 'utils' in parts:
 scripts_idx = parts.index('scripts')
 if scripts_idx + 1 < len(parts) and parts[scripts_idx + 1] == 'utils':
 return False
 except (ValueError, IndexError):
 pass
 
 return True


def extract_variable_name(line: str, path: str) -> str:
 """extractvariable"""
 # matchvariablevaluemode
 patterns = [
 r'(\w+)\s*[:=]\s*["\']?' + re.escape(path), # var = "path" var: "path"
 r'["\']?' + re.escape(path) + r'["\']?\s*[:=]\s*(\w+)', # "path" = var
 r'(\w+)\s*=\s*["\']?' + re.escape(path), # var = "path"
 r'--(\w+)\s+["\']?' + re.escape(path), # --arg "path"
 ]
 
 for pattern in patterns:
 match = re.search(pattern, line, re.IGNORECASE)
 if match:
 return match.group(1)
 
 return ""


def find_paths_in_line(line: str, line_num: int, file_path: Path) -> List[Dict[str, Any]]:
 """allneedsmappingpath"""
 found_paths = []
 
 for old_path, new_path in PATH_MAPPINGS:
 #
 escaped_old = re.escape(old_path)
 # matchpath / 
 pattern = escaped_old + r'(?=/|"|\'| |\n|$|,|\)|]|})'
 
 matches = list(re.finditer(pattern, line))
 for match in matches:
 start_pos = match.start()
 end_pos = match.end()
 
 # get30
 context_start = max(0, start_pos - 30)
 context_end = min(len(line), end_pos + 30)
 context = line[context_start:context_end]
 
 # extractvariable
 var_name = extract_variable_name(line, old_path)
 
 found_paths.append({
 "file_path": str(file_path),
 "line_number": line_num,
 "variable_name": var_name,
 "context": context.strip(),
 "original_path": old_path,
 "replaced_path": new_path,
 "position": (start_pos, end_pos)
 })
 
 return found_paths


def scan_file_for_paths(file_path: Path, root_dir: Path) -> Tuple[List[Dict], List[Dict]]:
 """fileallneedsreplaceno need toreplacepath"""
 replaced_paths = [] # canreplacepath
 all_old_paths = set() # allpathset
 
 try:
 with open(file_path, 'r', encoding='utf-8') as f:
 lines = f.readlines()
 except UnicodeDecodeError:
 try:
 with open(file_path, 'r', encoding='latin-1') as f:
 lines = f.readlines()
 except Exception as e:
 print(f"⚠️ readfile {file_path}: {e}")
 return [], []
 except Exception as e:
 print(f"⚠️ readfilefail {file_path}: {e}")
 return [], []
 
 #
 for line_num, line in enumerate(lines, 1):
 # allneedsmappingpath
 found = find_paths_in_line(line, line_num, file_path.relative_to(root_dir))
 replaced_paths.extend(found)
 
 # recordallpathforcheckreplace
 for old_path in PATH_MAP_DICT.keys():
 if old_path in line:
 all_old_paths.add(old_path)
 
 return replaced_paths, []


def find_unreplaced_paths(file_path: Path, root_dir: Path) -> List[Dict[str, Any]]:
 """find unmapped old paths in filescontains ${DATA_DIR:-/path/to/data} ${DATA_DIR:-/path/to/data} mappingrule"""
 unreplaced = []
 
 try:
 with open(file_path, 'r', encoding='utf-8') as f:
 lines = f.readlines()
 except Exception:
 try:
 with open(file_path, 'r', encoding='latin-1') as f:
 lines = f.readlines()
 except:
 return []
 except:
 return []
 
 # defineneedsdetectionpathprefix
 old_prefixes = ['${DATA_DIR:-/path/to/data}', '${DATA_DIR:-/path/to/data}']
 
 # getallmappingpathprefixfor
 mapped_prefixes = set()
 for old_path, _ in PATH_MAPPINGS:
 # extractprefixfirstdirectory
 parts = old_path.split('/')
 if len(parts) >= 4:
 mapped_prefixes.add('/'.join(parts[:4])) # e.g. ${DATA_DIR:-/path/to/data}
 
 # matchpathregex
 # match ${DATA_DIR:-/path/to/data}/... ${DATA_DIR:-/path/to/data}/... format paths
 path_pattern = re.compile(r'(/data[12]/<user>/[^"\' \n,\)\]}]+)')
 
 for line_num, line in enumerate(lines, 1):
 # allmatchpath
 matches = path_pattern.finditer(line)
 for match in matches:
 found_path = match.group(1)
 
 # checkpathwhetheralreadymappingrule
 is_mapped = False
 for old_path, new_path in PATH_MAPPINGS:
 # checkwhethermappingpathprefixmatch
 if found_path.startswith(old_path + '/') or found_path == old_path:
 is_mapped = True
 break
 
 # ifmappingrulereplacelist
 if not is_mapped:
 # get
 start_pos = match.start()
 end_pos = match.end()
 context_start = max(0, start_pos - 30)
 context_end = min(len(line), end_pos + 30)
 context = line[context_start:context_end].strip()
 
 # extractvariable
 var_name = extract_variable_name(line, found_path)
 
 # inferpath
 suggested_path = ""
 if found_path.startswith('${DATA_DIR:-/path/to/data}/checkpoint'):
 suggested_path = found_path.replace('${DATA_DIR:-/path/to/data}/checkpoint', '${CHECKPOINT_DIR:-/path/to/checkpoints}')
 elif found_path.startswith('${EXT_DATA_DIR:-/path/to/external}'):
 # pathneedsconfirm
 suggested_path = found_path.replace('${EXT_DATA_DIR:-/path/to/external}', '${EXT_DATA_DIR:-/path/to/external}')
 elif found_path.startswith('${DATA_DIR:-/path/to/data}/checkpoint'):
 suggested_path = found_path.replace('${DATA_DIR:-/path/to/data}/checkpoint', '${CHECKPOINT_DIR:-/path/to/checkpoints}')
 elif found_path.startswith('${DATA_DIR:-/path/to/data}/base_model'):
 suggested_path = found_path.replace('${DATA_DIR:-/path/to/data}/base_model', '${DATA_DIR:-/path/to/data}/base_model')
 else:
 # replace
 if '${DATA_DIR:-/path/to/data}' in found_path:
 suggested_path = found_path.replace('${DATA_DIR:-/path/to/data}', '${DATA_DIR:-/path/to/data}')
 elif '${DATA_DIR:-/path/to/data}' in found_path:
 suggested_path = found_path.replace('${DATA_DIR:-/path/to/data}', '${DATA_DIR:-/path/to/data}')
 
 unreplaced.append({
 "file_path": str(file_path.relative_to(root_dir)),
 "line_number": line_num,
 "variable_name": var_name,
 "context": context,
 "original_path": found_path,
 "suggested_path": suggested_path,
 "note": "pathmappingruleneedsconfirm"
 })
 
 return unreplaced


def process_directory(root_dir: Path, mappings: List[Tuple[str, str]], dry_run: bool = False):
 """processdirectoryallfile"""
 root_dir = Path(root_dir)
 
 print(f"🔍 directory: {root_dir}")
 print(f"📋 pathmappingrule:")
 for old, new in mappings:
 print(f" {old}")
 print(f" → {new}")
 print()
 
 all_replaced = []
 all_unreplaced = []
 total_files = 0
 
 # traverseallfile
 for file_path in root_dir.rglob('*'):
 if not file_path.is_file():
 continue
 
 if not should_process_file(file_path):
 continue
 
 total_files += 1
 
 # file
 replaced, _ = scan_file_for_paths(file_path, root_dir)
 unreplaced = find_unreplaced_paths(file_path, root_dir)
 
 all_replaced.extend(replaced)
 all_unreplaced.extend(unreplaced)
 
 if replaced:
 print(f"✅ {file_path.relative_to(root_dir)}: {len(replaced)} replacepath")
 
 print()
 print("=" * 60)
 print(f"📊 statistics:")
 print(f" file: {total_files}")
 print(f" replacepath: {len(all_replaced)} ")
 print(f" needsconfirmpath: {len(all_unreplaced)} ")
 print("=" * 60)
 
 # saveresult
 output_dir = root_dir / "scripts" / "utils" / "path_replacement_logs"
 output_dir.mkdir(parents=True, exist_ok=True)
 
 replaced_file = output_dir / "replaced_paths.json"
 unreplaced_file = output_dir / "unreplaced_paths.json"
 
 # savereplacepath
 with open(replaced_file, 'w', encoding='utf-8') as f:
 json.dump(all_replaced, f, indent=2, ensure_ascii=False)
 print(f"\n💾 savereplacepath: {replaced_file}")
 print(f" {len(all_replaced)} record")
 
 # savereplacepath
 with open(unreplaced_file, 'w', encoding='utf-8') as f:
 json.dump(all_unreplaced, f, indent=2, ensure_ascii=False)
 print(f"💾 saveconfirmpath: {unreplaced_file}")
 print(f" {len(all_unreplaced)} record")
 
 # ifdry_runmodeexecutereplace
 if not dry_run and all_replaced:
 print("\n🔄 startexecutereplace...")
 replace_paths_in_files(root_dir, all_replaced)
 
 return all_replaced, all_unreplaced


def replace_paths_in_files(root_dir: Path, replacements: List[Dict[str, Any]]):
 """according torecordexecutereplace"""
 # file
 files_to_modify = defaultdict(list)
 for rep in replacements:
 file_path = root_dir / rep["file_path"]
 files_to_modify[file_path].append(rep)
 
 total_replacements = 0
 
 for file_path, reps in files_to_modify.items():
 try:
 with open(file_path, 'r', encoding='utf-8') as f:
 content = f.read()
 except:
 try:
 with open(file_path, 'r', encoding='latin-1') as f:
 content = f.read()
 except Exception as e:
 print(f"⚠️ readfile {file_path}: {e}")
 continue
 
 original_content = content
 
 # allreplacereplace
 for rep in sorted(reps, key=lambda x: x["position"][0], reverse=True):
 old_path = rep["original_path"]
 new_path = rep["replaced_path"]
 
 # replace
 escaped_old = re.escape(old_path)
 pattern = escaped_old + r'(?=/|"|\'| |\n|$|,|\)|]|})'
 content = re.sub(pattern, new_path, content, count=1)
 
 # iffile
 if content != original_content:
 try:
 with open(file_path, 'w', encoding='utf-8') as f:
 f.write(content)
 total_replacements += len(reps)
 print(f"✅ {file_path.relative_to(root_dir)}: replace {len(reps)} ")
 except Exception as e:
 print(f"⚠️ writefilefail {file_path}: {e}")
 
 print(f"\n✅ totalreplace {total_replacements} path")


def main():
 import argparse
 
 parser = argparse.ArgumentParser(description='replacefilepath')
 parser.add_argument('--root', type=str, default='${SCICORE_ROOT:-/path/to/scicore-mol}',
 help='processdirectory')
 parser.add_argument('--dry-run', action='store_true', default=True,
 help='checkfiledefault')
 parser.add_argument('--execute', action='store_true',
 help='executereplace--dry-run')
 parser.add_argument('--mapping', type=str, nargs=2, action='append',
 help='definemappingrule: --mapping old_path new_path')
 
 args = parser.parse_args()
 
 root_dir = Path(args.root)
 if not root_dir.exists():
 print(f"❌ directory: {root_dir}")
 return
 
 # usedefinemappingdefaultmapping
 mappings = PATH_MAPPINGS
 if args.mapping:
 mappings = [(old, new) for old, new in args.mapping]
 
 dry_run = not args.execute
 
 if dry_run:
 print("🔍 modefile")
 print(" use --execute executereplace")
 print()
 else:
 print("⚠️ executemodefile")
 print()
 
 replaced, unreplaced = process_directory(root_dir, mappings, dry_run=dry_run)
 
 if dry_run and replaced:
 print("\n💡 use --execute executereplace")


if __name__ == '__main__':
 main()
