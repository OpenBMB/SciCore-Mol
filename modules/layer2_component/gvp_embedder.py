from __future__ import annotations

import os
import pickle
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set


# GPU inferencethread/
_ENCODER_LOCK = threading.Lock()


def _ensure_import_path(gvp_root: Path) -> None:
 """
 gvp-gnn directory sys.path `import modules.gnn` 
 """
 gvp_root = gvp_root.resolve()
 if str(gvp_root) not in sys.path:
 sys.path.insert(0, str(gvp_root))


def build_gvp_encoder(device: str, gvp_root: Path, gnn_ckpt: Optional[str]) -> object:
 """
 buildload GVPEncoderoutputdimension 256
 """
 _ensure_import_path(gvp_root)

 try:
 import torch
 except Exception as e:
 raise RuntimeError(" torchinstall torch/torch_geometric run embedding") from e

 try:
 from modules.gnn import GVPEncoder # type: ignore
 except Exception as e:
 raise ImportError(
 " modules.gnn.GVPEncodercheck gvp_root whether `gvp-gnn` directory"
 "anddependencytorch_geometric/torch_scatter/rdkitwhether"
 ) from e

 gvp_cfg = {
 "node_dims": (10, 1),
 "edge_dims": (1, 1),
 "hidden_scalar_dim": 256,
 "hidden_vector_dim": 16,
 "output_dim": 256,
 "num_layers": 4,
 }
 dev = torch.device(device)
 encoder = GVPEncoder(**gvp_cfg).to(dev)
 encoder.eval()

 if not gnn_ckpt:
 raise ValueError(" --gnn-ckptallowuserandominitialize GVPEncoder generate embedding")

 if not os.path.isfile(gnn_ckpt):
 raise FileNotFoundError(f"GVP checkpoint : {gnn_ckpt}")

 sd = torch.load(gnn_ckpt, map_location="cpu")
 state = sd.get("model_state_dict", sd)
 state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
 encoder.load_state_dict(state, strict=False)
 return encoder


def _l2_normalize(vec: List[float]) -> Optional[List[float]]:
 s = 0.0
 for v in vec:
 s += float(v) * float(v)
 if s <= 0.0:
 return None
 inv = (s ** 0.5)
 if inv <= 0.0:
 return None
 return [float(v) / inv for v in vec]


def smiles_to_embedding(smiles: str, encoder: object, device: str) -> Optional[List[float]]:
 """
 SMILES encode embeddingList[float]
 failreturns None
 """
 import torch # latency torch environmentload cache

 try:
 if device.startswith("cuda"):
 with _ENCODER_LOCK:
 emb = encoder.forward_from_smiles(smiles)
 else:
 emb = encoder.forward_from_smiles(smiles)

 if isinstance(emb, torch.Tensor):
 emb_list = emb.squeeze().detach().to("cpu").float().numpy().tolist()
 # vectorfailGVPEncoder parsefailreturns 0
 emb_list = _l2_normalize(emb_list)
 return emb_list
 return None
 except Exception:
 return None


def embed_unique_smiles(
 unique_smiles: Set[str],
 encoder: object,
 device: str,
 max_workers: int,
) -> Dict[str, Optional[List[float]]]:
 """
 parallel SMILES encodereturns {smiles: embedding or None}
 """
 import concurrent.futures as futures

 cache: Dict[str, Optional[List[float]]] = {}
 if not unique_smiles:
 return cache

 try:
 from tqdm import tqdm
 use_tqdm = True
 except ImportError:
 use_tqdm = False

 def _task(smi: str):
 return smi, smiles_to_embedding(smi, encoder, device)

 with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
 fs = [ex.submit(_task, smi) for smi in unique_smiles]
 iterator = futures.as_completed(fs)
 if use_tqdm:
 iterator = tqdm(iterator, total=len(fs), desc="Embedding SMILES", unit="mol")
 for f in iterator:
 smi, emb = f.result()
 cache[smi] = emb
 return cache


def load_smiles_cache(path: str | Path) -> Dict[str, List[float]]:
 p = Path(path)
 if not p.exists():
 return {}
 with p.open("rb") as f:
 obj = pickle.load(f)
 if not isinstance(obj, dict):
 raise ValueError(f"cache fileformatincorrect: {p}")
 out: Dict[str, List[float]] = {}
 for k, v in obj.items():
 if isinstance(k, str) and isinstance(v, list) and v:
 out[k] = [float(x) for x in v]
 return out


def save_smiles_cache(path: str | Path, cache: Dict[str, List[float]]) -> None:
 p = Path(path)
 p.parent.mkdir(parents=True, exist_ok=True)
 with p.open("wb") as f:
 pickle.dump(cache, f, protocol=4)
