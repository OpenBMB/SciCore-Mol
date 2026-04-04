from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# pathuse
from .io_utils import iter_jsonl, open_text
from .masking import MaskingConfig, apply_dynamic_mask

try:
 from tqdm import tqdm
except ImportError:
 tqdm = None

try:
 from tqdm import tqdm
except ImportError:
 tqdm = None

try:
 from torch.utils.data import IterableDataset, Dataset
 TORCH_AVAILABLE = True
except ImportError:
 # if torchdefineclass
 class IterableDataset: # type: ignore
 pass

 class Dataset: # type: ignore
 pass

 TORCH_AVAILABLE = False


class Layer2JsonlIterable(IterableDataset):
 """
 min Iterable Dataset Python
 - supportsread .jsonl / .jsonl.gz
 - optionalgeneratedynamic mask view
 """

 def __init__(
 self,
 path: str | Path,
 *,
 masking: bool = True,
 masking_cfg: Optional[MaskingConfig] = None,
 filter_has_yield: Optional[bool] = None,
 ) -> None:
 """
 Args:
 path: JSONLfilepath
 masking: whetherdynamicmasking
 masking_cfg: Maskingconfig
 filter_has_yield: ifTruereturnshas_yield=TruedataifFalsereturnshas_yield=FalsedataifNonefilter
 """
 self.path = str(path)
 self.masking = bool(masking)
 self.masking_cfg = masking_cfg or MaskingConfig()
 self.filter_has_yield = filter_has_yield

 def __iter__(self) -> Iterator[dict[str, Any]]:
 # supports DDPeachprocessprocessdata
 try:
 import torch.distributed as dist
 if dist.is_initialized():
 rank = dist.get_rank()
 world_size = dist.get_world_size()
 else:
 rank = 0
 world_size = 1
 except Exception:
 rank = 0
 world_size = 1
 
 view_id = 0
 for idx, ex in enumerate(iter_jsonl(self.path)):
 # DDP: eachprocessprocess rank data
 if idx % world_size != rank:
 continue
 
 # has_yieldfilter
 if self.filter_has_yield is not None:
 has_yield = ex.get("has_yield", False)
 if has_yield != self.filter_has_yield:
 continue
 
 if not self.masking:
 yield ex
 continue
 view_id += 1
 yield apply_dynamic_mask(ex, self.masking_cfg, view_id=view_id)


def _build_offsets_with_filter(path: str | Path, filter_has_yield: bool) -> List[int]:
 """
 .jsonl filebuild byte offset indexhas_yieldfiltersupports .gz
 """
 p = Path(path)
 if p.suffix == ".gz":
 raise ValueError("Indexed modesupports .gz Iterable decompress")
 
 is_main = True
 try:
 import torch.distributed as dist
 if dist.is_initialized():
 is_main = (dist.get_rank() == 0)
 except Exception:
 pass
 
 if is_main:
 file_size_gb = p.stat().st_size / 1024 / 1024 / 1024
 print(f"[INFO] buildindexfilterhas_yield={filter_has_yield}: {p.name} (filesize: {file_size_gb:.2f} GB)...")
 
 offsets: List[int] = []
 off = 0
 import json
 
 with p.open("rb") as f:
 if tqdm is not None and is_main:
 iterator = tqdm(f, desc="buildindexfilter", unit="", unit_scale=True)
 else:
 iterator = f
 
 for line in iterator:
 try:
 ex = json.loads(line.decode("utf-8").strip())
 has_yield = ex.get("has_yield", False)
 if has_yield == filter_has_yield:
 offsets.append(off)
 except Exception:
 pass # skipinvalid
 off += len(line)
 
 if is_main:
 print(f"[INFO] indexbuildcomplete: {len(offsets):,} filterhas_yield={filter_has_yield}")
 
 return offsets


def _build_offsets(path: str | Path) -> List[int]:
 """
 .jsonl filebuild byte offset indexsupports .gz
 """
 p = Path(path)
 if p.suffix == ".gz":
 raise ValueError("Indexed modesupports .gz Iterable decompress")

 # checkwhetherprocessDDPeachprocessprint
 is_main = True
 try:
 import torch.distributed as dist
 if dist.is_initialized():
 is_main = (dist.get_rank() == 0)
 except Exception:
 pass
 
 if is_main:
 file_size_gb = p.stat().st_size / 1024 / 1024 / 1024
 print(f"[INFO] buildindex: {p.name} (filesize: {file_size_gb:.2f} GB)...")
 
 offsets: List[int] = []
 off = 0
 
 with p.open("rb") as f:
 # use tqdm progress
 if tqdm is not None and is_main:
 iterator = tqdm(f, desc="buildindex", unit="", unit_scale=True)
 else:
 iterator = f
 
 for line in iterator:
 offsets.append(off)
 off += len(line)
 
 if is_main:
 print(f"[INFO] indexbuildcomplete: {len(offsets):,} ")
 
 return offsets


class Layer2JsonlIndexed(Dataset):
 """
 random Dataset .jsonl 
 description
 - forneeds shuffle training
 - supports .gzexternaldecompress IterableDataset + shuffle
 """

 def __init__(
 self,
 path: str | Path,
 *,
 masking: bool = True,
 masking_cfg: Optional[MaskingConfig] = None,
 filter_has_yield: Optional[bool] = None,
 ) -> None:
 """
 Args:
 path: JSONLfilepath
 masking: whetherdynamicmasking
 masking_cfg: Maskingconfig
 filter_has_yield: ifTruereturnshas_yield=TruedataifFalsereturnshas_yield=FalsedataifNonefilter
 """
 self.path = str(path)
 self.masking = bool(masking)
 self.masking_cfg = masking_cfg or MaskingConfig()
 self.filter_has_yield = filter_has_yield
 # iffilterconditionneedsfilebuildvalidindex
 if self.filter_has_yield is not None:
 self._offsets = _build_offsets_with_filter(self.path, filter_has_yield=self.filter_has_yield)
 else:
 self._offsets = _build_offsets(self.path)

 def __len__(self) -> int:
 return len(self._offsets)

 def __getitem__(self, idx: int) -> dict[str, Any]:
 import json

 off = self._offsets[idx]
 try:
 with open(self.path, "rb") as f:
 f.seek(off)
 line = f.readline().decode("utf-8").strip()
 ex = json.loads(line)
 except Exception as e:
 raise RuntimeError(f"readdatafail (idx={idx}, offset={off}): {e}")
 
 # has_yieldfilterIndexedmodebuildindexfilterno need tofiltercheck
 if self.filter_has_yield is not None:
 has_yield = ex.get("has_yield", False)
 if has_yield != self.filter_has_yield:
 # shouldindexfiltercheck
 raise RuntimeError(f"datafilterconditionmatch: has_yield={has_yield}, filter={self.filter_has_yield}")
 
 if not self.masking:
 return ex
 return apply_dynamic_mask(ex, self.masking_cfg, view_id=idx)

