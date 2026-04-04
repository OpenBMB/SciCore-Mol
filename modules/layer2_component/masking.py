from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


ROLE_SOLVENT = "SOLVENT"
ROLE_CATALYST = "CATALYST"
ROLE_REAGENT = "REAGENT"
ROLE_REACTANT = "REACTANT"
ROLE_PRODUCT = "PRODUCT"


@dataclass(frozen=True)
class MaskingConfig:
 seed: int = 42
 p_forward: float = 0.4
 p_retro: float = 0.3
 p_condition: float = 0.2
 p_random: float = 0.1
 p_yield_full: float = 0.0 # task2a: full info, predict yield
 p_yield_with_product: float = 0.0 # task2b: mask product + predict yield

 # retro taskrandom mask INPUT token countrange
 retro_min_mask: int = 1
 retro_max_mask: int = 2

 # random taskrandom mask token 
 random_token_ratio: float = 0.15

 # amount mask valuemodelvector/parameterreplace
 mask_amt_value: float = 0.0

 # whether forward/retro/random mask amount
 mask_amount_in_forward: bool = True
 mask_amount_in_retro: bool = True
 mask_amount_in_random: bool = True


@dataclass(frozen=True)
class EvalMaskingConfig:
 """
 evaluation MaskingConfig LLM template taskconsistent
 - task1 (task1_mask_product): Forward Synthesis - maskproductpredictionproduct
 - task2a (task2a_predict_yield_full): predictionyieldyielddataaccording toreaction_idrandom
 - task2b (task2b_predict_product_and_yield): maskproductpredictionproductyieldyielddataaccording toreaction_idrandom
 - task3 (task3_mask_role): maskreactioncondition//
 - task4 (task4_mask_reactant): Retrosynthesis - maskreaction
 
 NOTEtask2a2busereaction_idhashvaluedecideLLM templateconsistent
 """
 seed: int = 42
 # evaluationtaskLLM templategenerateconsistent
 # yielddatatask2random2a2b
 p_task1_mask_product: float = 0.25 # Forward Synthesis: maskproduct
 p_task4_mask_reactant: float = 0.25 # Retrosynthesis: maskreaction
 p_task3_mask_role: float = 0.2 # maskreactioncondition
 p_task2_yield: float = 0.3 # yieldtasktotal2a2b

 # retro taskrandom mask INPUT token countrange
 retro_min_mask: int = 1
 retro_max_mask: int = 2

 # amount mask valuemodelvector/parameterreplace
 mask_amt_value: float = 0.0

 # whether forward/retro mask amount
 mask_amount_in_forward: bool = True
 mask_amount_in_retro: bool = True


def _stable_rng(seed: int, reaction_id: str, extra: str) -> random.Random:
 h = hashlib.md5(f"{seed}\t{reaction_id}\t{extra}".encode("utf-8")).digest()
 rng_seed = int.from_bytes(h[:8], byteorder="little", signed=False)
 return random.Random(rng_seed)


def _choose_task(rng: random.Random, cfg: MaskingConfig, has_yield: bool = False) -> str:
 ps = [cfg.p_forward, cfg.p_retro, cfg.p_condition, cfg.p_random,
 cfg.p_yield_full, cfg.p_yield_with_product]
 s = sum(ps)
 if s <= 0:
 return "forward"
 r = rng.random() * s
 cum = 0.0
 tasks = ["forward", "retro", "condition", "random",
 "yield_full", "yield_with_product"]
 for p, task in zip(ps, tasks):
 cum += p
 if r < cum:
 # yield tasks require has_yield; fall back to forward if not available
 if task in ("yield_full", "yield_with_product") and not has_yield:
 return "forward"
 return task
 return "random"


def _choose_eval_task(
 rng: random.Random, 
 cfg: EvalMaskingConfig, 
 reaction_id: str,
 has_yield: bool
) -> str:
 """
 evaluationtaskchoose LLM template taskconsistent
 
 task
 1. choosetasktypetask1/task3/task4/yieldtotaltask
 2. ifchooseyieldtaskyielddataaccording toreaction_idhashdecidetask2atask2b
 """
 ps = [cfg.p_task1_mask_product, cfg.p_task4_mask_reactant, cfg.p_task3_mask_role, cfg.p_task2_yield]
 s = sum(ps)
 if s <= 0:
 return "task1_mask_product"
 
 r = rng.random() * s
 if r < ps[0]:
 return "task1_mask_product" # Forward Synthesis
 r -= ps[0]
 if r < ps[1]:
 return "task4_mask_reactant" # Retrosynthesis
 r -= ps[1]
 if r < ps[2]:
 return "task3_mask_role" # mask reaction conditions
 # otherwise yield task
 if has_yield:
 # use reaction_id hash decide task2a task2b LLM template consistent
 import hashlib
 seed_hash = int(hashlib.md5(reaction_id.encode()).hexdigest(), 16)
 if (seed_hash % 2) == 0:
 return "task2a_predict_yield_full"
 else:
 return "task2b_predict_product_and_yield"
 else:
 # yield data task1
 return "task1_mask_product"


def _mask_amount_fields(token: dict[str, Any], cfg: MaskingConfig | EvalMaskingConfig, targets: dict[str, Any]) -> None:
 for ch in ("moles", "mass", "volume"):
 key_v = f"amt_{ch}_log"
 key_m = f"amt_{ch}_mask"
 key_pm = f"amt_{ch}_pred_mask"

 if int(token.get(key_m, 0)) != 1:
 token[key_pm] = 0
 continue

 true_v = token.get(key_v)
 if true_v is None:
 token[key_pm] = 0
 continue

 token[key_pm] = 1
 targets.setdefault("amount", []).append((token["_idx"], ch, true_v))
 token[key_v] = float(cfg.mask_amt_value)


def apply_dynamic_mask(
 example: dict[str, Any], 
 cfg: MaskingConfig | EvalMaskingConfig, 
 *, 
 view_id: int = 0
) -> dict[str, Any]:
 """
 input Layer2 reaction record layer2_*.jsonl dict
 outputmasked viewcontains pred_mask targetsfortraining

 
 - function“generate mask ” tensor padding
 - `emb` mask Nonerecord `emb_pred_mask=1`value targets
 """
 reaction_id = str(example.get("reaction_id", ""))
 rng = _stable_rng(cfg.seed, reaction_id, f"view:{view_id}")

 # originaldata
 ex = copy.deepcopy(example)
 tokens: List[dict[str, Any]] = ex.get("tokens") or []
 if not isinstance(tokens, list):
 raise ValueError("example.tokens must list")

 # tokens index targets record
 for i, t in enumerate(tokens):
 if not isinstance(t, dict):
 raise ValueError("tokens[*] must dict")
 t["_idx"] = i
 t["emb_pred_mask"] = 0
 t["amt_moles_pred_mask"] = 0
 t["amt_mass_pred_mask"] = 0
 t["amt_volume_pred_mask"] = 0

 # optionalshuffle
 rng.shuffle(tokens)
 # shuffle idxtargets new
 for i, t in enumerate(tokens):
 t["_idx"] = i

 # according to config typechoosetask
 has_yield = (ex.get("yield_reg") is not None) or (ex.get("yield_bin") is not None)
 if isinstance(cfg, EvalMaskingConfig):
 task = _choose_eval_task(rng, cfg, reaction_id, has_yield)
 else:
 task = _choose_task(rng, cfg, has_yield=has_yield)
 targets: Dict[str, Any] = {"task": task, "embedding": [], "amount": []}

 # helpermask embedding
 def _mask_emb(t: dict[str, Any]) -> None:
 emb = t.get("emb")
 if emb is None:
 return
 t["emb_pred_mask"] = 1
 targets["embedding"].append((t["_idx"], emb))
 t["emb"] = None

 # choose token 
 input_tokens = [t for t in tokens if t.get("token_type") == "INPUT"]
 outcome_tokens = [t for t in tokens if t.get("token_type") == "OUTCOME"]

 # taskmappingevaltask -> trainingtask
 # task1_mask_product (Forward Synthesis): maskproduct
 # task4_mask_reactant (Retrosynthesis): maskreaction
 # task3_mask_role: maskreactioncondition
 # task2a_predict_yield_full: predictionyieldmasktoken
 # task2b_predict_product_and_yield: maskproductpredictionproductyield
 
 if task == "task1_mask_product" or task == "forward":
 # task1_mask_product: Forward Synthesis - maskproduct
 for t in outcome_tokens:
 _mask_emb(t)
 if cfg.mask_amount_in_forward:
 _mask_amount_fields(t, cfg, targets)
 ex["yield_pred_mask"] = 0 # task1predictionyield

 elif task == "task4_mask_reactant" or task == "retro":
 # task4_mask_reactant: Retrosynthesis - maskreaction
 k_min = max(1, cfg.retro_min_mask)
 k_max = max(k_min, cfg.retro_max_mask)
 k = min(len(input_tokens), rng.randint(k_min, k_max))
 for t in rng.sample(input_tokens, k=k):
 _mask_emb(t)
 if cfg.mask_amount_in_retro:
 _mask_amount_fields(t, cfg, targets)
 ex["yield_pred_mask"] = 0 # task4predictionyield

 elif task == "task3_mask_role" or task == "condition":
 # task3_mask_role: maskreactioncondition//
 for t in input_tokens:
 role = t.get("reaction_role")
 if role in {ROLE_SOLVENT, ROLE_CATALYST, ROLE_REAGENT}:
 _mask_emb(t)
 ex["yield_pred_mask"] = 0 # task3predictionyield

 elif task == "task2a_predict_yield_full" or task == "yield_full":
 # task2a: predictionyieldmasktoken
 ex["yield_pred_mask"] = 1
 
 elif task == "task2b_predict_product_and_yield" or task == "yield_with_product":
 # task2b: maskproductpredictionproductyield
 for t in outcome_tokens:
 _mask_emb(t)
 if cfg.mask_amount_in_forward:
 _mask_amount_fields(t, cfg, targets)
 ex["yield_pred_mask"] = 1
 
 else: # random (traininguse)
 if not isinstance(cfg, EvalMaskingConfig):
 # mask random token embedding + optional amount
 n = len(tokens)
 k = max(1, int(round(n * cfg.random_token_ratio)))
 for t in rng.sample(tokens, k=min(k, n)):
 _mask_emb(t)
 if cfg.mask_amount_in_random:
 _mask_amount_fields(t, cfg, targets)
 ex["yield_pred_mask"] = 0

 # internalmodelinput
 for t in tokens:
 t.pop("_idx", None)

 ex["tokens"] = tokens
 ex["targets"] = targets
 return ex

