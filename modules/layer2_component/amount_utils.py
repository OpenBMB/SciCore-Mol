from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


# mol / g / L
_MOLES_TO_MOL = {
 "MOLE": 1.0,
 "MILLIMOLE": 1e-3,
 "MICROMOLE": 1e-6,
 "NANOMOLE": 1e-9,
}
_MASS_TO_G = {
 "GRAM": 1.0,
 "MILLIGRAM": 1e-3,
 "MICROGRAM": 1e-6,
}
_VOLUME_TO_L = {
 "LITER": 1.0,
 "MILLILITER": 1e-3,
 "MICROLITER": 1e-6,
}


@dataclass(frozen=True)
class AmountChannels:
 # log1p(value) + maskdatawhethervalue
 amt_moles_log: Optional[float]
 amt_moles_mask: int
 amt_mass_log: Optional[float]
 amt_mass_mask: int
 amt_volume_log: Optional[float]
 amt_volume_mask: int
 volume_includes_solutes: Optional[bool]


def _to_base_value(q: dict[str, Any], unit_map: dict[str, float]) -> Optional[float]:
 try:
 value = float(q.get("value"))
 except Exception:
 return None
 units = q.get("units")
 if not isinstance(units, str) or units not in unit_map:
 return None
 x = value * float(unit_map[units])
 if not math.isfinite(x) or x <= 0.0:
 return None
 return x


def amount_to_channels(amount: Any) -> AmountChannels:
 """
 extractstage amount dict feature
 - mass->moles each
 - unmeasured ignore
 - x>0 valuegenerate log1potherwisemask=0
 """
 if not isinstance(amount, dict):
 return AmountChannels(
 amt_moles_log=None,
 amt_moles_mask=0,
 amt_mass_log=None,
 amt_mass_mask=0,
 amt_volume_log=None,
 amt_volume_mask=0,
 volume_includes_solutes=None,
 )

 moles_x = None
 mass_x = None
 volume_x = None

 if isinstance(amount.get("moles"), dict):
 moles_x = _to_base_value(amount["moles"], _MOLES_TO_MOL)
 if isinstance(amount.get("mass"), dict):
 mass_x = _to_base_value(amount["mass"], _MASS_TO_G)
 if isinstance(amount.get("volume"), dict):
 volume_x = _to_base_value(amount["volume"], _VOLUME_TO_L)

 volume_includes_solutes = None
 if volume_x is not None and "volume_includes_solutes" in amount:
 try:
 volume_includes_solutes = bool(amount["volume_includes_solutes"])
 except Exception:
 volume_includes_solutes = None

 def _log1p_or_none(x: Optional[float]) -> tuple[Optional[float], int]:
 if x is None:
 return None, 0
 return math.log1p(x), 1

 moles_log, moles_mask = _log1p_or_none(moles_x)
 mass_log, mass_mask = _log1p_or_none(mass_x)
 volume_log, volume_mask = _log1p_or_none(volume_x)

 return AmountChannels(
 amt_moles_log=moles_log,
 amt_moles_mask=moles_mask,
 amt_mass_log=mass_log,
 amt_mass_mask=mass_mask,
 amt_volume_log=volume_log,
 amt_volume_mask=volume_mask,
 volume_includes_solutes=volume_includes_solutes,
 )


def build_amount_feature(
 moles: float = 1.0,
 mass: float = 0.0,
 volume: float = 0.0,
 data_mask: list[bool] = None,
 pred_mask: list[bool] = None,
 volume_includes_solutes: bool = False,
) -> list[float]:
 """
 build amount featurevector10 
 
 featureformat
 [moles_log, moles_data_mask, moles_pred_mask,
 mass_log, mass_data_mask, mass_pred_mask,
 vol_log, vol_data_mask, vol_pred_mask,
 volume_includes_solutes]
 
 Args:
 moles: mol
 mass: g
 volume: L
 data_mask: [moles_data_mask, mass_data_mask, vol_data_mask]datawhether
 pred_mask: [moles_pred_mask, mass_pred_mask, vol_pred_mask]whetherprediction
 volume_includes_solutes: whethercontains
 
 Returns:
 10 featurevectorlist[float]
 """
 import math
 
 if data_mask is None:
 data_mask = [False, False, False]
 if pred_mask is None:
 pred_mask = [False, False, False]
 
 # compute log1pifvalue 0 Noneuse 0.0
 moles_log = math.log1p(moles) if moles > 0.0 else 0.0
 mass_log = math.log1p(mass) if mass > 0.0 else 0.0
 vol_log = math.log1p(volume) if volume > 0.0 else 0.0
 
 # convert float
 moles_data_mask = 1.0 if data_mask[0] else 0.0
 mass_data_mask = 1.0 if data_mask[1] else 0.0
 vol_data_mask = 1.0 if data_mask[2] else 0.0
 
 moles_pred_mask = 1.0 if pred_mask[0] else 0.0
 mass_pred_mask = 1.0 if pred_mask[1] else 0.0
 vol_pred_mask = 1.0 if pred_mask[2] else 0.0
 
 vis_f = 1.0 if volume_includes_solutes else 0.0
 
 return [
 moles_log,
 moles_data_mask,
 moles_pred_mask,
 mass_log,
 mass_data_mask,
 mass_pred_mask,
 vol_log,
 vol_data_mask,
 vol_pred_mask,
 vis_f,
 ]

