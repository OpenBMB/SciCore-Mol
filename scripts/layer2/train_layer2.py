#!/usr/bin/env python3
"""
training Layer2 model
: python scripts/layer2/train_layer2.py [--config scripts/layer2/layer2_train_config.yaml]
"""

import sys
import argparse
from pathlib import Path

# directorypath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.layer2_component.Layer2Trainer import main as trainer_main

if __name__ == "__main__":
 # defaultconfigpath
 DEFAULT_CONFIG = "${SCICORE_ROOT:-/path/to/scicore-mol}/scripts/layer2/layer2_train_config.yaml"
 
 parser = argparse.ArgumentParser(description="training Layer2 model")
 parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
 help=f"configfilepathdefault: {DEFAULT_CONFIG}")
 
 args = parser.parse_args()
 
 # sys.argv Layer2Trainer.main
 import sys
 sys.argv = ["Layer2Trainer", "--config", args.config]
 
 trainer_main()
