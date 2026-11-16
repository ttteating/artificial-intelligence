#!/usr/bin/env python3
"""Baseline runner for contract compliance multi-hop reasoning.

Conditions:
  B0: BFS (pure, no evaluator)
  B1: BFS + rule engine mask
  M0: MCTS (uniform prior, random rollout)
  M1: MCTS (LLM prior + classifier rollout scoring)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project setup
sys.path.insert(0, r"d:/program/artificial intelligence")

from state.core import State, Action, SearchResult, UNDETERMINED
from search.bfs import bfs
from search.mcts import mcts
from search.common import aux_fields
from data.ace_loader import load_split
from data.clause_graph import load_graph

from heuristic import mask as rule_mask, risk  # rule engine interface

# LLM prior integration for M1 condition
from llm.adapter import build_evaluator, LLMClient as _LLMClient
from llm.llm_client import LLMClient as _LLMClientFull

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
"""Single-condition experiment entry.

    python -m experiments.run_baselines --condition B0 --split dev --limit 10
    python -m experiments.run_baselines --condition A1 --split dev

Product: results/rows_{condition}_{split}.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path








