import json
from pathlib import Path
from typing import Dict, Union, List, Tuple
import logging

from collections import defaultdict

import numpy as np

from ase import data as ase_data

from pymatgen.core.structure import Molecule

import torch

from torch_geometric.data import InMemoryDataset

from minimal_basis.data import DataPoint
from minimal_basis.utils import separate_graph, sn2_graph, sn2_positions

logger = logging.getLogger(__name__)


class ConnectivityDataset(InMemoryDataset):
    def __init__(self):
        raise NotImplementedError
