"""
Synthesis of Thought (SoT) package.
A framework for multi-path reasoning synthesis rather than single-path selection.
"""

from .nodes import Node, NodeStatus
from .frontier import (
    Frontier, 
    FrontierPolicy, 
    DepthFirstPolicy, 
    BreadthFirstPolicy
)
from .branching import (
    BranchingDecisionFramework,
    SmartBranchingPolicy
)
from .llm import LLM
from .config import Config
from .controller import Controller, create_root_node, get_store_stats

__version__ = "1.0.0"
__author__ = "SoT Team"

__all__ = [
    # Core models
    'Node',
    'NodeStatus',
    
    # Frontier management
    'Frontier',
    'FrontierPolicy',
    'DepthFirstPolicy',
    'BreadthFirstPolicy',
    'SmartBranchingPolicy',
    'BranchingDecisionFramework',
    
    # LLM interface
    'LLM',
    
    # Configuration and Controller
    'Config',
    'Controller',
    'create_root_node',
    'get_store_stats',
]
