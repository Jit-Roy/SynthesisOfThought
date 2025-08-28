"""
Synthesis of Thought (SoT) package.
A framework for multi-path reasoning synthesis rather than single-path selection.
"""

from .nodes import Node, NodeStatus
from .frontier import (
    Frontier, 
    FrontierPolicy, 
    DepthFirstPolicy, 
    BreadthFirstPolicy,
    SmartBranchingPolicy,
    BranchingDecisionFramework
)
from .llm import LLM

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
]
