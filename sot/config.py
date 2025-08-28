"""
Configuration for Synthesis of Thought system.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for SoT system."""
    max_depth: int = 4
    max_branches_per_node: int = 3
    max_total_nodes: int = 50
    max_iterations: int = 10
    temperature: float = 0.7
    enable_verification: bool = True
    enable_pruning: bool = True
    min_score_threshold: float = 0.4
