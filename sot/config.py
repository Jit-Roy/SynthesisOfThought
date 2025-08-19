"""
Configuration settings, thresholds, default parameters, and budgets for SoT.
Central configuration management for the Synthesis of Thought system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json


@dataclass
class SoTConfig:
    """
    Configuration class for Synthesis of Thought parameters.
    Contains all tunable parameters for the SoT reasoning process.
    """
    
    # Core expansion parameters
    max_depth: int = 8                          # Maximum reasoning depth
    max_branches_per_node: int = 3              # Default branching factor
    max_total_nodes: int = 100                  # Total node budget
    max_iterations: int = 50                    # Maximum reasoning iterations
    
    # Quality thresholds
    min_score_threshold: float = 0.3            # Minimum score for keeping nodes
    pruning_threshold: float = 0.2              # Threshold for aggressive pruning
    consensus_threshold: float = 0.6            # Threshold for consensus building
    
    # LLM parameters
    temperature: float = 0.7                    # Sampling temperature
    max_tokens_per_call: int = 200              # Token limit per LLM call
    max_retries: int = 3                        # Max retries for failed LLM calls
    
    # Processing options
    enable_pruning: bool = True                 # Enable low-score pruning
    enable_verification: bool = True            # Enable step verification
    enable_caching: bool = True                 # Enable LLM response caching
    enable_adaptive: bool = False               # Enable adaptive parameters
    
    # Synthesis parameters
    max_synthesis_paths: int = 5                # Number of final synthesized paths
    complementarity_threshold: float = 0.5     # Threshold for complementary merging
    consensus_weight: float = 0.3               # Weight for consensus paths
    
    # Resource budgets
    max_llm_calls: int = 200                   # Maximum LLM API calls
    max_verification_calls: int = 50            # Maximum verification calls
    time_budget_seconds: int = 300              # Maximum processing time
    
    # Scoring weights for path evaluation
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'confidence': 0.25,
        'completeness': 0.20,
        'coherence': 0.15,
        'novelty': 0.10,
        'evidence_support': 0.15,
        'consistency': 0.15
    })
    
    # Frontier policy configuration
    frontier_policy: str = "best_first"         # "best_first", "depth_first", "breadth_first", "ucb"
    ucb_exploration_weight: float = 1.414       # UCB exploration parameter
    
    # Adaptive parameters (when enable_adaptive=True)
    adaptation_frequency: int = 5               # Adapt every N iterations
    min_branching_factor: int = 1               # Minimum adaptive branching
    max_branching_factor: int = 5               # Maximum adaptive branching
    quality_history_size: int = 10              # Size of quality tracking window
    
    # Debug and logging
    log_level: str = "INFO"                     # Logging level
    save_intermediate: bool = False             # Save intermediate states
    debug_mode: bool = False                    # Enable debug outputs
    
    # Experimental features
    use_beam_search: bool = False               # Use beam search instead of tree search
    beam_width: int = 3                         # Beam width if using beam search
    diversity_bonus: float = 0.1                # Bonus for diverse reasoning paths
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Ensure positive values for core parameters
        assert self.max_depth > 0, "max_depth must be positive"
        assert self.max_branches_per_node > 0, "max_branches_per_node must be positive"
        assert self.max_total_nodes > 0, "max_total_nodes must be positive"
        assert self.max_iterations > 0, "max_iterations must be positive"
        
        # Ensure thresholds are in valid ranges
        assert 0.0 <= self.min_score_threshold <= 1.0, "min_score_threshold must be in [0,1]"
        assert 0.0 <= self.pruning_threshold <= 1.0, "pruning_threshold must be in [0,1]"
        assert 0.0 <= self.consensus_threshold <= 1.0, "consensus_threshold must be in [0,1]"
        assert 0.0 <= self.temperature <= 2.0, "temperature should be in [0,2]"
        
        # Ensure weights sum to approximately 1.0
        weight_sum = sum(self.scoring_weights.values())
        if not (0.9 <= weight_sum <= 1.1):
            # Auto-normalize if close
            if 0.5 <= weight_sum <= 1.5:
                normalized = {k: v/weight_sum for k, v in self.scoring_weights.items()}
                self.scoring_weights = normalized
            else:
                raise ValueError(f"Scoring weights sum to {weight_sum}, should be close to 1.0")
        
        # Validate adaptive parameters
        assert self.min_branching_factor <= self.max_branching_factor
        assert self.adaptation_frequency > 0
        assert self.quality_history_size > 0
        
        # Validate frontier policy
        valid_policies = ["best_first", "depth_first", "breadth_first", "ucb", "random"]
        assert self.frontier_policy in valid_policies, f"Invalid frontier_policy: {self.frontier_policy}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SoTConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SoTConfig':
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SoTConfig':
        """Load configuration from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
    
    def copy(self) -> 'SoTConfig':
        """Create a copy of the configuration."""
        return SoTConfig.from_dict(self.to_dict())
    
    def update(self, **kwargs) -> 'SoTConfig':
        """Create a new configuration with updated parameters."""
        new_data = self.to_dict()
        new_data.update(kwargs)
        return SoTConfig.from_dict(new_data)


# Predefined configurations for common use cases

class PresetConfigs:
    """Predefined configurations for different scenarios."""
    
    @staticmethod
    def fast_exploration() -> SoTConfig:
        """Configuration optimized for fast exploration."""
        return SoTConfig(
            max_depth=5,
            max_branches_per_node=2,
            max_total_nodes=30,
            max_iterations=20,
            min_score_threshold=0.4,
            temperature=0.8,
            enable_verification=False,
            max_synthesis_paths=3
        )
    
    @staticmethod
    def thorough_analysis() -> SoTConfig:
        """Configuration for thorough, high-quality analysis."""
        return SoTConfig(
            max_depth=10,
            max_branches_per_node=4,
            max_total_nodes=200,
            max_iterations=100,
            min_score_threshold=0.2,
            temperature=0.6,
            enable_verification=True,
            enable_pruning=True,
            max_synthesis_paths=7
        )
    
    @staticmethod
    def balanced() -> SoTConfig:
        """Balanced configuration for general use."""
        return SoTConfig(
            max_depth=8,
            max_branches_per_node=3,
            max_total_nodes=100,
            max_iterations=50,
            min_score_threshold=0.3,
            temperature=0.7,
            enable_verification=True,
            enable_pruning=True,
            max_synthesis_paths=5
        )
    
    @staticmethod
    def resource_constrained() -> SoTConfig:
        """Configuration for limited computational resources."""
        return SoTConfig(
            max_depth=4,
            max_branches_per_node=2,
            max_total_nodes=20,
            max_iterations=15,
            min_score_threshold=0.5,
            temperature=0.7,
            enable_verification=False,
            enable_pruning=True,
            max_synthesis_paths=3,
            max_llm_calls=50,
            time_budget_seconds=60
        )
    
    @staticmethod
    def creative_exploration() -> SoTConfig:
        """Configuration encouraging creative, diverse reasoning."""
        return SoTConfig(
            max_depth=6,
            max_branches_per_node=4,
            max_total_nodes=80,
            max_iterations=40,
            min_score_threshold=0.2,
            temperature=0.9,
            enable_verification=True,
            diversity_bonus=0.2,
            max_synthesis_paths=6,
            scoring_weights={
                'confidence': 0.20,
                'completeness': 0.15,
                'coherence': 0.15,
                'novelty': 0.25,  # Higher novelty weight
                'evidence_support': 0.10,
                'consistency': 0.15
            }
        )
    
    @staticmethod
    def adaptive_learning() -> SoTConfig:
        """Configuration with adaptive parameters enabled."""
        return SoTConfig(
            max_depth=8,
            max_branches_per_node=3,
            max_total_nodes=120,
            max_iterations=60,
            min_score_threshold=0.3,
            temperature=0.7,
            enable_adaptive=True,
            adaptation_frequency=3,
            min_branching_factor=1,
            max_branching_factor=5,
            quality_history_size=8,
            max_synthesis_paths=5
        )


# Configuration validation utilities

def validate_config_compatibility(config: SoTConfig) -> List[str]:
    """
    Validate configuration for potential issues and return warnings.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check for potentially problematic combinations
    if config.max_total_nodes > 500 and config.enable_verification:
        warnings.append(
            "Large node budget with verification enabled may be slow. "
            "Consider disabling verification or reducing node budget."
        )
    
    if config.temperature > 1.0 and config.min_score_threshold > 0.5:
        warnings.append(
            "High temperature with high score threshold may limit exploration. "
            "Consider lowering score threshold or temperature."
        )
    
    if config.max_branches_per_node > 5 and config.max_depth > 6:
        warnings.append(
            "High branching factor with deep search may cause exponential explosion. "
            "Monitor resource usage carefully."
        )
    
    if not config.enable_pruning and config.max_total_nodes > 100:
        warnings.append(
            "Large node budget without pruning may waste resources on low-quality paths."
        )
    
    if config.enable_adaptive and config.adaptation_frequency == 1:
        warnings.append(
            "Very frequent adaptation may cause instability. "
            "Consider increasing adaptation_frequency."
        )
    
    # Check budget consistency
    estimated_calls = config.max_total_nodes * config.max_branches_per_node * 2  # Generate + score
    if estimated_calls > config.max_llm_calls:
        warnings.append(
            f"LLM call budget ({config.max_llm_calls}) may be insufficient. "
            f"Estimated needs: {estimated_calls} calls."
        )
    
    return warnings


def optimize_config_for_budget(base_config: SoTConfig, 
                              max_time_seconds: Optional[int] = None,
                              max_llm_calls: Optional[int] = None) -> SoTConfig:
    """
    Optimize configuration to fit within specified budgets.
    
    Args:
        base_config: Base configuration to optimize
        max_time_seconds: Maximum time budget
        max_llm_calls: Maximum LLM call budget
        
    Returns:
        Optimized configuration
    """
    config = base_config.copy()
    
    if max_time_seconds:
        config.time_budget_seconds = min(config.time_budget_seconds, max_time_seconds)
        
        # Reduce parameters for tight time budgets
        if max_time_seconds < 60:
            config.max_total_nodes = min(config.max_total_nodes, 30)
            config.max_iterations = min(config.max_iterations, 20)
            config.enable_verification = False
    
    if max_llm_calls:
        config.max_llm_calls = min(config.max_llm_calls, max_llm_calls)
        
        # Estimate and adjust node budget based on LLM calls
        calls_per_node = config.max_branches_per_node * 2  # Generate + score
        if config.enable_verification:
            calls_per_node += 1
        
        max_nodes_for_budget = max_llm_calls // calls_per_node
        config.max_total_nodes = min(config.max_total_nodes, max_nodes_for_budget)
    
    return config
