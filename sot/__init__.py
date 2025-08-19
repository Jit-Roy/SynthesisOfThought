"""
Synthesis of Thought (SoT) package.
A framework for multi-path reasoning synthesis rather than single-path selection.
"""

from .models import ReasoningNode, NodeMetadata, SamplingInfo, NodeStatus
from .storage import NodeStore
from .frontier import FrontierManager, FrontierPolicy, HighestScorePolicy, UCBPolicy
from .llm_wrapper import LLMWrapper, MockLLMWrapper
from .verifier import VerificationResult, MockVerifier, NLIVerifier
from .merge import PathScore, GreedySynthesis
from .controller import SoTController, SoTSimple, SoTAdaptive
from .config import SoTConfig, PresetConfigs
from .utils import (
    hash_dict, hash_object, content_hash,
    deduplicate_by_content, normalize_text, setup_logger,
    validate_probability, normalize_scores
)

__version__ = "1.0.0"
__author__ = "SoT Team"

__all__ = [
    # Core models
    'ReasoningNode',
    'NodeMetadata', 
    'SamplingInfo',
    'NodeStatus',
    
    # Storage
    'NodeStore',
    
    # Frontier management
    'FrontierManager',
    'FrontierPolicy',
    'HighestScorePolicy',
    'UCBPolicy',
    
    # LLM integration
    'LLMWrapper',
    'MockLLMWrapper',
    
    # Verification
    'VerificationResult',
    'NLIVerifier',
    'MockVerifier',
    
    # Merging/synthesis
    'PathScore',
    'GreedySynthesis',
    
    # Main controllers
    'SoTController',
    'SoTSimple',
    'SoTAdaptive',
    
    # Configuration
    'SoTConfig',
    'PresetConfigs',
    
    # Utilities
    'hash_dict',
    'hash_object',
    'content_hash',
    'deduplicate_by_content',
    'normalize_text',
    'setup_logger',
    'validate_probability',
    'normalize_scores',
]
