"""
SoT algorithm controllers: simple and adaptive branching loops.
Main orchestrators for the Synthesis of Thought reasoning process.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import time
import logging

from .models import ReasoningNode, NodeStatus, create_root_node
from .storage import NodeStore
from .frontier import Frontier, create_best_first_frontier
from .llm_wrapper import LLMWrapper, MockLLMWrapper
from .verifier import Verifier, MockVerifier
from .merge import MergeOperator, GreedySynthesis, SynthesizedPath, calculate_path_probabilities
from .config import SoTConfig


class SoTController(ABC):
    """Abstract base class for Synthesis of Thought controllers."""
    
    @abstractmethod
    def solve(self, question: str) -> Dict[str, Any]:
        """
        Solve a question using Synthesis of Thought.
        
        Args:
            question: The question to reason about
            
        Returns:
            Dictionary containing results and metadata
        """
        pass


class SoTSimple(SoTController):
    """
    Simple SoT implementation with fixed branching and basic policies.
    Good for initial testing and baseline comparisons.
    """
    
    def __init__(self, 
                 config: Optional[SoTConfig] = None,
                 llm_wrapper: Optional[LLMWrapper] = None,
                 verifier: Optional[Verifier] = None,
                 merge_operator: Optional[MergeOperator] = None,
                 frontier: Optional[Frontier] = None):
        
        self.config = config or SoTConfig()
        self.llm_wrapper = llm_wrapper or MockLLMWrapper()
        self.verifier = verifier or MockVerifier()
        self.merge_operator = merge_operator or GreedySynthesis()
        self.frontier = frontier or create_best_first_frontier()
        
        self.store = NodeStore()
        
        # Statistics tracking
        self.stats = {
            'total_nodes_generated': 0,
            'total_llm_calls': 0,
            'total_verification_calls': 0,
            'start_time': None,
            'end_time': None,
            'iterations': 0
        }
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger(f'{self.__class__.__name__}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def solve(self, question: str) -> Dict[str, Any]:
        """Solve using simple SoT algorithm."""
        self.logger.info(f"Starting simple SoT for: {question}")
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize with root node
            root = create_root_node(question)
            self.store.add_node(root)
            self.frontier.add_node(root)
            
            # Main reasoning loop
            self._reasoning_loop()
            
            # Synthesize final paths
            all_paths = self.store.get_all_paths()
            synthesized_paths = self.merge_operator.merge_paths(all_paths, self.store)
            
            # Calculate probabilities
            probabilities = calculate_path_probabilities(synthesized_paths)
            
            self.stats['end_time'] = time.time()
            
            result = {
                'question': question,
                'success': True,
                'synthesized_paths': synthesized_paths,
                'probabilities': probabilities,
                'store_stats': self.store.get_stats(),
                'processing_stats': self.stats.copy(),
                'config': self.config.to_dict()
            }
            
            self.logger.info("Simple SoT completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in simple SoT: {str(e)}")
            self.stats['end_time'] = time.time()
            
            return {
                'question': question,
                'success': False,
                'error': str(e),
                'store_stats': self.store.get_stats(),
                'processing_stats': self.stats.copy()
            }
    
    def _reasoning_loop(self) -> None:
        """Main reasoning loop."""
        while (not self.frontier.is_empty() and 
               len(self.store) < self.config.max_total_nodes and
               self.stats['iterations'] < self.config.max_iterations):
            
            self.stats['iterations'] += 1
            
            # Get next node to expand
            current_node = self.frontier.pop_best()
            if not current_node:
                break
            
            self.logger.debug(f"Expanding node {current_node.id} at depth {current_node.depth}")
            
            # Check expansion criteria
            if (current_node.depth >= self.config.max_depth or
                current_node.score < self.config.min_score_threshold):
                current_node.mark_terminal(success=False)
                continue
            
            # Expand the node
            self._expand_node(current_node)
            
            # Prune if enabled
            if self.config.enable_pruning:
                self._prune_low_scoring_nodes()
        
        self.logger.info(f"Reasoning loop completed after {self.stats['iterations']} iterations")
    
    def _expand_node(self, node: ReasoningNode) -> None:
        """Expand a single node."""
        try:
            # Build context for LLM
            context = self._build_context(node)
            
            # Generate candidate steps
            candidates = self.llm_wrapper.generate_steps(
                context,
                max_steps=self.config.max_branches_per_node,
                temperature=self.config.temperature
            )
            self.stats['total_llm_calls'] += 1
            
            if not candidates:
                node.mark_terminal(success=False)
                return
            
            # Score candidates
            scores = []
            for candidate in candidates:
                score = self.llm_wrapper.score_step(candidate, context)
                scores.append(score)
                self.stats['total_llm_calls'] += 1
            
            # Create child nodes
            for candidate, score in zip(candidates, scores):
                child = ReasoningNode(
                    step_text=candidate,
                    parent=node.id,
                    depth=node.depth + 1,
                    score=score
                )
                
                node.add_child(child.id)
                self.store.add_node(child)
                
                # Verify if enabled
                if self.config.enable_verification:
                    self._verify_node(child, context)
                
                # Add to frontier if suitable
                self.frontier.add_node(child)
                
                self.stats['total_nodes_generated'] += 1
            
            self.logger.debug(f"Generated {len(candidates)} children for node {node.id}")
            
        except Exception as e:
            self.logger.error(f"Error expanding node {node.id}: {str(e)}")
            node.mark_terminal(success=False)
    
    def _build_context(self, node: ReasoningNode) -> str:
        """Build context string for LLM calls."""
        path = self.store.get_path_to_node(node.id)
        
        context_parts = []
        
        # Add the question
        if path:
            context_parts.append(f"Question: {path[0].step_text}")
        
        # Add reasoning steps so far
        if len(path) > 1:
            context_parts.append("\nReasoning so far:")
            for i, step_node in enumerate(path[1:], 1):
                context_parts.append(f"{i}. {step_node.step_text}")
        
        context_parts.append(f"\nCurrent step: {node.step_text}")
        context_parts.append("What are the next possible reasoning steps?")
        
        return "\n".join(context_parts)
    
    def _verify_node(self, node: ReasoningNode, context: str) -> None:
        """Verify a node using the verifier."""
        try:
            result = self.verifier.verify_step(node.step_text, context, node)
            self.stats['total_verification_calls'] += 1
            
            # Update node metadata
            node.metadata.verified = result.verified
            node.metadata.evidence.extend(result.evidence)
            node.metadata.contradictions.extend(result.contradictions)
            node.verifier_score = result.confidence
            
            # Adjust node score based on verification
            if result.verified:
                node.score = min(1.0, node.score * 1.1)  # Small boost
            else:
                node.score = max(0.0, node.score * 0.8)  # Penalty
                
        except Exception as e:
            self.logger.error(f"Error verifying node {node.id}: {str(e)}")
    
    def _prune_low_scoring_nodes(self) -> None:
        """Prune nodes below threshold."""
        pruned_count = 0
        
        for node in list(self.store.nodes.values()):
            if (node.score < self.config.min_score_threshold and
                not node.is_root() and
                node.status == NodeStatus.ACTIVE):
                
                node.mark_pruned()
                self.frontier.remove_node(node.id)
                pruned_count += 1
        
        if pruned_count > 0:
            self.logger.debug(f"Pruned {pruned_count} low-scoring nodes")


class SoTAdaptive(SoTController):
    """
    Adaptive SoT implementation with dynamic branching and advanced policies.
    Adjusts strategy based on progress and resource usage.
    """
    
    def __init__(self, 
                 config: Optional[SoTConfig] = None,
                 llm_wrapper: Optional[LLMWrapper] = None,
                 verifier: Optional[Verifier] = None,
                 merge_operator: Optional[MergeOperator] = None,
                 frontier: Optional[Frontier] = None):
        
        self.config = config or SoTConfig()
        self.llm_wrapper = llm_wrapper or MockLLMWrapper()
        self.verifier = verifier or MockVerifier()
        self.merge_operator = merge_operator or GreedySynthesis()
        self.frontier = frontier or create_best_first_frontier()
        
        self.store = NodeStore()
        
        # Adaptive parameters
        self.current_branching_factor = self.config.max_branches_per_node
        self.expansion_budget = self.config.max_total_nodes
        self.quality_threshold = self.config.min_score_threshold
        
        # Performance tracking for adaptation
        self.performance_history = []
        self.recent_scores = []
        
        # Statistics
        self.stats = {
            'total_nodes_generated': 0,
            'total_llm_calls': 0,
            'total_verification_calls': 0,
            'adaptations_made': 0,
            'start_time': None,
            'end_time': None,
            'iterations': 0
        }
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger(f'{self.__class__.__name__}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def solve(self, question: str) -> Dict[str, Any]:
        """Solve using adaptive SoT algorithm."""
        self.logger.info(f"Starting adaptive SoT for: {question}")
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize
            root = create_root_node(question)
            self.store.add_node(root)
            self.frontier.add_node(root)
            
            # Adaptive reasoning loop
            self._adaptive_reasoning_loop()
            
            # Synthesize results
            all_paths = self.store.get_all_paths()
            synthesized_paths = self.merge_operator.merge_paths(all_paths, self.store)
            probabilities = calculate_path_probabilities(synthesized_paths)
            
            self.stats['end_time'] = time.time()
            
            result = {
                'question': question,
                'success': True,
                'synthesized_paths': synthesized_paths,
                'probabilities': probabilities,
                'store_stats': self.store.get_stats(),
                'processing_stats': self.stats.copy(),
                'config': self.config.to_dict(),
                'adaptations': self.stats['adaptations_made']
            }
            
            self.logger.info("Adaptive SoT completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in adaptive SoT: {str(e)}")
            self.stats['end_time'] = time.time()
            
            return {
                'question': question,
                'success': False,
                'error': str(e),
                'store_stats': self.store.get_stats(),
                'processing_stats': self.stats.copy()
            }
    
    def _adaptive_reasoning_loop(self) -> None:
        """Adaptive reasoning loop with dynamic parameter adjustment."""
        consecutive_low_quality = 0
        
        while (not self.frontier.is_empty() and 
               len(self.store) < self.expansion_budget and
               self.stats['iterations'] < self.config.max_iterations):
            
            self.stats['iterations'] += 1
            
            # Adapt parameters periodically
            if self.stats['iterations'] % 5 == 0:
                self._adapt_parameters()
            
            # Get next node to expand
            current_node = self.frontier.pop_best()
            if not current_node:
                break
            
            # Adaptive expansion criteria
            if (current_node.depth >= self.config.max_depth or
                current_node.score < self.quality_threshold):
                current_node.mark_terminal(success=False)
                continue
            
            # Expand with current branching factor
            children_generated = self._adaptive_expand_node(current_node)
            
            # Track quality for adaptation
            if children_generated > 0:
                child_scores = [
                    child.score for child in self.store.get_children(current_node.id)
                ]
                avg_child_score = sum(child_scores) / len(child_scores)
                self.recent_scores.append(avg_child_score)
                
                # Keep recent history bounded
                if len(self.recent_scores) > 10:
                    self.recent_scores.pop(0)
                
                # Track consecutive low quality
                if avg_child_score < self.config.min_score_threshold:
                    consecutive_low_quality += 1
                else:
                    consecutive_low_quality = 0
                
                # Emergency adaptation for consistently low quality
                if consecutive_low_quality >= 3:
                    self._emergency_adaptation()
                    consecutive_low_quality = 0
            
            # Dynamic pruning
            if self.config.enable_pruning:
                self._adaptive_pruning()
        
        self.logger.info(f"Adaptive reasoning completed after {self.stats['iterations']} iterations")
    
    def _adaptive_expand_node(self, node: ReasoningNode) -> int:
        """Expand node with adaptive branching factor."""
        try:
            context = self._build_context(node)
            
            # Generate candidates
            candidates = self.llm_wrapper.generate_steps(
                context,
                max_steps=self.current_branching_factor,
                temperature=self.config.temperature
            )
            self.stats['total_llm_calls'] += 1
            
            if not candidates:
                node.mark_terminal(success=False)
                return 0
            
            # Score and filter candidates
            scored_candidates = []
            for candidate in candidates:
                score = self.llm_wrapper.score_step(candidate, context)
                self.stats['total_llm_calls'] += 1
                
                # Only keep candidates above adaptive threshold
                if score >= self.quality_threshold:
                    scored_candidates.append((candidate, score))
            
            # Create children from qualified candidates
            children_created = 0
            for candidate, score in scored_candidates:
                child = ReasoningNode(
                    step_text=candidate,
                    parent=node.id,
                    depth=node.depth + 1,
                    score=score
                )
                
                node.add_child(child.id)
                self.store.add_node(child)
                
                if self.config.enable_verification:
                    self._verify_node(child, context)
                
                self.frontier.add_node(child)
                children_created += 1
                self.stats['total_nodes_generated'] += 1
            
            return children_created
            
        except Exception as e:
            self.logger.error(f"Error in adaptive expansion of node {node.id}: {str(e)}")
            node.mark_terminal(success=False)
            return 0
    
    def _adapt_parameters(self) -> None:
        """Adapt parameters based on recent performance."""
        if len(self.recent_scores) < 3:
            return
        
        avg_recent_quality = sum(self.recent_scores) / len(self.recent_scores)
        
        old_branching = self.current_branching_factor
        old_threshold = self.quality_threshold
        
        # Adapt branching factor based on quality
        if avg_recent_quality > 0.7:
            # High quality - can afford to branch more
            self.current_branching_factor = min(
                self.current_branching_factor + 1,
                self.config.max_branches_per_node + 2
            )
        elif avg_recent_quality < 0.4:
            # Low quality - be more selective
            self.current_branching_factor = max(
                self.current_branching_factor - 1,
                1
            )
        
        # Adapt quality threshold based on budget usage
        budget_used = len(self.store) / self.expansion_budget
        
        if budget_used > 0.8:
            # Running out of budget - be more selective
            self.quality_threshold = min(
                self.quality_threshold + 0.1,
                0.8
            )
        elif budget_used < 0.3 and avg_recent_quality > 0.6:
            # Plenty of budget and good quality - be more permissive
            self.quality_threshold = max(
                self.quality_threshold - 0.05,
                self.config.min_score_threshold
            )
        
        # Log adaptations
        if (old_branching != self.current_branching_factor or 
            abs(old_threshold - self.quality_threshold) > 0.01):
            
            self.stats['adaptations_made'] += 1
            self.logger.info(
                f"Adapted parameters: branching {old_branching}->{self.current_branching_factor}, "
                f"threshold {old_threshold:.3f}->{self.quality_threshold:.3f}"
            )
    
    def _emergency_adaptation(self) -> None:
        """Emergency adaptation when quality is consistently low."""
        self.logger.warning("Emergency adaptation triggered due to low quality")
        
        # Reduce branching and increase temperature for more diverse exploration
        self.current_branching_factor = max(1, self.current_branching_factor - 1)
        self.quality_threshold = max(0.2, self.quality_threshold - 0.1)
        
        # Could also adjust LLM temperature here if wrapper supports it
        self.stats['adaptations_made'] += 1
    
    def _adaptive_pruning(self) -> None:
        """Adaptive pruning based on current state."""
        # More aggressive pruning when budget is tight
        budget_used = len(self.store) / self.expansion_budget
        
        if budget_used > 0.7:
            # Aggressive pruning
            prune_threshold = self.quality_threshold + 0.1
        else:
            # Standard pruning
            prune_threshold = self.quality_threshold
        
        pruned = 0
        for node in list(self.store.nodes.values()):
            if (node.score < prune_threshold and
                not node.is_root() and
                node.status == NodeStatus.ACTIVE):
                
                node.mark_pruned()
                self.frontier.remove_node(node.id)
                pruned += 1
        
        if pruned > 0:
            self.logger.debug(f"Adaptive pruning removed {pruned} nodes")
    
    def _build_context(self, node: ReasoningNode) -> str:
        """Build context - same as simple version."""
        path = self.store.get_path_to_node(node.id)
        
        context_parts = []
        if path:
            context_parts.append(f"Question: {path[0].step_text}")
        
        if len(path) > 1:
            context_parts.append("\nReasoning so far:")
            for i, step_node in enumerate(path[1:], 1):
                context_parts.append(f"{i}. {step_node.step_text}")
        
        context_parts.append(f"\nCurrent step: {node.step_text}")
        context_parts.append("What are the next possible reasoning steps?")
        
        return "\n".join(context_parts)
    
    def _verify_node(self, node: ReasoningNode, context: str) -> None:
        """Verify node - same as simple version."""
        try:
            result = self.verifier.verify_step(node.step_text, context, node)
            self.stats['total_verification_calls'] += 1
            
            node.metadata.verified = result.verified
            node.metadata.evidence.extend(result.evidence)
            node.metadata.contradictions.extend(result.contradictions)
            node.verifier_score = result.confidence
            
            if result.verified:
                node.score = min(1.0, node.score * 1.1)
            else:
                node.score = max(0.0, node.score * 0.8)
                
        except Exception as e:
            self.logger.error(f"Error verifying node {node.id}: {str(e)}")


# Factory functions
def create_simple_sot(config: Optional[SoTConfig] = None,
                     llm_wrapper: Optional[LLMWrapper] = None) -> SoTController:
    """Create a simple SoT controller."""
    return SoTSimple(config=config, llm_wrapper=llm_wrapper)


def create_adaptive_sot(config: Optional[SoTConfig] = None,
                       llm_wrapper: Optional[LLMWrapper] = None) -> SoTController:
    """Create an adaptive SoT controller."""
    return SoTAdaptive(config=config, llm_wrapper=llm_wrapper)
