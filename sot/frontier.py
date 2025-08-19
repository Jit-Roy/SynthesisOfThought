"""
Priority queue wrapper and frontier policies for node expansion.
Manages which nodes to expand next based on different strategies.
"""

import heapq
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Tuple, Dict, Any
from enum import Enum
import random

from .nodes import ReasoningNode, NodeStatus


class FrontierPolicy(ABC):
    """Abstract base class for frontier expansion policies."""
    
    @abstractmethod
    def score_node(self, node: ReasoningNode) -> float:
        """
        Compute a priority score for a node. Higher scores = higher priority.
        """
        pass
    
    @abstractmethod
    def should_expand(self, node: ReasoningNode) -> bool:
        """
        Determine if a node should be considered for expansion.
        """
        pass


class HighestScorePolicy(FrontierPolicy):
    """Expand nodes with highest confidence scores first."""
    
    def score_node(self, node: ReasoningNode) -> float:
        return node.score
    
    def should_expand(self, node: ReasoningNode) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf() and 
                node.score > 0.0)


class DepthFirstPolicy(FrontierPolicy):
    """Expand deepest nodes first (depth-first search)."""
    
    def score_node(self, node: ReasoningNode) -> float:
        # Higher depth = higher priority
        return float(node.depth)
    
    def should_expand(self, node: ReasoningNode) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf())


class BreadthFirstPolicy(FrontierPolicy):
    """Expand shallowest nodes first (breadth-first search)."""
    
    def score_node(self, node: ReasoningNode) -> float:
        # Lower depth = higher priority (negative for max-heap behavior)
        return -float(node.depth)
    
    def should_expand(self, node: ReasoningNode) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf())


class UCBPolicy(FrontierPolicy):
    """
    Upper Confidence Bound policy - balances exploitation and exploration.
    Favors nodes with high scores but also considers uncertainty.
    """
    
    def __init__(self, exploration_weight: float = 1.414):
        self.exploration_weight = exploration_weight
        self.visit_counts: Dict[str, int] = {}
        self.total_visits = 0
    
    def score_node(self, node: ReasoningNode) -> float:
        visits = self.visit_counts.get(node.id, 0)
        if visits == 0:
            return float('inf')  # Unvisited nodes get highest priority
        
        if self.total_visits <= 1:
            return node.score
        
        # UCB1 formula: mean + exploration_term
        exploration_term = self.exploration_weight * (
            (2 * log(self.total_visits) / visits) ** 0.5
        )
        return node.score + exploration_term
    
    def should_expand(self, node: ReasoningNode) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf())
    
    def record_visit(self, node_id: str) -> None:
        """Record that a node was visited/expanded."""
        self.visit_counts[node_id] = self.visit_counts.get(node_id, 0) + 1
        self.total_visits += 1


class RandomPolicy(FrontierPolicy):
    """Random expansion policy for baseline comparison."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def score_node(self, node: ReasoningNode) -> float:
        return random.random()
    
    def should_expand(self, node: ReasoningNode) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf())


class CompositePolicy(FrontierPolicy):
    """
    Combines multiple policies with weights.
    Useful for multi-objective optimization.
    """
    
    def __init__(self, policies_and_weights: List[Tuple[FrontierPolicy, float]]):
        self.policies_and_weights = policies_and_weights
        self.total_weight = sum(weight for _, weight in policies_and_weights)
    
    def score_node(self, node: ReasoningNode) -> float:
        total_score = 0.0
        for policy, weight in self.policies_and_weights:
            score = policy.score_node(node)
            total_score += score * weight
        
        return total_score / self.total_weight if self.total_weight > 0 else 0.0
    
    def should_expand(self, node: ReasoningNode) -> bool:
        # Node should be expandable according to at least one policy
        return any(policy.should_expand(node) for policy, _ in self.policies_and_weights)


class Frontier:
    """
    Priority queue wrapper for managing the frontier of expandable nodes.
    """
    
    def __init__(self, policy: FrontierPolicy):
        self.policy = policy
        self._heap: List[Tuple[float, str, ReasoningNode]] = []
        self._node_map: Dict[str, ReasoningNode] = {}
        self._counter = 0  # For tie-breaking
    
    def add_node(self, node: ReasoningNode) -> None:
        """Add a node to the frontier if it should be expanded."""
        if not self.policy.should_expand(node):
            return
        
        # Remove old version if it exists
        if node.id in self._node_map:
            self.remove_node(node.id)
        
        # Add to heap (negate score for max-heap behavior)
        priority = -self.policy.score_node(node)
        heapq.heappush(self._heap, (priority, self._counter, node))
        self._node_map[node.id] = node
        self._counter += 1
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the frontier.
        Note: This doesn't remove from heap immediately (lazy deletion).
        """
        if node_id in self._node_map:
            del self._node_map[node_id]
            return True
        return False
    
    def pop_best(self) -> Optional[ReasoningNode]:
        """Remove and return the highest priority node."""
        while self._heap:
            priority, counter, node = heapq.heappop(self._heap)
            
            # Check if this node is still valid (not lazily deleted)
            if node.id in self._node_map:
                del self._node_map[node.id]
                
                # Record visit if policy supports it
                if hasattr(self.policy, 'record_visit'):
                    self.policy.record_visit(node.id)
                
                return node
        
        return None
    
    def peek_best(self) -> Optional[ReasoningNode]:
        """Return the highest priority node without removing it."""
        while self._heap:
            priority, counter, node = self._heap[0]
            
            if node.id in self._node_map:
                return node
            else:
                # Remove stale entry
                heapq.heappop(self._heap)
        
        return None
    
    def update_node(self, node: ReasoningNode) -> None:
        """Update a node's priority in the frontier."""
        if node.id in self._node_map:
            # Remove and re-add to update priority
            self.remove_node(node.id)
            self.add_node(node)
    
    def get_all_nodes(self) -> List[ReasoningNode]:
        """Get all nodes currently in the frontier."""
        return list(self._node_map.values())
    
    def clear(self) -> None:
        """Clear the frontier."""
        self._heap.clear()
        self._node_map.clear()
        self._counter = 0
    
    def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        return len(self._node_map) == 0
    
    def size(self) -> int:
        """Get the number of nodes in the frontier."""
        return len(self._node_map)
    
    def contains(self, node_id: str) -> bool:
        """Check if a node is in the frontier."""
        return node_id in self._node_map
    
    def get_stats(self) -> Dict[str, Any]:
        """Get frontier statistics."""
        if not self._node_map:
            return {'size': 0}
        
        nodes = list(self._node_map.values())
        scores = [node.score for node in nodes]
        depths = [node.depth for node in nodes]
        
        return {
            'size': len(nodes),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_depth': sum(depths) / len(depths),
            'min_depth': min(depths),
            'max_depth': max(depths),
            'policy_type': type(self.policy).__name__
        }


# Import math for UCB policy
def log(x):
    """Math log function for UCB policy."""
    import math
    return math.log(x)


# Pre-configured frontier instances for common use cases
def create_best_first_frontier() -> Frontier:
    """Create a frontier that expands highest-scoring nodes first."""
    return Frontier(HighestScorePolicy())


def create_depth_first_frontier() -> Frontier:
    """Create a frontier that expands deepest nodes first."""
    return Frontier(DepthFirstPolicy())


def create_breadth_first_frontier() -> Frontier:
    """Create a frontier that expands shallowest nodes first."""
    return Frontier(BreadthFirstPolicy())


def create_ucb_frontier(exploration_weight: float = 1.414) -> Frontier:
    """Create a frontier with UCB (Upper Confidence Bound) policy."""
    return Frontier(UCBPolicy(exploration_weight))


def create_balanced_frontier() -> Frontier:
    """Create a frontier that balances score and depth."""
    composite_policy = CompositePolicy([
        (HighestScorePolicy(), 0.7),
        (DepthFirstPolicy(), 0.3)
    ])
    return Frontier(composite_policy)
