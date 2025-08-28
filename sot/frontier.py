"""
Priority queue wrapper and frontier policies for node expansion.
Manages which nodes to expand next based on different strategies.
"""

import heapq
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from .nodes import Node, NodeStatus
from .branching import BranchingDecisionFramework, SmartBranchingPolicy


class FrontierPolicy(ABC):
    """Abstract base class for frontier expansion policies."""

    @abstractmethod
    def should_expand(self, node: Node) -> bool:
        """
        Determine if a node should be considered for expansion.
        """
        pass
    
    @abstractmethod
    def score_node(self, node: Node) -> float:
        """
        Score a node for priority ordering.
        Higher scores mean higher priority.
        """
        pass


class DepthFirstPolicy(FrontierPolicy):
    """Expand deepest nodes first (depth-first search)."""
    
    def should_expand(self, node: Node) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf())
    
    def score_node(self, node: Node) -> float:
        return float(node.depth)


class BreadthFirstPolicy(FrontierPolicy):
    """Expand shallowest nodes first (breadth-first search)."""
    
    def should_expand(self, node: Node) -> bool:
        return (node.status == NodeStatus.ACTIVE and 
                node.is_leaf())
    
    def score_node(self, node: Node) -> float:
        return -float(node.depth)  # Negative to prioritize shallow nodes


class Frontier:
    """
    Priority queue wrapper for managing the frontier of expandable nodes.
    """
    
    def __init__(self, policy: FrontierPolicy):
        self.policy = policy
        self._heap: List[Tuple[float, str, Node]] = []
        self._node_map: Dict[str, Node] = {}
        self._counter = 0  # For tie-breaking
    
    def add_node(self, node: Node) -> None:
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
    
    def pop_best(self) -> Optional[Node]:
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
    
    def peek_best(self) -> Optional[Node]:
        """Return the highest priority node without removing it."""
        while self._heap:
            priority, counter, node = self._heap[0]
            
            if node.id in self._node_map:
                return node
            else:
                # Remove stale entry
                heapq.heappop(self._heap)
        
        return None
    
    def update_node(self, node: Node) -> None:
        """Update a node's priority in the frontier."""
        if node.id in self._node_map:
            # Remove and re-add to update priority
            self.remove_node(node.id)
            self.add_node(node)
    
    def get_all_nodes(self) -> List[Node]:
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
        scores = [self.policy.score_node(node) for node in nodes]
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
    
    def should_branch_or_solve_internally(self, node: Node, problem_context: str, 
                                        subproblem_description: str) -> Dict[str, Any]:
        """
        Convenience method to use branching decision framework.
        Only works if the policy is SmartBranchingPolicy.
        """
        if isinstance(self.policy, SmartBranchingPolicy):
            return self.policy.should_branch_or_solve_internally(
                node, problem_context, subproblem_description
            )
        else:
            # Fallback to simple heuristic
            framework = BranchingDecisionFramework()
            return framework.should_branch(node, problem_context, subproblem_description)