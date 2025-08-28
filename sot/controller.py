"""
Simple controller for SoT reasoning using existing components.
"""

from typing import Dict, Any, List, Optional
from .config import Config
from .llm import LLM
from .nodes import Node, NodeStatus
from .frontier import Frontier, DepthFirstPolicy
from .branching import SmartBranchingPolicy
import time


class Controller:
    """
    Simple controller that orchestrates SoT reasoning using your existing components.
    """
    
    def __init__(self, config: Config, llm_wrapper: LLM):
        self.config = config
        self.llm = llm_wrapper
        self.nodes: Dict[str, Node] = {}
        self.root_id: Optional[str] = None
        
        # Use your existing frontier with smart branching policy
        policy = SmartBranchingPolicy(llm=llm_wrapper, base_policy=DepthFirstPolicy())
        self.frontier = Frontier(policy)
        
        # Stats tracking
        self.stats = {
            'iterations': 0,
            'total_nodes_generated': 0,
            'total_llm_calls': 0,
            'start_time': 0,
            'end_time': 0
        }
    
    def solve(self, question: str) -> Dict[str, Any]:
        """
        Main solving method using your existing architecture.
        """
        self.stats['start_time'] = time.time()
        
        try:
            # Create root node
            root = create_root_node(question)
            self.nodes[root.id] = root
            self.root_id = root.id
            self.frontier.add_node(root)
            self.stats['total_nodes_generated'] += 1
            
            # Main reasoning loop
            while (not self.frontier.is_empty() and 
                   self.stats['iterations'] < self.config.max_iterations and
                   self.stats['total_nodes_generated'] < self.config.max_total_nodes):
                
                self.stats['iterations'] += 1
                
                # Get next node to expand
                current_node = self.frontier.pop_best()
                if not current_node:
                    break
                
                # Generate reasoning steps
                context = current_node.history or current_node.step_text
                steps = self.llm.generate_steps(context)
                self.stats['total_llm_calls'] += 1
                
                if not steps:
                    current_node.mark_terminal(success=False)
                    continue
                
                # Create child nodes for each step
                for step in steps[:self.config.max_branches_per_node]:
                    if self.stats['total_nodes_generated'] >= self.config.max_total_nodes:
                        break
                    
                    child = Node(
                        step_text=step,
                        depth=current_node.depth + 1,
                        parent=current_node.id
                    )
                    
                    # Update child history
                    child.update_history(self.nodes)
                    
                    # Score the step
                    score = self.llm.score_step(step, context)
                    self.stats['total_llm_calls'] += 1
                    
                    # Apply pruning if enabled
                    if self.config.enable_pruning and score < self.config.min_score_threshold:
                        child.mark_pruned()
                    
                    # Add to store and frontier
                    self.nodes[child.id] = child
                    current_node.add_child(child.id, self.nodes)
                    
                    if child.status == NodeStatus.ACTIVE:
                        self.frontier.add_node(child)
                    
                    self.stats['total_nodes_generated'] += 1
                
                # Mark current node as processed
                if current_node.depth >= self.config.max_depth:
                    current_node.mark_terminal(success=True)
            
            self.stats['end_time'] = time.time()
            
            # Generate result paths
            paths = self._extract_paths()
            
            return {
                'success': True,
                'synthesized_paths': paths,
                'probabilities': [1.0 / len(paths)] * len(paths) if paths else [],
                'processing_stats': self.stats,
                'store_stats': get_store_stats(self.nodes, self.root_id)
            }
            
        except Exception as e:
            self.stats['end_time'] = time.time()
            return {
                'success': False,
                'error': str(e),
                'processing_stats': self.stats,
                'store_stats': get_store_stats(self.nodes, self.root_id)
            }
    
    def _extract_paths(self) -> List['ReasoningPath']:
        """Extract reasoning paths from the tree."""
        paths = []
        
        # Find terminal nodes
        terminal_nodes = [node for node in self.nodes.values() 
                         if node.status == NodeStatus.TERMINAL_SUCCESS or node.is_leaf()]
        
        # Create paths from terminal nodes back to root
        for terminal in terminal_nodes[:5]:  # Limit to top 5 paths
            path_nodes = []
            current = terminal
            visited = set()
            
            while current and current.id not in visited:
                visited.add(current.id)
                path_nodes.append(current)
                if current.parent:
                    current = self.nodes.get(current.parent)
                else:
                    break
            
            if path_nodes:
                path_nodes.reverse()  # Root to terminal order
                paths.append(ReasoningPath(path_nodes))
        
        return paths


class ReasoningPath:
    """Simple path representation."""
    
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.confidence = sum(1.0 for node in nodes if node.status == NodeStatus.ACTIVE) / len(nodes) if nodes else 0.0


def create_root_node(question: str) -> Node:
    """Create a root node for the given question."""
    return Node(
        step_text=f"Question: {question}",
        depth=0,
        history=f"Question: {question}"
    )


def get_store_stats(nodes: Dict[str, Node], root_id: Optional[str]) -> Dict[str, Any]:
    """Get statistics about the node store."""
    if not nodes:
        return {
            'total_nodes': 0,
            'max_depth': 0,
            'frontier_size': 0,
            'terminal_nodes': 0
        }
    
    depths = [node.depth for node in nodes.values()]
    terminal_count = sum(1 for node in nodes.values() 
                        if node.status in [NodeStatus.TERMINAL_SUCCESS, NodeStatus.TERMINAL_FAILURE] or node.is_leaf())
    
    return {
        'total_nodes': len(nodes),
        'max_depth': max(depths) if depths else 0,
        'frontier_size': sum(1 for node in nodes.values() if node.status == NodeStatus.ACTIVE and node.is_leaf()),
        'terminal_nodes': terminal_count
    }
