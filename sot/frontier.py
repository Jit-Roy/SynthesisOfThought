"""
Priority queue wrapper and frontier policies for node expansion.
Manages which nodes to expand next based on different strategies.
"""

import heapq
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from .nodes import Node, NodeStatus


class BranchingDecisionFramework:
    """
    Core Decision Framework to determine whether to solve a subproblem 
    inside the current node (internal processing) or branch into child nodes.
    """
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def should_branch(self, node: Node, problem_context: str, subproblem_description: str) -> Dict[str, Any]:
        """
        Determine whether to solve the subproblem in current node or branch into children.
        
        Args:
            node: Current node being processed
            problem_context: Context of the main problem
            subproblem_description: Description of the subproblem to be solved
            
        Returns:
            Dict containing decision, reasoning, and confidence
        """
        # First, evaluate using heuristic rules
        heuristic_decision = self._evaluate_heuristics(node, problem_context, subproblem_description)
        
        # If LLM is available, get additional reasoning
        llm_decision = None
        if self.llm:
            llm_decision = self._get_llm_decision(node, problem_context, subproblem_description)
        
        # Combine decisions
        final_decision = self._combine_decisions(heuristic_decision, llm_decision)
        
        return final_decision
    
    def _evaluate_heuristics(self, node: Node, problem_context: str, subproblem_description: str) -> Dict[str, Any]:
        """
        Evaluate using heuristic rules to decide branching vs internal processing.
        """
        reasons_for_current_node = []
        reasons_for_branching = []
        score_current = 0
        score_branching = 0
        
        # Check for simple iteration patterns
        if self._is_simple_iteration(subproblem_description):
            reasons_for_current_node.append("Simple iteration pattern detected")
            score_current += 2
        
        # Check for bounded search space
        if self._has_bounded_search_space(subproblem_description):
            reasons_for_current_node.append("Bounded search space")
            score_current += 1
        
        # Check for linear/sequential computation
        if self._is_linear_computation(subproblem_description):
            reasons_for_current_node.append("Linear/sequential computation")
            score_current += 1
        
        # Check for manageable memory requirements
        if self._has_manageable_memory(subproblem_description):
            reasons_for_current_node.append("Manageable memory requirements")
            score_current += 1
        
        # Check for different problem states
        if self._creates_different_states(subproblem_description):
            reasons_for_branching.append("Each choice leads to different problem states")
            score_branching += 2
        
        # Check for backtracking needs
        if self._needs_backtracking(subproblem_description):
            reasons_for_branching.append("Requires backtracking or parallel exploration")
            score_branching += 2
        
        # Check for different constraints
        if self._has_different_constraints(subproblem_description):
            reasons_for_branching.append("Subproblems have different constraints")
            score_branching += 1
        
        # Check for memoization benefits
        if self._benefits_from_memoization(subproblem_description):
            reasons_for_branching.append("Benefits from caching intermediate results")
            score_branching += 1
        
        # Check for exponential growth
        if self._has_exponential_growth(subproblem_description):
            reasons_for_branching.append("Exponential growth potential")
            score_branching += 2
        
        # Node depth consideration
        if node.depth > 5:
            reasons_for_current_node.append("Deep node - prefer internal processing")
            score_current += 1
        
        # Current node complexity
        if len(node.children) > 3:
            reasons_for_current_node.append("Node already has many children")
            score_current += 1
        
        decision = "branch" if score_branching > score_current else "current_node"
        confidence = abs(score_branching - score_current) / max(score_branching + score_current, 1)
        
        return {
            "decision": decision,
            "confidence": confidence,
            "score_current": score_current,
            "score_branching": score_branching,
            "reasons_current": reasons_for_current_node,
            "reasons_branching": reasons_for_branching,
            "method": "heuristic"
        }
    
    def _get_llm_decision(self, node: Node, problem_context: str, subproblem_description: str) -> Dict[str, Any]:
        """
        Use LLM to make a more nuanced decision about branching.
        """
        prompt = f"""
You are an expert in problem-solving strategies and decision trees. Given the following context, decide whether to solve the subproblem in the current node or create child nodes for branching.

PROBLEM CONTEXT:
{problem_context}

CURRENT NODE INFO:
- Depth: {node.depth}
- Current children: {len(node.children)}
- Node status: {node.status.value}
- Step text: {node.step_text[:200]}...

SUBPROBLEM TO SOLVE:
{subproblem_description}

DECISION CRITERIA:
Use Current Node (Internal Processing) when:
- Simple iterations that don't require backtracking
- Bounded search space (like checking x=0,1,2,...)
- Linear/sequential computation
- Manageable memory requirements
- Subproblems don't generate complex branches

Use Child Nodes (Branching) when:
- Each choice leads to fundamentally different problem states
- Need backtracking or parallel exploration
- Subproblems have different constraints or rules
- Want to memorize/cache intermediate results
- Decision tree has exponential growth potential

Please respond with:
1. DECISION: either "current_node" or "branch"
2. REASONING: Explain your decision in 1-2 sentences
3. CONFIDENCE: A number from 0.0 to 1.0

Format: DECISION: [choice] | REASONING: [explanation] | CONFIDENCE: [number]
"""
        
        try:
            response = self.llm.generate_steps(prompt)
            if response:
                return self._parse_llm_response(response[0])
        except Exception as e:
            print(f"LLM decision failed: {e}")
        
        return None
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured decision."""
        try:
            parts = response.split("|")
            decision_part = parts[0].strip()
            reasoning_part = parts[1].strip() if len(parts) > 1 else ""
            confidence_part = parts[2].strip() if len(parts) > 2 else "0.5"
            
            decision = "current_node" if "current_node" in decision_part.lower() else "branch"
            reasoning = reasoning_part.replace("REASONING:", "").strip()
            confidence = float(confidence_part.replace("CONFIDENCE:", "").strip())
            
            return {
                "decision": decision,
                "reasoning": reasoning,
                "confidence": min(max(confidence, 0.0), 1.0),
                "method": "llm"
            }
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return {
                "decision": "current_node",
                "reasoning": "Failed to parse LLM response",
                "confidence": 0.5,
                "method": "llm_fallback"
            }
    
    def _combine_decisions(self, heuristic_decision: Dict, llm_decision: Optional[Dict]) -> Dict[str, Any]:
        """Combine heuristic and LLM decisions."""
        if not llm_decision:
            return heuristic_decision
        
        # Weight the decisions based on confidence
        heuristic_weight = heuristic_decision["confidence"]
        llm_weight = llm_decision["confidence"]
        
        total_weight = heuristic_weight + llm_weight
        if total_weight == 0:
            final_decision = heuristic_decision["decision"]
        else:
            # Use weighted voting
            heuristic_vote = 1 if heuristic_decision["decision"] == "branch" else 0
            llm_vote = 1 if llm_decision["decision"] == "branch" else 0
            
            weighted_score = (heuristic_vote * heuristic_weight + llm_vote * llm_weight) / total_weight
            final_decision = "branch" if weighted_score > 0.5 else "current_node"
        
        return {
            "decision": final_decision,
            "confidence": (heuristic_weight + llm_weight) / 2,
            "heuristic_decision": heuristic_decision,
            "llm_decision": llm_decision,
            "method": "combined"
        }
    
    # Heuristic helper methods
    def _is_simple_iteration(self, description: str) -> bool:
        """Check if the subproblem involves simple iteration."""
        iteration_keywords = ["x=0", "x=1", "iterate", "loop", "for each", "check values", "try values"]
        return any(keyword in description.lower() for keyword in iteration_keywords)
    
    def _has_bounded_search_space(self, description: str) -> bool:
        """Check if the search space is bounded."""
        bounded_keywords = ["finite", "limited", "bounded", "0 to", "small range", "few values"]
        return any(keyword in description.lower() for keyword in bounded_keywords)
    
    def _is_linear_computation(self, description: str) -> bool:
        """Check if computation is linear/sequential."""
        linear_keywords = ["sequential", "linear", "step by step", "one by one", "incrementally"]
        return any(keyword in description.lower() for keyword in linear_keywords)
    
    def _has_manageable_memory(self, description: str) -> bool:
        """Check if memory requirements are manageable."""
        # Assume manageable unless explicitly mentioned as complex
        complex_keywords = ["exponential space", "large memory", "store all", "cache everything"]
        return not any(keyword in description.lower() for keyword in complex_keywords)
    
    def _creates_different_states(self, description: str) -> bool:
        """Check if choices lead to different problem states."""
        state_keywords = ["different states", "branch", "multiple paths", "different approaches", "alternatives"]
        return any(keyword in description.lower() for keyword in state_keywords)
    
    def _needs_backtracking(self, description: str) -> bool:
        """Check if backtracking is needed."""
        backtrack_keywords = ["backtrack", "undo", "try and fail", "dead end", "explore alternatives"]
        return any(keyword in description.lower() for keyword in backtrack_keywords)
    
    def _has_different_constraints(self, description: str) -> bool:
        """Check if subproblems have different constraints."""
        constraint_keywords = ["different constraints", "varying rules", "conditional", "depends on"]
        return any(keyword in description.lower() for keyword in constraint_keywords)
    
    def _benefits_from_memoization(self, description: str) -> bool:
        """Check if memoization would be beneficial."""
        memo_keywords = ["cache", "memoize", "store results", "reuse", "overlapping subproblems"]
        return any(keyword in description.lower() for keyword in memo_keywords)
    
    def _has_exponential_growth(self, description: str) -> bool:
        """Check if the problem has exponential growth potential."""
        exp_keywords = ["exponential", "combinatorial", "factorial", "recursive", "tree explosion"]
        return any(keyword in description.lower() for keyword in exp_keywords)


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


class SmartBranchingPolicy(FrontierPolicy):
    """
    Policy that uses the BranchingDecisionFramework to make intelligent
    decisions about when to expand nodes vs solve internally.
    """
    
    def __init__(self, llm=None, base_policy=None):
        self.branching_framework = BranchingDecisionFramework(llm)
        self.base_policy = base_policy or DepthFirstPolicy()
        self.decision_cache = {}
    
    def should_expand(self, node: Node) -> bool:
        return self.base_policy.should_expand(node)
    
    def score_node(self, node: Node) -> float:
        return self.base_policy.score_node(node)
    
    def should_branch_or_solve_internally(self, node: Node, problem_context: str, 
                                        subproblem_description: str) -> Dict[str, Any]:
        """
        Main method to decide whether to branch or solve internally.
        """
        cache_key = f"{node.id}_{hash(subproblem_description)}"
        
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]
        
        decision = self.branching_framework.should_branch(
            node, problem_context, subproblem_description
        )
        
        self.decision_cache[cache_key] = decision
        return decision


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