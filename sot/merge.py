"""
Merge operator and greedy synthesis logic.
Handles combining multiple reasoning paths into synthesized solutions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import math
from collections import defaultdict

from .models import ReasoningNode, NodeStatus
from .storage import NodeStore


@dataclass
class PathScore:
    """Comprehensive scoring for a reasoning path."""
    confidence: float = 0.0      # Average node confidence
    completeness: float = 0.0    # How complete the reasoning appears
    coherence: float = 0.0       # How well steps flow together
    novelty: float = 0.0         # How unique/novel the approach is
    evidence_support: float = 0.0 # Evidence backing the path
    consistency: float = 0.0     # Internal consistency score
    composite: float = 0.0       # Overall composite score
    
    def __post_init__(self):
        """Calculate composite score after initialization."""
        self.composite = self._calculate_composite()
    
    def _calculate_composite(self) -> float:
        """Calculate weighted composite score."""
        weights = {
            'confidence': 0.25,
            'completeness': 0.20,
            'coherence': 0.15,
            'novelty': 0.10,
            'evidence_support': 0.15,
            'consistency': 0.15
        }
        
        return (weights['confidence'] * self.confidence +
                weights['completeness'] * self.completeness +
                weights['coherence'] * self.coherence +
                weights['novelty'] * self.novelty +
                weights['evidence_support'] * self.evidence_support +
                weights['consistency'] * self.consistency)


@dataclass 
class SynthesizedPath:
    """A path created by merging multiple reasoning trajectories."""
    nodes: List[ReasoningNode]
    source_paths: List[List[str]]  # Original path node IDs that contributed
    weights: List[float]          # Weight/probability for each node
    score: PathScore
    confidence: float = 0.0
    synthesis_method: str = "unknown"
    
    def get_text_sequence(self) -> List[str]:
        """Get the text sequence of the synthesized path."""
        return [node.step_text for node in self.nodes]
    
    def get_total_weight(self) -> float:
        """Get the total weight of this synthesized path."""
        return sum(self.weights)
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = self.get_total_weight()
        if total > 0:
            self.weights = [w / total for w in self.weights]


class MergeOperator(ABC):
    """Abstract base class for path merging operations."""
    
    @abstractmethod
    def merge_paths(self, paths: List[List[ReasoningNode]], 
                   store: NodeStore) -> List[SynthesizedPath]:
        """
        Merge multiple reasoning paths into synthesized paths.
        
        Args:
            paths: List of reasoning paths (each path is a list of nodes)
            store: Node store for context and additional operations
        
        Returns:
            List of synthesized paths
        """
        pass
    
    @abstractmethod
    def score_path(self, path: List[ReasoningNode], store: NodeStore) -> PathScore:
        """Score a reasoning path comprehensively."""
        pass


class GreedySynthesis(MergeOperator):
    """
    Greedy synthesis approach that combines paths based on scores and complementarity.
    """
    
    def __init__(self, max_synthesis_paths: int = 5):
        self.max_synthesis_paths = max_synthesis_paths
    
    def merge_paths(self, paths: List[List[ReasoningNode]], 
                   store: NodeStore) -> List[SynthesizedPath]:
        """Merge paths using greedy synthesis strategy."""
        if not paths:
            return []
        
        # Score all input paths
        path_scores = [(path, self.score_path(path, store)) for path in paths]
        
        # Sort by composite score
        path_scores.sort(key=lambda x: x[1].composite, reverse=True)
        
        synthesized = []
        
        # Strategy 1: Keep top individual paths as-is
        for path, score in path_scores[:self.max_synthesis_paths]:
            weights = [node.probability * score.composite for node in path]
            synthesis = SynthesizedPath(
                nodes=path.copy(),
                source_paths=[[node.id for node in path]],
                weights=weights,
                score=score,
                confidence=score.composite,
                synthesis_method="individual"
            )
            synthesized.append(synthesis)
        
        # Strategy 2: Find complementary fragments and merge
        complementary_pairs = self._find_complementary_pairs(
            [ps[0] for ps in path_scores], store
        )
        
        for path1_idx, path2_idx, complement_score in complementary_pairs:
            if len(synthesized) >= self.max_synthesis_paths:
                break
                
            path1, score1 = path_scores[path1_idx]
            path2, score2 = path_scores[path2_idx]
            
            merged = self._merge_complementary_paths(
                path1, path2, score1, score2, complement_score, store
            )
            
            if merged:
                synthesized.append(merged)
        
        # Strategy 3: Consensus-based synthesis
        if len(synthesized) < self.max_synthesis_paths:
            consensus_path = self._create_consensus_path(
                [ps[0] for ps in path_scores], store
            )
            if consensus_path:
                synthesized.append(consensus_path)
        
        # Sort by confidence and return top results
        synthesized.sort(key=lambda x: x.confidence, reverse=True)
        return synthesized[:self.max_synthesis_paths]
    
    def score_path(self, path: List[ReasoningNode], store: NodeStore) -> PathScore:
        """Score a path comprehensively."""
        if not path:
            return PathScore()
        
        # Confidence: average of node scores
        confidence = sum(node.score for node in path) / len(path)
        
        # Completeness: based on terminal status and depth
        completeness = self._score_completeness(path)
        
        # Coherence: how well steps connect logically
        coherence = self._score_coherence(path)
        
        # Novelty: uniqueness compared to other paths
        novelty = self._score_novelty(path, store)
        
        # Evidence support: amount of supporting evidence
        evidence_support = self._score_evidence_support(path)
        
        # Consistency: internal consistency using verification
        consistency = self._score_consistency(path)
        
        return PathScore(
            confidence=confidence,
            completeness=completeness,
            coherence=coherence,
            novelty=novelty,
            evidence_support=evidence_support,
            consistency=consistency
        )
    
    def _score_completeness(self, path: List[ReasoningNode]) -> float:
        """Score how complete a reasoning path is."""
        if not path:
            return 0.0
        
        last_node = path[-1]
        
        # Terminal success gets high completeness
        if last_node.status == NodeStatus.TERMINAL_SUCCESS:
            return 1.0
        
        # Terminal failure gets low completeness
        if last_node.status == NodeStatus.TERMINAL_FAILURE:
            return 0.2
        
        # For non-terminal paths, consider depth and structure
        depth_factor = min(last_node.depth / 5.0, 0.5)  # Bonus up to depth 5
        
        # Check if path has logical conclusion indicators
        conclusion_indicators = ['therefore', 'thus', 'in conclusion', 'consequently']
        has_conclusion = any(
            any(indicator in node.step_text.lower() for indicator in conclusion_indicators)
            for node in path
        )
        
        conclusion_bonus = 0.3 if has_conclusion else 0.0
        
        return min(1.0, 0.3 + depth_factor + conclusion_bonus)
    
    def _score_coherence(self, path: List[ReasoningNode]) -> float:
        """Score logical coherence of steps."""
        if len(path) < 2:
            return 1.0
        
        coherence_score = 0.0
        transitions = len(path) - 1
        
        for i in range(1, len(path)):
            prev_text = path[i-1].step_text.lower()
            curr_text = path[i].step_text.lower()
            
            # Look for logical connectors
            connectors = ['therefore', 'thus', 'because', 'since', 'so', 
                         'consequently', 'as a result', 'this means', 'which leads to']
            
            has_connector = any(conn in curr_text for conn in connectors)
            
            # Check for thematic continuity (word overlap)
            prev_words = set(prev_text.split())
            curr_words = set(curr_text.split())
            overlap = len(prev_words.intersection(curr_words))
            continuity = min(overlap / 10.0, 0.3)  # Cap at 0.3
            
            # Score this transition
            transition_score = 0.4  # baseline
            if has_connector:
                transition_score += 0.4
            transition_score += continuity
            
            coherence_score += transition_score
        
        return coherence_score / transitions
    
    def _score_novelty(self, path: List[ReasoningNode], store: NodeStore) -> float:
        """Score the novelty/uniqueness of a path."""
        all_paths = store.get_all_paths()
        if len(all_paths) <= 1:
            return 1.0
        
        # Compare with other paths
        similarity_scores = []
        for other_path in all_paths:
            if other_path != path:
                similarity = self._calculate_path_similarity(path, other_path)
                similarity_scores.append(similarity)
        
        if not similarity_scores:
            return 1.0
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        return max(0.0, 1.0 - avg_similarity)
    
    def _score_evidence_support(self, path: List[ReasoningNode]) -> float:
        """Score evidence support for the path."""
        if not path:
            return 0.0
        
        total_evidence = 0
        verified_nodes = 0
        
        for node in path:
            total_evidence += len(node.metadata.evidence)
            if node.metadata.verified:
                verified_nodes += 1
        
        evidence_density = min(total_evidence / len(path) / 3.0, 1.0)  # Normalize
        verification_ratio = verified_nodes / len(path)
        
        return (evidence_density * 0.4 + verification_ratio * 0.6)
    
    def _score_consistency(self, path: List[ReasoningNode]) -> float:
        """Score internal consistency of the path."""
        if not path:
            return 1.0
        
        # Check for contradictions in metadata
        contradiction_count = sum(len(node.metadata.contradictions) for node in path)
        contradiction_penalty = min(contradiction_count * 0.2, 0.8)
        
        # Check for logical consistency (simple heuristics)
        consistency_score = 1.0 - contradiction_penalty
        
        # Look for contradictory statements within the path
        texts = [node.step_text.lower() for node in path]
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts[i+1:], i+1):
                if self._texts_contradict(text1, text2):
                    consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _texts_contradict(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection between texts."""
        contradictory_pairs = [
            ('increase', 'decrease'), ('rise', 'fall'), ('hot', 'cold'),
            ('fast', 'slow'), ('large', 'small'), ('possible', 'impossible')
        ]
        
        for word1, word2 in contradictory_pairs:
            if ((word1 in text1 and word2 in text2) or 
                (word2 in text1 and word1 in text2)):
                return True
        
        return False
    
    def _calculate_path_similarity(self, path1: List[ReasoningNode], 
                                  path2: List[ReasoningNode]) -> float:
        """Calculate similarity between two paths."""
        if not path1 or not path2:
            return 0.0
        
        # Text-based similarity
        text1 = ' '.join(node.step_text for node in path1).lower()
        text2 = ' '.join(node.step_text for node in path2).lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        
        # Length similarity factor
        len_factor = 1.0 - abs(len(path1) - len(path2)) / max(len(path1), len(path2))
        
        return jaccard * len_factor
    
    def _find_complementary_pairs(self, paths: List[List[ReasoningNode]], 
                                 store: NodeStore) -> List[Tuple[int, int, float]]:
        """Find pairs of paths that complement each other."""
        complementary = []
        
        for i, path1 in enumerate(paths):
            for j, path2 in enumerate(paths[i+1:], i+1):
                complement_score = self._assess_complementarity(path1, path2)
                if complement_score > 0.5:  # Threshold for complementarity
                    complementary.append((i, j, complement_score))
        
        # Sort by complementarity score
        complementary.sort(key=lambda x: x[2], reverse=True)
        return complementary
    
    def _assess_complementarity(self, path1: List[ReasoningNode], 
                               path2: List[ReasoningNode]) -> float:
        """Assess how well two paths complement each other."""
        if not path1 or not path2:
            return 0.0
        
        # Check for different approaches (low similarity but no contradiction)
        similarity = self._calculate_path_similarity(path1, path2)
        
        # Look for complementary concepts
        complementary_concepts = [
            ('cause', 'effect'), ('problem', 'solution'), ('theory', 'evidence'),
            ('structure', 'function'), ('internal', 'external')
        ]
        
        text1 = ' '.join(node.step_text for node in path1).lower()
        text2 = ' '.join(node.step_text for node in path2).lower()
        
        complement_score = 0.0
        for concept1, concept2 in complementary_concepts:
            if ((concept1 in text1 and concept2 in text2) or
                (concept2 in text1 and concept1 in text2)):
                complement_score += 0.2
        
        # Good complementarity: different approaches (low similarity) with complementary concepts
        if similarity < 0.4 and complement_score > 0.0:
            return complement_score + (1.0 - similarity) * 0.3
        
        return 0.0
    
    def _merge_complementary_paths(self, path1: List[ReasoningNode], 
                                  path2: List[ReasoningNode],
                                  score1: PathScore, score2: PathScore,
                                  complement_score: float,
                                  store: NodeStore) -> Optional[SynthesizedPath]:
        """Merge two complementary paths."""
        
        # Simple interleaving strategy
        merged_nodes = []
        merged_weights = []
        source_paths = []
        
        # Add nodes from both paths, alternating
        max_len = max(len(path1), len(path2))
        
        for i in range(max_len):
            if i < len(path1):
                merged_nodes.append(path1[i])
                merged_weights.append(path1[i].probability * score1.composite)
                source_paths.append([path1[i].id])
            
            if i < len(path2) and i > 0:  # Skip root from second path
                merged_nodes.append(path2[i])
                merged_weights.append(path2[i].probability * score2.composite)
                source_paths.append([path2[i].id])
        
        # Score the merged path
        merged_score = PathScore(
            confidence=(score1.confidence + score2.confidence) / 2,
            completeness=max(score1.completeness, score2.completeness),
            coherence=min(score1.coherence, score2.coherence) * 0.9,  # Slight penalty
            novelty=(score1.novelty + score2.novelty) / 2,
            evidence_support=(score1.evidence_support + score2.evidence_support) / 2,
            consistency=min(score1.consistency, score2.consistency) * 0.9
        )
        
        # Boost score based on complementarity
        merged_score.composite *= (1.0 + complement_score * 0.2)
        
        return SynthesizedPath(
            nodes=merged_nodes,
            source_paths=source_paths,
            weights=merged_weights,
            score=merged_score,
            confidence=merged_score.composite,
            synthesis_method="complementary_merge"
        )
    
    def _create_consensus_path(self, paths: List[List[ReasoningNode]], 
                              store: NodeStore) -> Optional[SynthesizedPath]:
        """Create a consensus path from multiple paths."""
        if not paths:
            return None
        
        # Group steps by depth and find most common ones
        steps_by_depth = defaultdict(list)
        
        for path in paths:
            for node in path:
                steps_by_depth[node.depth].append(node)
        
        # Build consensus path
        consensus_nodes = []
        weights = []
        source_paths = []
        
        for depth in sorted(steps_by_depth.keys()):
            nodes_at_depth = steps_by_depth[depth]
            
            # Find most frequent step text at this depth
            step_counts = defaultdict(list)
            for node in nodes_at_depth:
                normalized_text = node.step_text.lower().strip()
                step_counts[normalized_text].append(node)
            
            if step_counts:
                # Pick most common step
                most_common_text = max(step_counts.keys(), 
                                     key=lambda k: len(step_counts[k]))
                representative_nodes = step_counts[most_common_text]
                
                # Use the highest-scoring representative
                best_node = max(representative_nodes, key=lambda n: n.score)
                
                consensus_nodes.append(best_node)
                weights.append(len(representative_nodes) / len(nodes_at_depth))
                source_paths.append([n.id for n in representative_nodes])
        
        if not consensus_nodes:
            return None
        
        # Score the consensus path
        score = self.score_path(consensus_nodes, store)
        
        return SynthesizedPath(
            nodes=consensus_nodes,
            source_paths=source_paths,
            weights=weights,
            score=score,
            confidence=score.composite,
            synthesis_method="consensus"
        )


# Factory functions
def create_greedy_synthesis(max_paths: int = 5) -> MergeOperator:
    """Create a greedy synthesis operator."""
    return GreedySynthesis(max_paths)


def calculate_path_probabilities(synthesized_paths: List[SynthesizedPath]) -> List[float]:
    """Calculate normalized probabilities for synthesized paths."""
    if not synthesized_paths:
        return []
    
    # Use confidence scores as basis for probabilities
    confidences = [path.confidence for path in synthesized_paths]
    
    # Apply softmax for normalization
    max_conf = max(confidences)
    exp_confs = [math.exp(conf - max_conf) for conf in confidences]
    total_exp = sum(exp_confs)
    
    if total_exp == 0:
        # Fallback to uniform distribution
        return [1.0 / len(synthesized_paths)] * len(synthesized_paths)
    
    probabilities = [exp_conf / total_exp for exp_conf in exp_confs]
    
    return probabilities
