"""
NLI/evidence/checker wrappers and tool-call recording helpers.
Handles verification of reasoning steps for consistency and validity.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import random
import re

from .models import ReasoningNode


class VerificationResult:
    """Container for verification results."""
    
    def __init__(self, verified: bool, confidence: float = 0.5, 
                 evidence: Optional[List[str]] = None,
                 contradictions: Optional[List[str]] = None,
                 explanation: str = ""):
        self.verified = verified
        self.confidence = confidence  # 0.0 to 1.0
        self.evidence = evidence or []
        self.contradictions = contradictions or []
        self.explanation = explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'verified': self.verified,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'contradictions': self.contradictions,
            'explanation': self.explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create from dictionary."""
        return cls(
            verified=data.get('verified', False),
            confidence=data.get('confidence', 0.5),
            evidence=data.get('evidence', []),
            contradictions=data.get('contradictions', []),
            explanation=data.get('explanation', '')
        )


class Verifier(ABC):
    """Abstract base class for reasoning step verification."""
    
    @abstractmethod
    def verify_step(self, step: str, context: str, 
                   node: Optional[ReasoningNode] = None) -> VerificationResult:
        """
        Verify a reasoning step for validity and consistency.
        
        Args:
            step: The reasoning step to verify
            context: The context/history leading to this step
            node: Optional node object for additional metadata
        
        Returns:
            VerificationResult with verification status and details
        """
        pass
    
    @abstractmethod
    def verify_consistency(self, steps: List[str]) -> VerificationResult:
        """
        Verify consistency across multiple reasoning steps.
        
        Args:
            steps: List of reasoning steps to check for consistency
        
        Returns:
            VerificationResult indicating if steps are consistent
        """
        pass


class MockVerifier(Verifier):
    """Mock verifier for testing and development."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        # Patterns that suggest good reasoning
        self.positive_patterns = [
            r'\b(because|therefore|thus|consequently|as a result)\b',
            r'\b(evidence|data|research|study)\b',
            r'\b(analysis|examination|investigation)\b',
            r'\b(specific|detailed|precise)\b'
        ]
        
        # Patterns that suggest poor reasoning
        self.negative_patterns = [
            r'\b(maybe|possibly|might|could be|perhaps)\b',
            r'\b(always|never|impossible|definitely)\b',
            r'\b(obviously|clearly|definitely)\b',
            r'\b(random|arbitrary|guess)\b'
        ]
        
        # Contradiction indicators
        self.contradiction_patterns = [
            (r'\b(not|no|never)\b', r'\b(yes|always|definitely)\b'),
            (r'\b(increase|rise|grow)\b', r'\b(decrease|fall|shrink)\b'),
            (r'\b(hot|warm)\b', r'\b(cold|cool)\b'),
            (r'\b(fast|quick)\b', r'\b(slow|sluggish)\b')
        ]
    
    def verify_step(self, step: str, context: str, 
                   node: Optional[ReasoningNode] = None) -> VerificationResult:
        """Mock verification of a single step."""
        step_lower = step.lower()
        context_lower = context.lower()
        
        # Base verification score
        confidence = 0.5
        verified = True
        evidence = []
        contradictions = []
        
        # Check for positive indicators
        positive_score = 0
        for pattern in self.positive_patterns:
            if re.search(pattern, step_lower):
                positive_score += 1
                evidence.append(f"Contains reasoning indicator: {pattern}")
        
        # Check for negative indicators
        negative_score = 0
        for pattern in self.negative_patterns:
            if re.search(pattern, step_lower):
                negative_score += 1
                contradictions.append(f"Contains weak reasoning: {pattern}")
        
        # Adjust confidence based on indicators
        confidence += positive_score * 0.1
        confidence -= negative_score * 0.15
        
        # Check for contradictions with context
        for pos_pattern, neg_pattern in self.contradiction_patterns:
            if (re.search(pos_pattern, step_lower) and 
                re.search(neg_pattern, context_lower)):
                contradictions.append(f"Potential contradiction: step says {pos_pattern}, context implies {neg_pattern}")
                confidence -= 0.2
                verified = False
        
        # Length and complexity bonus
        if len(step) > 30:
            confidence += 0.05
            evidence.append("Step is detailed")
        
        # Check for logical connectors
        if any(word in step_lower for word in ['because', 'therefore', 'thus', 'so']):
            confidence += 0.1
            evidence.append("Contains logical connectors")
        
        # Contextual relevance
        context_words = set(context_lower.split())
        step_words = set(step_lower.split())
        overlap = len(context_words.intersection(step_words))
        if overlap > 2:
            confidence += 0.1
            evidence.append("Contextually relevant")
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine verification status
        if confidence < 0.3:
            verified = False
        
        explanation = f"Step verification: {len(evidence)} positive indicators, {len(contradictions)} issues"
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            explanation=explanation
        )
    
    def verify_consistency(self, steps: List[str]) -> VerificationResult:
        """Mock consistency verification across steps."""
        if len(steps) < 2:
            return VerificationResult(
                verified=True, 
                confidence=1.0,
                explanation="Single step - no consistency issues"
            )
        
        contradictions = []
        evidence = []
        
        # Check for contradictions between steps
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps[i+1:], i+1):
                step1_lower = step1.lower()
                step2_lower = step2.lower()
                
                for pos_pattern, neg_pattern in self.contradiction_patterns:
                    if (re.search(pos_pattern, step1_lower) and 
                        re.search(neg_pattern, step2_lower)):
                        contradictions.append(
                            f"Contradiction between step {i+1} and step {j+1}: "
                            f"'{pos_pattern}' vs '{neg_pattern}'"
                        )
        
        # Check for logical flow
        logical_flow = 0
        for i in range(len(steps) - 1):
            step1 = steps[i].lower()
            step2 = steps[i+1].lower()
            
            # Look for logical connectors in step2 that reference step1
            if any(connector in step2 for connector in ['therefore', 'thus', 'so', 'because']):
                logical_flow += 1
                evidence.append(f"Good logical flow from step {i+1} to step {i+2}")
        
        # Calculate overall consistency
        total_comparisons = len(steps) * (len(steps) - 1) // 2
        contradiction_ratio = len(contradictions) / max(1, total_comparisons)
        
        confidence = 1.0 - contradiction_ratio
        confidence += (logical_flow / max(1, len(steps) - 1)) * 0.2  # Bonus for flow
        confidence = max(0.0, min(1.0, confidence))
        
        verified = confidence > 0.6 and len(contradictions) == 0
        
        explanation = f"Consistency check: {len(contradictions)} contradictions, {logical_flow} logical connections"
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            explanation=explanation
        )


class NLIVerifier(Verifier):
    """
    Natural Language Inference verifier.
    Template for integration with NLI models like RoBERTa.
    """
    
    def __init__(self, model_name: str = "roberta-large-mnli"):
        self.model_name = model_name
        # In a real implementation, you would load the model here:
        # from transformers import pipeline
        # self.nli_pipeline = pipeline("text-classification", model=model_name)
    
    def verify_step(self, step: str, context: str, 
                   node: Optional[ReasoningNode] = None) -> VerificationResult:
        """Verify step using NLI model."""
        # Template implementation - would use actual NLI model
        
        # Format for NLI: premise (context) and hypothesis (step)
        premise = context
        hypothesis = step
        
        # In real implementation:
        # result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
        # entailment_score = next(r['score'] for r in result if r['label'] == 'ENTAILMENT')
        
        # Mock implementation
        entailment_score = self._mock_nli_score(premise, hypothesis)
        
        verified = entailment_score > 0.5
        confidence = entailment_score
        
        evidence = []
        contradictions = []
        
        if entailment_score > 0.7:
            evidence.append("Strong entailment from context")
        elif entailment_score < 0.3:
            contradictions.append("Weak or contradictory relationship to context")
        
        explanation = f"NLI entailment score: {entailment_score:.3f}"
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            explanation=explanation
        )
    
    def verify_consistency(self, steps: List[str]) -> VerificationResult:
        """Verify consistency using pairwise NLI."""
        if len(steps) < 2:
            return VerificationResult(verified=True, confidence=1.0)
        
        contradictions = []
        evidence = []
        consistency_scores = []
        
        # Check pairwise consistency
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps[i+1:], i+1):
                score = self._mock_nli_score(step1, step2)
                consistency_scores.append(score)
                
                if score < 0.2:  # Strong contradiction
                    contradictions.append(
                        f"Contradiction between step {i+1} and step {j+1} (score: {score:.3f})"
                    )
                elif score > 0.8:  # Strong consistency
                    evidence.append(
                        f"Strong consistency between step {i+1} and step {j+1} (score: {score:.3f})"
                    )
        
        # Overall consistency
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        verified = avg_consistency > 0.5 and len(contradictions) == 0
        
        explanation = f"Average pairwise consistency: {avg_consistency:.3f}"
        
        return VerificationResult(
            verified=verified,
            confidence=avg_consistency,
            evidence=evidence,
            contradictions=contradictions,
            explanation=explanation
        )
    
    def _mock_nli_score(self, premise: str, hypothesis: str) -> float:
        """Mock NLI scoring for template purposes."""
        # Simple heuristic based on word overlap and length
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        
        overlap = len(premise_words.intersection(hypothesis_words))
        union = len(premise_words.union(hypothesis_words))
        
        if union == 0:
            return 0.5
        
        jaccard = overlap / union
        
        # Adjust based on some heuristics
        if "not" in hypothesis.lower() and "not" not in premise.lower():
            jaccard *= 0.3  # Likely contradiction
        elif any(word in hypothesis.lower() for word in ["therefore", "thus", "so"]):
            jaccard *= 1.2  # Likely entailment
        
        return max(0.0, min(1.0, jaccard))


class ToolCallVerifier(Verifier):
    """
    Verifier that can make tool calls to check facts and calculations.
    """
    
    def __init__(self):
        self.tool_registry: Dict[str, Any] = {}
        self.tool_call_history: List[Dict[str, Any]] = []
    
    def register_tool(self, name: str, tool_func: callable) -> None:
        """Register a tool function for verification."""
        self.tool_registry[name] = tool_func
    
    def verify_step(self, step: str, context: str, 
                   node: Optional[ReasoningNode] = None) -> VerificationResult:
        """Verify step potentially using tool calls."""
        evidence = []
        contradictions = []
        tool_calls = []
        
        # Look for mathematical expressions to verify
        math_pattern = r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
        math_matches = re.findall(math_pattern, step)
        
        verified = True
        confidence = 0.7  # Base confidence
        
        for match in math_matches:
            a, op, b, result = match
            a, b, result = float(a), float(b), float(result)
            
            # Calculate correct result
            if op == '+':
                correct = a + b
            elif op == '-':
                correct = a - b
            elif op == '*':
                correct = a * b
            elif op == '/':
                correct = a / b if b != 0 else float('inf')
            else:
                continue
            
            tool_call = {
                'tool': 'calculator',
                'input': f"{a} {op} {b}",
                'expected': result,
                'actual': correct,
                'correct': abs(correct - result) < 0.001
            }
            tool_calls.append(tool_call)
            
            if tool_call['correct']:
                evidence.append(f"Calculation verified: {a} {op} {b} = {result}")
                confidence += 0.1
            else:
                contradictions.append(f"Calculation error: {a} {op} {b} = {correct}, not {result}")
                confidence -= 0.3
                verified = False
        
        # Record tool calls if node provided
        if node and tool_calls:
            node.metadata.tool_calls.extend(tool_calls)
        
        # Store in history
        self.tool_call_history.extend(tool_calls)
        
        confidence = max(0.0, min(1.0, confidence))
        
        explanation = f"Tool verification: {len(tool_calls)} tool calls, {len(evidence)} confirmations"
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            explanation=explanation
        )
    
    def verify_consistency(self, steps: List[str]) -> VerificationResult:
        """Verify consistency using available tools."""
        # For now, just verify each step individually and aggregate
        all_evidence = []
        all_contradictions = []
        confidences = []
        
        for step in steps:
            result = self.verify_step(step, "")
            all_evidence.extend(result.evidence)
            all_contradictions.extend(result.contradictions)
            confidences.append(result.confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        verified = len(all_contradictions) == 0 and avg_confidence > 0.5
        
        explanation = f"Tool-based consistency check across {len(steps)} steps"
        
        return VerificationResult(
            verified=verified,
            confidence=avg_confidence,
            evidence=all_evidence,
            contradictions=all_contradictions,
            explanation=explanation
        )


# Factory functions
def create_mock_verifier(seed: Optional[int] = None) -> Verifier:
    """Create a mock verifier for testing."""
    return MockVerifier(seed=seed)


def create_nli_verifier(model_name: str = "roberta-large-mnli") -> Verifier:
    """Create an NLI-based verifier."""
    return NLIVerifier(model_name)


def create_tool_verifier() -> ToolCallVerifier:
    """Create a tool-call based verifier."""
    return ToolCallVerifier()
