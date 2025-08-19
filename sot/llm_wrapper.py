"""
Small wrapper for LLM interactions: generate steps and score steps.
Includes both mock implementations and real LLM wrapper patterns.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import random
import time

from .models import ReasoningNode


class LLMWrapper(ABC):
    """Abstract base class for LLM interactions."""
    
    @abstractmethod
    def generate_steps(self, context: str, max_steps: int = 3, 
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """
        Generate next reasoning steps given context.
        
        Args:
            context: The reasoning context/history
            max_steps: Maximum number of steps to generate
            temperature: Sampling temperature
            **kwargs: Additional LLM parameters
        
        Returns:
            List of generated reasoning step texts
        """
        pass
    
    @abstractmethod
    def score_step(self, step: str, context: str, **kwargs) -> float:
        """
        Score the quality of a reasoning step.
        
        Args:
            step: The reasoning step to score
            context: The context in which the step appears
            **kwargs: Additional parameters
        
        Returns:
            Score between 0.0 and 1.0
        """
        pass


class MockLLMWrapper(LLMWrapper):
    """
    Mock LLM implementation for testing and development.
    Generates reasonable-looking steps based on simple heuristics.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        # Templates for different types of reasoning
        self.templates = {
            'causal': [
                "This happened because {cause}",
                "The underlying cause is {cause}",
                "We can trace this back to {cause}"
            ],
            'analytical': [
                "Let's analyze {aspect}",
                "Consider the {aspect} perspective",
                "Breaking down {aspect} reveals"
            ],
            'evidential': [
                "The evidence suggests {conclusion}",
                "Based on available data, {conclusion}",
                "Research indicates {conclusion}"
            ],
            'inferential': [
                "This implies that {inference}",
                "We can conclude {inference}",
                "It follows that {inference}"
            ]
        }
        
        self.aspects = ["structural", "functional", "environmental", "temporal", "causal"]
        self.causes = ["material failure", "external force", "design flaw", "wear and tear"]
        self.conclusions = ["multiple factors are involved", "a single cause is likely", "further investigation is needed"]
        self.inferences = ["the problem is solvable", "prevention is possible", "similar cases exist"]
    
    def generate_steps(self, context: str, max_steps: int = 3, 
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """Generate mock reasoning steps."""
        steps = []
        
        # Adjust randomness based on temperature
        num_steps = min(max_steps, random.randint(1, max_steps))
        
        for i in range(num_steps):
            step_type = random.choice(list(self.templates.keys()))
            template = random.choice(self.templates[step_type])
            
            # Fill in template based on type
            if step_type == 'causal':
                cause = random.choice(self.causes)
                step = template.format(cause=cause)
            elif step_type == 'analytical':
                aspect = random.choice(self.aspects)
                step = template.format(aspect=aspect)
            elif step_type == 'evidential':
                conclusion = random.choice(self.conclusions)
                step = template.format(conclusion=conclusion)
            elif step_type == 'inferential':
                inference = random.choice(self.inferences)
                step = template.format(inference=inference)
            else:
                step = "Let's continue the reasoning process"
            
            # Add some context-aware modifications
            if "glass" in context.lower():
                step = step.replace("material failure", "glass fracture")
                step = step.replace("external force", "impact force")
            elif "problem" in context.lower():
                step = step.replace("This happened", "The problem occurred")
            
            steps.append(step)
        
        return steps
    
    def score_step(self, step: str, context: str, **kwargs) -> float:
        """Score a reasoning step using simple heuristics."""
        score = 0.5  # baseline
        
        step_lower = step.lower()
        context_lower = context.lower()
        
        # Bonus for specific terminology
        specific_terms = ['because', 'therefore', 'analysis', 'evidence', 'data', 
                         'research', 'study', 'indicates', 'suggests', 'implies']
        for term in specific_terms:
            if term in step_lower:
                score += 0.1
        
        # Bonus for contextual relevance
        context_words = set(context_lower.split())
        step_words = set(step_lower.split())
        overlap = len(context_words.intersection(step_words))
        if overlap > 2:
            score += 0.15
        
        # Penalty for vague language
        vague_terms = ['maybe', 'possibly', 'might', 'could be', 'perhaps', 'unclear']
        for term in vague_terms:
            if term in step_lower:
                score -= 0.1
        
        # Bonus for logical connectors
        logical_terms = ['thus', 'consequently', 'as a result', 'this leads to']
        for term in logical_terms:
            if term in step_lower:
                score += 0.15
        
        # Length bonus (longer steps often more detailed)
        if len(step) > 50:
            score += 0.1
        
        return max(0.0, min(1.0, score))


class OpenAIWrapper(LLMWrapper):
    """
    Wrapper for OpenAI GPT models.
    This is a template - you'll need to implement with actual API calls.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **default_kwargs):
        self.api_key = api_key
        self.model = model
        self.default_kwargs = default_kwargs
        
        # You would initialize the OpenAI client here
        # import openai
        # self.client = openai.Client(api_key=api_key)
    
    def generate_steps(self, context: str, max_steps: int = 3, 
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """Generate steps using OpenAI API."""
        # Merge kwargs with defaults
        params = {**self.default_kwargs, **kwargs}
        params['temperature'] = temperature
        
        prompt = self._build_generation_prompt(context, max_steps)
        
        # Example API call structure (you'll need to implement this):
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     **params
        # )
        # 
        # # Parse response to extract steps
        # steps = self._parse_steps_response(response.choices[0].message.content)
        # return steps[:max_steps]
        
        # Placeholder implementation
        raise NotImplementedError("Implement with actual OpenAI API calls")
    
    def score_step(self, step: str, context: str, **kwargs) -> float:
        """Score a step using OpenAI API."""
        prompt = self._build_scoring_prompt(step, context)
        
        # Similar API call pattern as above
        # You would implement the actual API call here
        
        raise NotImplementedError("Implement with actual OpenAI API calls")
    
    def _build_generation_prompt(self, context: str, max_steps: int) -> str:
        """Build prompt for step generation."""
        return f"""Given the following reasoning context, generate up to {max_steps} logical next steps:

Context:
{context}

Generate reasoning steps that:
1. Build logically on the context
2. Are specific and actionable
3. Move toward a conclusion
4. Are clearly distinct from each other

Steps:"""
    
    def _build_scoring_prompt(self, step: str, context: str) -> str:
        """Build prompt for step scoring."""
        return f"""Score the following reasoning step on a scale from 0.0 to 1.0:

Context:
{context}

Step to score:
{step}

Scoring criteria:
- Logical consistency (0.3)
- Specificity and clarity (0.3) 
- Relevance to context (0.2)
- Potential to advance reasoning (0.2)

Provide only a numeric score between 0.0 and 1.0:"""


class AnthropicWrapper(LLMWrapper):
    """
    Wrapper for Anthropic Claude models.
    Template implementation - needs actual API integration.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet", **default_kwargs):
        self.api_key = api_key
        self.model = model
        self.default_kwargs = default_kwargs
    
    def generate_steps(self, context: str, max_steps: int = 3, 
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """Generate steps using Anthropic API."""
        raise NotImplementedError("Implement with actual Anthropic API calls")
    
    def score_step(self, step: str, context: str, **kwargs) -> float:
        """Score a step using Anthropic API."""
        raise NotImplementedError("Implement with actual Anthropic API calls")


class LLMWrapperWithCaching(LLMWrapper):
    """
    Decorator wrapper that adds caching to any LLM wrapper.
    Useful for avoiding repeated API calls during development.
    """
    
    def __init__(self, base_wrapper: LLMWrapper, cache_size: int = 1000):
        self.base_wrapper = base_wrapper
        self.cache_size = cache_size
        self._generation_cache: Dict[str, List[str]] = {}
        self._scoring_cache: Dict[str, float] = {}
    
    def generate_steps(self, context: str, max_steps: int = 3, 
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """Generate steps with caching."""
        # Create cache key
        cache_key = f"{context}|{max_steps}|{temperature}|{sorted(kwargs.items())}"
        
        if cache_key in self._generation_cache:
            return self._generation_cache[cache_key]
        
        # Generate and cache
        steps = self.base_wrapper.generate_steps(context, max_steps, temperature, **kwargs)
        
        # Manage cache size
        if len(self._generation_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._generation_cache))
            del self._generation_cache[oldest_key]
        
        self._generation_cache[cache_key] = steps
        return steps
    
    def score_step(self, step: str, context: str, **kwargs) -> float:
        """Score step with caching."""
        cache_key = f"{step}|{context}|{sorted(kwargs.items())}"
        
        if cache_key in self._scoring_cache:
            return self._scoring_cache[cache_key]
        
        # Score and cache
        score = self.base_wrapper.score_step(step, context, **kwargs)
        
        # Manage cache size
        if len(self._scoring_cache) >= self.cache_size:
            oldest_key = next(iter(self._scoring_cache))
            del self._scoring_cache[oldest_key]
        
        self._scoring_cache[cache_key] = score
        return score
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._generation_cache.clear()
        self._scoring_cache.clear()


class LLMWrapperWithRetry(LLMWrapper):
    """
    Decorator wrapper that adds retry logic for API failures.
    """
    
    def __init__(self, base_wrapper: LLMWrapper, max_retries: int = 3, 
                 retry_delay: float = 1.0):
        self.base_wrapper = base_wrapper
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate_steps(self, context: str, max_steps: int = 3, 
                      temperature: float = 0.7, **kwargs) -> List[str]:
        """Generate steps with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.base_wrapper.generate_steps(context, max_steps, temperature, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        return []  # Fallback
    
    def score_step(self, step: str, context: str, **kwargs) -> float:
        """Score step with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.base_wrapper.score_step(step, context, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                time.sleep(self.retry_delay * (2 ** attempt))
        
        return 0.5  # Fallback score


# Factory functions for common configurations
def create_mock_llm(seed: Optional[int] = None) -> LLMWrapper:
    """Create a mock LLM wrapper for testing."""
    return MockLLMWrapper(seed=seed)


def create_cached_llm(base_wrapper: LLMWrapper, cache_size: int = 1000) -> LLMWrapper:
    """Create a cached version of an LLM wrapper."""
    return LLMWrapperWithCaching(base_wrapper, cache_size)


def create_robust_llm(base_wrapper: LLMWrapper, max_retries: int = 3) -> LLMWrapper:
    """Create a robust version with retry logic."""
    return LLMWrapperWithRetry(base_wrapper, max_retries)
