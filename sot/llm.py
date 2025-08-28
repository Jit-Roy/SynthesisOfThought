"""
Simple LLM wrapper for generating and scoring reasoning steps using Gemini API.
"""

import google.generativeai as genai
from typing import List
import time
import re

class LLM:
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        """
        Initialize the LLM wrapper with Gemini API.
        
        Args:
            api_keys: List of Gemini API keys for rotation
            model_name: Gemini model name
        """
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Setup Gemini API."""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 200,
            }
        )
    
    def _rotate_api_key(self):
        """Rotate to next API key if current one fails."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._setup_gemini()
    
    def generate_steps(self, context: str) -> List[str]:
        """
        Generate reasoning steps given context using Gemini API.
        
        Args:
            context: The reasoning context/history
            temperature: Sampling temperature (0.0 to 1.0)
        
        Returns:
            List of generated reasoning step texts (may be empty if generation fails)
        """
        
        prompt = f"""Generate logical next reasoning steps for this context:
        {context}
        Each step should be 1-2 sentences. Format as numbered list:
        1. [step]
        2. [step]
        3. [step]"""

        for attempt in range(3):  # up to 3 retries
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    steps = self._parse_steps(response.text)
                    return steps
            except Exception as e:
                if attempt < 2 and len(self.api_keys) > 1:
                    self._rotate_api_key()
                    time.sleep(1)
        return []

    def generate_branches(self, context: str, num_branches: int = 3) -> List[str]:
        """
        Generate alternative reasoning branches given context using Gemini API.
        
        Args:
            context: The reasoning context/history
            num_branches: Number of alternative branches to generate
            temperature: Sampling temperature (0.0 to 1.0)
        
        Returns:
            List of generated branch step texts (may be empty if generation fails)
        """
        
        prompt = f"""Generate {num_branches} alternative reasoning branches for this context:
        {context}
        Each branch should explore a different line of reasoning.
        Format as numbered list:
        1. [branch idea]
        2. [branch idea]
        3. [branch idea]"""

        for attempt in range(3):  # up to 3 retries
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    branches = self._parse_steps(response.text, num_branches)
                    return branches
            except Exception as e:
                if attempt < 2 and len(self.api_keys) > 1:
                    self._rotate_api_key()
                    time.sleep(1)
        return []

    def _parse_steps(self, response: str) -> List[str]:
        """Parse steps from API response."""
        steps = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(('-', '*', '•'))):
                # Remove numbering
                step = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip()
                if step and len(step) > 10:
                    steps.append(step)
        
        # Fallback parsing
        if not steps and response:
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
            steps = sentences
        
        # Ensure we have steps
        if not steps:
            steps = ["Continue analyzing the situation.", "Consider key factors."]
        
        return steps
    
    def _parse_branches(self, response: str, num_branches: int = 3) -> List[str]:
        """Parse alternative reasoning branches from API response."""
        branches = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(('-', '*', '•'))):
                # Remove numbering/bullets
                branch = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip()
                if branch and len(branch) > 10:
                    branches.append(branch)
                    if len(branches) >= num_branches:
                        break
        
        # Fallback parsing: split into sentences
        if not branches and response:
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
            branches = sentences[:num_branches]
        
        # Ensure max length
        return branches[:num_branches]
    
    def summarize_history(self, prev_history: str, new_step: str, max_summary_len: int = 150) -> str:
        """
        Incrementally summarize reasoning history by combining previous history with new step.

        Args:
            prev_history: The parent's history string (can be summary + last few steps).
            new_step: The current node's step text.
            max_summary_len: Maximum length of the summary string (characters).

        Returns:
            Updated summarized history string.
        """
        # If no previous history, just return the step
        if not prev_history:
            return new_step.strip()

        prompt = f"""You are tracking reasoning steps.
        Update the history summary by integrating the new step into the previous history.
        Keep the total summary concise (under {max_summary_len} characters if possible).
        Previous history:
        {prev_history}
        New step:
        {new_step}
        Updated history:"""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                summary = response.text.strip()
                return summary
        except Exception as e:
            pass

        # Fallback = append new step to prev_history
        fallback = (prev_history + " → " + new_step).strip()
        return fallback