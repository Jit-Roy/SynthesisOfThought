"""
Simple LLM wrapper for generating and scoring reasoning steps using Gemini API.
"""

import google.generativeai as genai
from typing import List, Dict, Any
import time
import re

class LLM:
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash-lite"):
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
        
        Returns:
            List of generated reasoning step texts (may be empty if generation fails)
        """
        
        prompt = f"""You are helping with step-by-step reasoning. Given this context:

{context}

Please provide exactly 3 logical next steps to continue this reasoning process. Format your response as a numbered list:

1. [First step]
2. [Second step] 
3. [Third step]

Make each step clear and actionable."""

        for attempt in range(3):  # up to 3 retries
            try:
                print(f"🔄 LLM Attempt {attempt + 1}: Sending prompt...")
                response = self.model.generate_content(prompt)
                
                # Check for safety filtering
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason
                    
                    if finish_reason == 2:  # SAFETY
                        print(f"⚠️ Content filtered for safety reasons, trying simpler prompt...")
                        # Try with a much simpler, safer prompt
                        simple_prompt = f"List 3 steps to solve: {context.replace('Question:', '').strip()}"
                        response = self.model.generate_content(simple_prompt)
                        candidate = response.candidates[0] if response.candidates else None
                        finish_reason = candidate.finish_reason if candidate else None
                
                # Try to access response text safely
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        response_text = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                        if response_text:
                            print(f"📝 LLM Response received: {len(response_text)} characters")
                            print(f"🔍 Raw response: {response_text[:200]}...")
                            steps = self._parse_steps(response_text)
                            print(f"✅ Parsed {len(steps)} steps from response")
                            return steps
                
                print(f"⚠️ No valid content in response (finish_reason: {finish_reason if 'finish_reason' in locals() else 'unknown'})")
                    
            except Exception as e:
                print(f"❌ LLM Error on attempt {attempt + 1}: {e}")
                if attempt < 2 and len(self.api_keys) > 1:
                    print(f"🔄 Rotating API key...")
                    self._rotate_api_key()
                    time.sleep(1)
        
        print(f"❌ All LLM attempts failed, returning empty list")
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
        
        print(f"🔍 Parsing {len(lines)} lines from response...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(('-', '*', '•'))):
                # Remove numbering
                step = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip()
                if step and len(step) > 5:  # Reduced minimum length
                    steps.append(step)
                    print(f"  ✅ Parsed step {len(steps)}: {step[:50]}...")
        
        # Fallback parsing - split by sentences
        if not steps and response:
            print(f"🔄 Fallback parsing: splitting by sentences...")
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
            steps = sentences[:3]  # Take first 3 sentences
            for i, step in enumerate(steps):
                print(f"  📝 Fallback step {i+1}: {step[:50]}...")
        
        # If still no steps, return empty list
        if not steps:
            print(f"🔄 No steps could be parsed from response")
        
        print(f"✅ Final result: {len(steps)} steps parsed")
        return steps[:3]  # Return max 3 steps
    
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

    def score_step(self, step: str, context: str) -> float:
        """
        Score a reasoning step quality (0.0 to 1.0).
        
        Args:
            step: The reasoning step to score
            context: The context/history for scoring
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        prompt = f"""Rate the quality of this reasoning step on a scale of 0.0 to 1.0:

Context: {context}
Step: {step}

Consider:
- Logical consistency
- Relevance to the problem
- Clarity and precision
- Builds on previous context

Respond with just a number between 0.0 and 1.0:"""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                # Extract number from response
                import re
                numbers = re.findall(r'0\.\d+|1\.0|0\.0', response.text)
                if numbers:
                    return float(numbers[0])
        except Exception as e:
            pass
        
        # Fallback scoring based on step length and basic heuristics
        if len(step.strip()) < 10:
            return 0.3
        elif len(step.strip()) > 200:
            return 0.6
        else:
            return 0.7

    def get_current_key_info(self) -> Dict[str, Any]:
        """Get information about current API key."""
        return {
            'current_index': self.current_key_index,
            'total_keys': len(self.api_keys),
            'model_name': self.model_name
        }