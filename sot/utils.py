"""
Utility functions for hashing, deduplication, normalization, logging helpers.
Common utilities used throughout the Synthesis of Thought system.
"""

import hashlib
import logging
import re
import time
from functools import wraps
from typing import Any, Dict, List, Set, Optional, Callable, Union, TypeVar
from dataclasses import asdict, is_dataclass
import json


# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable)


# =============================================================================
# Hashing Utilities
# =============================================================================

def hash_dict(data: Dict[str, Any]) -> str:
    """
    Create a deterministic hash of a dictionary.
    
    Args:
        data: Dictionary to hash
        
    Returns:
        Hexadecimal hash string
    """
    # Sort keys for deterministic ordering
    sorted_items = sorted(data.items())
    json_str = json.dumps(sorted_items, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()


def hash_object(obj: Any) -> str:
    """
    Create a hash of any object by converting to string representation.
    
    Args:
        obj: Object to hash
        
    Returns:
        Hexadecimal hash string
    """
    if is_dataclass(obj):
        obj_dict = asdict(obj)
        return hash_dict(obj_dict)
    elif isinstance(obj, dict):
        return hash_dict(obj)
    elif isinstance(obj, (list, tuple)):
        # Convert to string representation
        str_repr = str(sorted(obj) if all(isinstance(x, (str, int, float)) for x in obj) else obj)
        return hashlib.md5(str_repr.encode('utf-8')).hexdigest()
    else:
        # Fallback to string representation
        str_repr = str(obj)
        return hashlib.md5(str_repr.encode('utf-8')).hexdigest()


def content_hash(text: str) -> str:
    """
    Create a hash of text content, normalizing whitespace.
    
    Args:
        text: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', text.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


# =============================================================================
# Deduplication Utilities
# =============================================================================

def deduplicate_by_content(items: List[str], 
                          similarity_threshold: float = 0.9) -> List[str]:
    """
    Remove duplicate items based on content similarity.
    
    Args:
        items: List of text items
        similarity_threshold: Threshold for considering items similar
        
    Returns:
        Deduplicated list
    """
    if not items:
        return []
    
    # Simple hash-based deduplication for exact matches
    seen_hashes = set()
    unique_items = []
    
    for item in items:
        item_hash = content_hash(item)
        if item_hash not in seen_hashes:
            seen_hashes.add(item_hash)
            unique_items.append(item)
    
    # For semantic similarity, would need more sophisticated methods
    # This is a simplified version focusing on exact content matches
    return unique_items


def deduplicate_dicts(items: List[Dict[str, Any]], 
                     key_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Remove duplicate dictionaries based on specified key fields or all fields.
    
    Args:
        items: List of dictionaries
        key_fields: Fields to use for comparison (None = all fields)
        
    Returns:
        Deduplicated list
    """
    if not items:
        return []
    
    seen_hashes = set()
    unique_items = []
    
    for item in items:
        if key_fields:
            # Only use specified fields for comparison
            key_data = {k: item.get(k) for k in key_fields if k in item}
        else:
            # Use all fields
            key_data = item
        
        item_hash = hash_dict(key_data)
        if item_hash not in seen_hashes:
            seen_hashes.add(item_hash)
            unique_items.append(item)
    
    return unique_items


def remove_duplicates_preserve_order(items: List[T]) -> List[T]:
    """
    Remove duplicates while preserving order.
    
    Args:
        items: List of items
        
    Returns:
        List with duplicates removed, order preserved
    """
    seen = set()
    result = []
    
    for item in items:
        # Handle unhashable types
        try:
            if item not in seen:
                seen.add(item)
                result.append(item)
        except TypeError:
            # For unhashable types, use string representation
            item_str = str(item)
            if item_str not in seen:
                seen.add(item_str)
                result.append(item)
    
    return result


# =============================================================================
# Text Normalization Utilities
# =============================================================================

def normalize_text(text: str, 
                  lowercase: bool = True,
                  remove_extra_whitespace: bool = True,
                  remove_punctuation: bool = False) -> str:
    """
    Normalize text for consistent processing.
    
    Args:
        text: Text to normalize
        lowercase: Convert to lowercase
        remove_extra_whitespace: Remove extra whitespace
        remove_punctuation: Remove punctuation
        
    Returns:
        Normalized text
    """
    result = text
    
    if lowercase:
        result = result.lower()
    
    if remove_extra_whitespace:
        result = re.sub(r'\s+', ' ', result.strip())
    
    if remove_punctuation:
        result = re.sub(r'[^\w\s]', '', result)
    
    return result


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction
    normalized = normalize_text(text, lowercase=True, remove_punctuation=True)
    words = normalized.split()
    
    # Filter by length and remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    return remove_duplicates_preserve_order(keywords)


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple text similarity based on word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(normalize_text(text1, remove_punctuation=True).split())
    words2 = set(normalize_text(text2, remove_punctuation=True).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


# =============================================================================
# Logging Utilities
# =============================================================================

def setup_logger(name: str, 
                level: str = "INFO",
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
                raise
        return wrapper
    return decorator


def log_function_calls(logger: logging.Logger, log_args: bool = False):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance
        log_args: Whether to log function arguments
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if log_args:
                logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            else:
                logger.debug(f"Calling {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


class ProgressLogger:
    """Simple progress logger for long-running operations."""
    
    def __init__(self, logger: logging.Logger, total: int, name: str = "Operation"):
        self.logger = logger
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def update(self, increment: int = 1):
        """Update progress and log if needed."""
        self.current += increment
        current_time = time.time()
        
        # Log every 10% progress or every 10 seconds
        progress_pct = (self.current / self.total) * 100
        time_since_last = current_time - self.last_log_time
        
        if progress_pct % 10 < (increment / self.total) * 100 or time_since_last > 10:
            elapsed = current_time - self.start_time
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                self.logger.info(f"{self.name}: {self.current}/{self.total} ({progress_pct:.1f}%) "
                               f"- ETA: {eta:.1f}s")
            self.last_log_time = current_time
    
    def finish(self):
        """Log completion."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} completed: {self.total} items in {elapsed:.2f}s")


# =============================================================================
# Data Structure Utilities
# =============================================================================

def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        data: Dictionary to flatten
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    def _flatten(obj, parent_key='', sep='.'):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data, sep=separator)


def safe_get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get a value from a nested dictionary using dot notation.
    
    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., 'a.b.c')
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def merge_dicts_deep(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    
    return result


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_probability(value: float, name: str = "probability") -> float:
    """
    Validate that a value is a valid probability (0-1).
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is not a valid probability
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")
    
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    return float(value)


def validate_positive_int(value: int, name: str = "value") -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is not a positive integer
    """
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value)}")
    
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    
    return value


def validate_non_empty_string(value: str, name: str = "string") -> str:
    """
    Validate that a value is a non-empty string.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is not a non-empty string
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value)}")
    
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    
    return value


# =============================================================================
# Math and Statistics Utilities
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value for division by zero
        
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to sum to 1.0.
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    total = sum(scores)
    if total == 0:
        # Equal distribution if all scores are zero
        return [1.0 / len(scores)] * len(scores)
    
    return [score / total for score in scores]


def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Compute weighted average of values.
    
    Args:
        values: List of values
        weights: List of weights
        
    Returns:
        Weighted average
        
    Raises:
        ValueError: If lists have different lengths
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    if not values:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    
    return safe_divide(weighted_sum, weight_sum, 0.0)


# =============================================================================
# File and Path Utilities
# =============================================================================

def ensure_directory_exists(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    import os
    os.makedirs(path, exist_ok=True)


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe = re.sub(r'\s+', '_', safe)  # Replace spaces with underscores
    safe = safe.strip('._')  # Remove leading/trailing dots and underscores
    
    # Ensure not empty
    return safe if safe else 'untitled'
