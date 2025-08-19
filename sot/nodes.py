"""
ReasoningNode implementation for Synthesis-of-Thought (SoT).
Contains robust serialization, history helpers, and small utilities.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import uuid
import json
import time

@dataclass
class ReasoningNode:
    """
    A node in the reasoning tree representing a single reasoning step.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_text: str = ""
    parent: Optional[str] = None
    depth: int = 0
    score: float = 0.5
    contradictions: List[str] = field(default_factory=list)  
    children: List[str] = field(default_factory=list)
    tokens_used: Optional[int] = None
    consensus_count: int = 1
    verifier_score: Optional[float] = None
    status: Literal["active", "pruned", "terminal_success", "terminal_failure"] = "active"
    _history_cache: Optional[List[str]] = field(default=None, repr=False)

    # --- Basic child management ---
    def add_child(self, child_id: str) -> None:
        if child_id not in self.children:
            self.children.append(child_id)

    def remove_child(self, child_id: str) -> None:
        if child_id in self.children:
            self.children.remove(child_id)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def mark_terminal(self, success: bool = True) -> None:
        self.status = "terminal_success" if success else "terminal_failure"

    def mark_pruned(self) -> None:
        self.status = "pruned"

    def update_score(self, new_score: float) -> None:
        self.score = max(0.0, min(1.0, float(new_score)))

    def add_contradiction(self, contradiction: str) -> None:
        if contradiction not in self.contradictions:
            self.contradictions.append(contradiction)

    # --- History / trace helpers ---
    def trace(self, node_store: Dict[str, "ReasoningNode"]) -> List[str]:
        """
        Return the list of step_text strings from root -> this node.
        Requires node_store: mapping of id -> ReasoningNode.
        """
        path_texts = []
        cur = self
        # Walk up to root
        while cur is not None:
            path_texts.append(cur.step_text)
            if cur.parent is None:
                break
            cur = node_store.get(cur.parent)
        return list(reversed(path_texts))


# Helper functions to create nodes
def create_root_node(question: str) -> ReasoningNode:
    return ReasoningNode(
        step_text=f"Q: {question}",
        depth=0,
        score=1.0,
        parent=None
    )


def create_child_node(parent: ReasoningNode, step_text: str, score: float = 0.5) -> ReasoningNode:
    child = ReasoningNode(
        step_text=step_text,
        parent=parent.id,
        depth=parent.depth + 1,
        score=score
    )
    parent.add_child(child.id)
    return child
