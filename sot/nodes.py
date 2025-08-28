"""
ReasoningNode implementation for Synthesis-of-Thought (SoT).
Contains robust serialization, history helpers, and small utilities.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import uuid


class NodeStatus(Enum):
    """Status values for reasoning nodes."""
    ACTIVE = "active"
    PRUNED = "pruned"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"


@dataclass
class Node:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent: Optional[str] = None
    children: Optional[List[str]] = field(default_factory=list)
    status: NodeStatus = NodeStatus.ACTIVE
    step_text: str = ""
    depth: int = 0
    history: str = ""  

    # --- Basic child management ---
    def add_child(self, child_id: str, node_store: Dict[str, "Node"], summarizer=None) -> None:
        if child_id not in self.children:
            self.children.append(child_id)
        child = node_store.get(child_id)
        if child is not None:
            child.parent = self.id  # ensure back-link
            child.update_history(node_store, summarizer)

    def remove_child(self, child_id: str) -> None:
        if child_id in self.children:
            self.children.remove(child_id)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def mark_terminal(self, success: bool = True) -> None:
        self.status = NodeStatus.TERMINAL_SUCCESS if success else NodeStatus.TERMINAL_FAILURE

    def mark_pruned(self) -> None:
        self.status = NodeStatus.PRUNED

    # --- History / trace helpers ---
    def trace(self, node_store: Dict[str, "Node"]) -> List[str]:
        path_texts = []
        visited = set()
        cur = self
        while cur is not None and cur.id not in visited:
            visited.add(cur.id)
            path_texts.append(cur.step_text)
            if cur.parent is None:
                break
            cur = node_store.get(cur.parent)
        return list(reversed(path_texts))
    
    def update_history(self, node_store: Dict[str, "Node"], summarizer=None) -> None:
        """
        Refresh the node's history as a simple string.
        - If summarizer is None: join full path with " → ".
        - If summarizer is provided: summary + last few steps as a string.
        """
        full_path = self.trace(node_store)
        if summarizer is None:
            self.history = " → ".join(full_path)
        else:
            try:
                summary = summarizer(full_path[:-3]) if len(full_path) > 3 else None
                recent = full_path[-3:]
                if summary:
                    self.history = summary + " → " + " → ".join(recent)
                else:
                    self.history = " → ".join(recent)
            except Exception:
                # fallback to full path
                self.history = " → ".join(full_path)