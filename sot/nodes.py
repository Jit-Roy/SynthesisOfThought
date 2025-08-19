"""
ReasoningNode implementation for Synthesis-of-Thought (SoT).
Contains robust serialization, history helpers, and small utilities.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import json
import time


class NodeStatus(Enum):
    """Status of a node in the reasoning tree."""
    ACTIVE = "active"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"
    PRUNED = "pruned"

@dataclass
class NodeMetadata:
    """Metadata container for additional node information used during SoT."""
    verified: bool = False                          
    # True if this step has been externally verified (e.g., tool check, evidence lookup)
    evidence: List[str] = field(default_factory=list)      
    # External evidence strings/IDs that support the step (URLs, doc ids, snippets)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  
    # List of tool-call records used to produce/verify this step.
    # Each entry example: {"tool": "calculator", "input": "2+2", "output": "4", "ok": True, "call_id": "..."}
    contradictions: List[str] = field(default_factory=list) 
    # Short descriptions or IDs of contradictions found between this step and other steps or facts.
    # Example: ["contradicts node: <node_id>", "NLI: contradiction with prior claim"]
    flags: Dict[str, Any] = field(default_factory=dict)      
    # Arbitrary small-key metadata for quick checks/heuristics.
    # Examples: {"needs_expansion": True, "is_assumption": True, "confidence_source": "llm_score"}


@dataclass
class SamplingInfo:
    """Information about how this step was generated."""
    temperature: float = 0.7
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: time.time())


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
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    children: List[str] = field(default_factory=list)
    tokens_used: Optional[int] = None
    origin: SamplingInfo = field(default_factory=SamplingInfo)
    consensus_count: int = 1
    cluster_id: Optional[str] = None
    verifier_score: Optional[float] = None
    is_terminal: bool = False
    status: NodeStatus = NodeStatus.ACTIVE
    probability: float = 1.0
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
        self.is_terminal = True
        self.status = NodeStatus.TERMINAL_SUCCESS if success else NodeStatus.TERMINAL_FAILURE

    def mark_pruned(self) -> None:
        self.status = NodeStatus.PRUNED

    def update_score(self, new_score: float) -> None:
        self.score = max(0.0, min(1.0, float(new_score)))

    def add_evidence(self, evidence: str) -> None:
        self.metadata.evidence.append(evidence)

    def add_contradiction(self, contradiction: str) -> None:
        self.metadata.contradictions.append(contradiction)

    def set_verified(self, verified: bool = True) -> None:
        self.metadata.verified = verified

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

    def get_history_nodes(self, node_store: Dict[str, "ReasoningNode"]) -> List["ReasoningNode"]:
        """Return the list of node objects from root -> this node."""
        nodes = []
        cur = self
        while cur is not None:
            nodes.append(cur)
            if cur.parent is None:
                break
            cur = node_store.get(cur.parent)
        return list(reversed(nodes))

    # --- Serialization helpers ---
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to a JSON-serializable dict.
        Ensures enums and nested dataclasses are converted to primitives.
        """
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            if is_dataclass(obj):
                return {k: convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        base = asdict(self)
        # Replace status enum with its value
        base['status'] = self.status.value if isinstance(self.status, Enum) else self.status
        return convert(base)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningNode":
        """
        Construct a ReasoningNode from a dict (reverse of to_dict).
        Accepts either native dataclasses or plain dicts for nested fields.
        """
        data = dict(data)  # copy to avoid modifying caller
        # Status
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = NodeStatus(data['status'])
        # Metadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = NodeMetadata(**data['metadata'])
        # Origin / sampling info
        if 'origin' in data and isinstance(data['origin'], dict):
            data['origin'] = SamplingInfo(**data['origin'])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningNode":
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReasoningNode):
            return NotImplemented
        return self.id == other.id


# Helper functions to create nodes
def create_root_node(question: str) -> ReasoningNode:
    return ReasoningNode(
        step_text=f"Q: {question}",
        depth=0,
        score=1.0,
        is_terminal=False,
        probability=1.0,
        parent=None
    )


def create_child_node(parent: ReasoningNode, step_text: str, score: float = 0.5, probability: float = 1.0) -> ReasoningNode:
    child = ReasoningNode(
        step_text=step_text,
        parent=parent.id,
        depth=parent.depth + 1,
        score=score,
        probability=probability
    )
    parent.add_child(child.id)
    return child
