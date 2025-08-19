"""
Node storage, persistence (JSON/JSONL), and snapshot helpers.
Manages the collection of ReasoningNodes with efficient access and serialization.
"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
from collections import defaultdict
import time

from .nodes import ReasoningNode, NodeStatus


class NodeStore:
    """
    Central storage for ReasoningNodes with persistence and querying capabilities.
    """
    
    def __init__(self):
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_id: Optional[str] = None
        
        # Index structures for efficient queries
        self._children_index: Dict[str, List[str]] = defaultdict(list)
        self._depth_index: Dict[int, List[str]] = defaultdict(list)
        self._status_index: Dict[NodeStatus, List[str]] = defaultdict(list)
        
        # Snapshots for rollback/checkpointing
        self._snapshots: Dict[str, Dict[str, Any]] = {}
    
    def add_node(self, node: ReasoningNode) -> None:
        """Add a node to the store and update indices."""
        self.nodes[node.id] = node
        
        # Update indices
        self._depth_index[node.depth].append(node.id)
        self._status_index[node.status].append(node.id)
        
        if node.parent:
            self._children_index[node.parent].append(node.id)
        
        # Set root if this is the first root node
        if node.is_root() and self.root_id is None:
            self.root_id = node.id
    
    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_root(self) -> Optional[ReasoningNode]:
        """Get the root node."""
        if self.root_id:
            return self.nodes.get(self.root_id)
        return None
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the store and update indices.
        Returns True if node was found and removed.
        """
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        # Remove from indices
        self._depth_index[node.depth].remove(node_id)
        self._status_index[node.status].remove(node_id)
        
        if node.parent:
            self._children_index[node.parent].remove(node_id)
        
        # Remove from store
        del self.nodes[node_id]
        
        # Clear root if this was the root
        if self.root_id == node_id:
            self.root_id = None
        
        return True
    
    def get_children(self, node_id: str) -> List[ReasoningNode]:
        """Get all children of a node."""
        child_ids = self._children_index.get(node_id, [])
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]
    
    def get_nodes_by_depth(self, depth: int) -> List[ReasoningNode]:
        """Get all nodes at a specific depth."""
        node_ids = self._depth_index.get(depth, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_nodes_by_status(self, status: NodeStatus) -> List[ReasoningNode]:
        """Get all nodes with a specific status."""
        node_ids = self._status_index.get(status, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_frontier_nodes(self) -> List[ReasoningNode]:
        """Get all active leaf nodes (frontier)."""
        frontier = []
        for node in self.nodes.values():
            if (node.status == NodeStatus.ACTIVE and 
                node.is_leaf() and 
                len(self.get_children(node.id)) == 0):
                frontier.append(node)
        return frontier
    
    def get_terminal_nodes(self) -> List[ReasoningNode]:
        """Get all terminal nodes."""
        terminal = []
        terminal.extend(self.get_nodes_by_status(NodeStatus.TERMINAL_SUCCESS))
        terminal.extend(self.get_nodes_by_status(NodeStatus.TERMINAL_FAILURE))
        return terminal
    
    def get_path_to_node(self, node_id: str) -> List[ReasoningNode]:
        """Get the complete path from root to the specified node."""
        node = self.get_node(node_id)
        if not node:
            return []
        
        return node.get_history_nodes(self.nodes)
    
    def get_all_paths(self) -> List[List[ReasoningNode]]:
        """Get all paths from root to leaf nodes."""
        paths = []
        
        # Get all leaf nodes (terminal + frontier)
        leaf_nodes = self.get_terminal_nodes() + self.get_frontier_nodes()
        
        for leaf in leaf_nodes:
            path = self.get_path_to_node(leaf.id)
            if path:
                paths.append(path)
        
        return paths
    
    def update_indices(self) -> None:
        """Rebuild all indices from scratch. Useful after bulk operations."""
        self._children_index.clear()
        self._depth_index.clear()
        self._status_index.clear()
        
        for node in self.nodes.values():
            self._depth_index[node.depth].append(node.id)
            self._status_index[node.status].append(node.id)
            
            if node.parent:
                self._children_index[node.parent].append(node.id)
    
    # === Persistence Methods ===
    
    def save_to_json(self, filepath: str) -> None:
        """Save the entire store to a JSON file."""
        data = {
            'root_id': self.root_id,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'metadata': {
                'total_nodes': len(self.nodes),
                'timestamp': time.time(),
                'max_depth': max((node.depth for node in self.nodes.values()), default=0)
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filepath: str) -> None:
        """Load the store from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.clear()
        self.root_id = data.get('root_id')
        
        # Load nodes
        for node_id, node_data in data.get('nodes', {}).items():
            node = ReasoningNode.from_dict(node_data)
            self.add_node(node)
    
    def save_to_jsonl(self, filepath: str) -> None:
        """Save nodes to JSONL format (one node per line)."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(filepath, 'w') as writer:
            # Write metadata first
            writer.write({
                'type': 'metadata',
                'root_id': self.root_id,
                'total_nodes': len(self.nodes),
                'timestamp': time.time()
            })
            
            # Write nodes
            for node in self.nodes.values():
                record = node.to_dict()
                record['type'] = 'node'
                writer.write(record)
    
    def load_from_jsonl(self, filepath: str) -> None:
        """Load nodes from JSONL format."""
        self.clear()
        
        with jsonlines.open(filepath, 'r') as reader:
            for record in reader:
                if record.get('type') == 'metadata':
                    self.root_id = record.get('root_id')
                elif record.get('type') == 'node':
                    node_data = {k: v for k, v in record.items() if k != 'type'}
                    node = ReasoningNode.from_dict(node_data)
                    self.add_node(node)
    
    # === Snapshot Methods ===
    
    def create_snapshot(self, name: str) -> None:
        """Create a named snapshot of the current state."""
        self._snapshots[name] = {
            'root_id': self.root_id,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'timestamp': time.time()
        }
    
    def restore_snapshot(self, name: str) -> bool:
        """Restore state from a named snapshot. Returns True if successful."""
        if name not in self._snapshots:
            return False
        
        snapshot = self._snapshots[name]
        
        self.clear()
        self.root_id = snapshot['root_id']
        
        for node_id, node_data in snapshot['nodes'].items():
            node = ReasoningNode.from_dict(node_data)
            self.add_node(node)
        
        return True
    
    def list_snapshots(self) -> List[str]:
        """List all available snapshot names."""
        return list(self._snapshots.keys())
    
    def delete_snapshot(self, name: str) -> bool:
        """Delete a named snapshot. Returns True if found and deleted."""
        if name in self._snapshots:
            del self._snapshots[name]
            return True
        return False
    
    # === Utility Methods ===
    
    def clear(self) -> None:
        """Clear all nodes and reset the store."""
        self.nodes.clear()
        self.root_id = None
        self._children_index.clear()
        self._depth_index.clear()
        self._status_index.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        if not self.nodes:
            return {'total_nodes': 0}
        
        status_counts = defaultdict(int)
        for node in self.nodes.values():
            status_counts[node.status.value] += 1
        
        depth_counts = defaultdict(int)
        for node in self.nodes.values():
            depth_counts[node.depth] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'root_id': self.root_id,
            'status_counts': dict(status_counts),
            'depth_counts': dict(depth_counts),
            'max_depth': max((node.depth for node in self.nodes.values()), default=0),
            'frontier_size': len(self.get_frontier_nodes()),
            'terminal_nodes': len(self.get_terminal_nodes())
        }
    
    def __len__(self) -> int:
        """Return the number of nodes in the store."""
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID exists in the store."""
        return node_id in self.nodes
    
    def __iter__(self) -> Iterator[ReasoningNode]:
        """Iterate over all nodes."""
        return iter(self.nodes.values())
