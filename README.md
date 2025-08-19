# Synthesis of Thought

A comprehensive Python framework for implementing Synthesis of Thought reasoning, where multiple reasoning paths are explored and synthesized rather than selecting a single best path.

## Overview

The Synthesis of Thought approach explores reasoning as a tree search over possible thinking moves, where:
- **Nodes** represent proposed reasoning steps
- **Edges** represent sequential reasoning relationships
- **Paths** are complete reasoning trajectories
- **Synthesis** combines multiple paths with probability weights

Unlike Tree of Thought which selects the best single path, Synthesis of Thought maintains and combines multiple reasoning approaches to provide a more comprehensive solution.

## Features

- **Comprehensive Node Structure**: Rich node representation with metadata, scoring, and verification
- **Tree Management**: Efficient tree operations with caching and duplicate detection
- **Path Operations**: Advanced path scoring, merging, and synthesis
- **Configurable System**: Flexible configuration for different reasoning scenarios
- **Visualization Tools**: Text and graphical tree visualization
- **Export Utilities**: Export results to JSON, Markdown, and comparison reports

## Installation

```bash
# Clone or download the project
cd Synthesis_Of_Thought

# Install required dependencies
pip install numpy networkx matplotlib
```

## Quick Start

```python
from synthesis import SynthesisOfThought, SynthesisConfig

# Configure the system
config = SynthesisConfig(
    max_depth=4,
    max_branches_per_node=3,
    max_total_nodes=50
)

# Create synthesis system with your LLM functions
sot = SynthesisOfThought(
    config=config,
    llm_generator=your_llm_generator,
    llm_scorer=your_llm_scorer,
    llm_verifier=your_llm_verifier
)

# Solve a question
result = sot.solve("Why did the glass break?")

# Access synthesized paths
for i, path in enumerate(result['synthesized_paths']):
    print(f"Path {i+1}: {path.get_text_sequence()}")
    print(f"Probability: {result['probabilities'][i]:.3f}")
```

## Core Components

### 1. Node (`node.py`)
- `ReasoningNode`: Core data structure for reasoning steps
- Rich metadata including evidence, contradictions, and verification
- Configurable scoring and probability weights

### 2. Tree (`tree.py`)
- `ReasoningTree`: Manages the tree structure and operations
- Efficient frontier management for active exploration
- Path caching and duplicate detection

### 3. Path Operations (`path_operations.py`)
- `PathOperations`: Advanced path analysis and synthesis
- Comprehensive scoring (confidence, completeness, coherence, novelty, evidence)
- Complementary fragment detection and merging

### 4. Synthesis Engine (`synthesis.py`)
- `SynthesisOfThought`: Main orchestrator for the reasoning process
- Configurable expansion, scoring, and verification
- Statistics tracking and logging

### 5. Utilities (`utils.py`)
- `TreeVisualizer`: Text and graphical tree visualization
- `TreeAnalyzer`: Analysis of branching patterns and consensus
- `ResultExporter`: Export to various formats

## Example Usage

See `example.py` for complete examples including:
- Glass breaking problem (detailed example)
- Custom question handling
- Result visualization and export

```bash
python example.py
```

## Configuration

The `SynthesisConfig` class provides extensive configuration options:

```python
config = SynthesisConfig(
    max_depth=10,              # Maximum reasoning depth
    max_branches_per_node=3,   # Branching factor
    max_total_nodes=100,       # Total node limit
    min_score_threshold=0.3,   # Minimum score for keeping nodes
    max_synthesis_paths=5,     # Number of final paths to synthesize
    enable_pruning=True,       # Enable low-score pruning
    enable_verification=True,  # Enable step verification
    temperature=0.7            # LLM sampling temperature
)
```

## LLM Integration

You need to provide three LLM functions:

### Generator Function
```python
def llm_generator(context: str, max_branches: int, temperature: float) -> List[str]:
    """Generate next reasoning steps given context."""
    # Call your LLM API here
    return ["step1", "step2", "step3"]
```

### Scorer Function
```python
def llm_scorer(step: str, context: str) -> float:
    """Score a reasoning step quality (0.0 to 1.0)."""
    # Use LLM to evaluate step quality
    return 0.8
```

### Verifier Function
```python
def llm_verifier(step: str, context: str) -> Dict[str, Any]:
    """Verify reasoning step validity."""
    return {
        'verified': True,
        'evidence': ["supporting evidence"],
        'contradictions': [],
        'score': 0.9
    }
```

## Tree Visualization

Generate visualizations of your reasoning trees:

```python
from utils import TreeVisualizer

visualizer = TreeVisualizer(sot.tree)

# Text visualization
print(visualizer.generate_text_tree())

# Graphical plot (requires matplotlib)
visualizer.plot_tree(save_path="reasoning_tree.png")
```

## Analysis and Export

Analyze reasoning patterns and export results:

```python
from utils import TreeAnalyzer, ResultExporter

# Analyze the tree
analyzer = TreeAnalyzer(sot.tree)
branching_analysis = analyzer.analyze_branching_patterns()
score_analysis = analyzer.analyze_score_distribution()

# Export results
ResultExporter.export_to_json(result, "result.json")
ResultExporter.export_to_markdown(result, "result.md")
```

## Key Concepts

### Synthesis vs Selection
Unlike traditional tree search that selects one best path, Synthesis of Thought:
- Maintains multiple promising paths
- Assigns probability weights to different approaches
- Combines complementary reasoning fragments
- Provides comprehensive coverage of the reasoning space

### Complementary Fragments
The system identifies complementary reasoning fragments that:
- Address different aspects of the problem
- Are non-contradictory
- Can be combined for more complete reasoning
- Represent different valid approaches

### Probability Weighting
Final paths receive probability weights based on:
- Individual path confidence scores
- Novelty and uniqueness
- Evidence support
- Coherence and completeness

## File Structure

```
Synthesis_Of_Thought/
├── node.py              # Node data structure
├── tree.py              # Tree management
├── path_operations.py   # Path analysis and synthesis
├── synthesis.py         # Main synthesis engine
├── utils.py             # Visualization and analysis utilities
├── example.py           # Usage examples
└── README.md           # This file
```

## Advanced Usage

### Custom Scoring
Implement custom scoring functions for domain-specific reasoning:

```python
def custom_scorer(step: str, context: str) -> float:
    # Domain-specific scoring logic
    score = base_score
    if "technical_term" in step:
        score += 0.2
    return min(1.0, score)
```

### Batch Processing
Process multiple questions in batch:

```python
questions = ["Question 1", "Question 2", "Question 3"]
results = []

for question in questions:
    result = sot.solve(question)
    results.append(result)

# Generate comparison report
from utils import create_comparison_report
create_comparison_report(results, "batch_results.md")
```

### Integration with Existing Systems
The framework is designed to integrate with existing LLM systems and reasoning frameworks. Simply implement the three required LLM functions using your preferred LLM API.

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## Contributing

Contributions are welcome! Areas for improvement include:
- Additional scoring methods
- Enhanced visualization options
- Integration examples with popular LLM APIs
- Performance optimizations
- Additional analysis utilities
