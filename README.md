# OpenCog Inferno AGI Operating System

A revolutionary approach to artificial general intelligence that implements cognitive processing as a fundamental kernel service. Instead of layering cognitive architectures on top of existing operating systems, this implementation makes thinking, reasoning, and intelligence emerge from the operating system itself.

## Overview

This project implements OpenCog as a pure Inferno kernel-based distributed AGI operating system. Drawing inspiration from both the OpenCog cognitive architecture and the Inferno distributed operating system, it creates a unified system where:

- **Cognitive processing is a kernel service** - Not an application running on top of an OS
- **Everything is a file** - Following Inferno's philosophy, cognitive resources are exposed through a hierarchical namespace
- **Distribution is transparent** - Cognitive processes can run on any node and access any AtomSpace
- **Intelligence emerges** - From the dynamic interaction of multiple cognitive subsystems

## Architecture

```
+--------------------------------------------------+
|              Emergent Intelligence               |
|  (Synergy, Goals, Reflection, Creativity)        |
+--------------------------------------------------+
|    PLN     |   MOSES   |   ECAN   |   Pattern   |
| Reasoning  | Learning  | Attention| Recognition |
+--------------------------------------------------+
|              Cognitive File System               |
|        (CogFS with Styx/9P Protocol)            |
+--------------------------------------------------+
|              AtomSpace Hypergraph                |
|         (Distributed Knowledge Base)            |
+--------------------------------------------------+
|           Cognitive Process Scheduler            |
|      (Attention-Based Process Management)       |
+--------------------------------------------------+
|              Memory Management                   |
|    (Forgetting, Consolidation, Caching)         |
+--------------------------------------------------+
```

## Core Components

### 1. AtomSpace Hypergraph (`atomspace/`)

The fundamental knowledge representation layer implementing a weighted, labeled hypergraph:

- **Nodes**: Represent concepts, predicates, schemas, and variables
- **Links**: Represent relationships with arbitrary arity
- **Truth Values**: Probabilistic confidence measures
- **Attention Values**: Economic attention allocation

### 2. Cognitive Kernel (`kernel/`)

The heart of the AGI operating system:

- **Cognitive Types** (`kernel/cognitive/types.py`): Core type definitions
- **Memory Manager** (`kernel/memory/manager.py`): Attention-based memory management with forgetting
- **PLN Engine** (`kernel/reasoning/pln.py`): Probabilistic Logic Networks for reasoning
- **ECAN** (`kernel/attention/ecan.py`): Economic Attention Networks for resource allocation
- **Pattern Recognition** (`kernel/pattern/recognition.py`): Subgraph mining and pattern detection
- **MOSES** (`kernel/learning/moses.py`): Meta-Optimizing Semantic Evolutionary Search for program learning

### 3. Cognitive File System (`fs/`)

Following Inferno's "everything is a file" philosophy:

```
/cog/
    atoms/          - Direct atom access by UUID
    types/          - Atoms organized by type
    attention/      - Attention-based views (focus, fringe)
    inference/      - Inference operations
    learning/       - Learning operations
    processes/      - Cognitive processes
    stats/          - System statistics
    query           - Query interface
```

### 4. Process Scheduler (`proc/`)

Attention-based cognitive process scheduling:

- **Priority Classes**: Urgent, High, Normal, Low, Background
- **Process Types**: Inference, Learning, Attention, Perception, Action
- **Goal-Directed**: Processes linked to cognitive goals
- **IPC**: Inter-process communication via channels

### 5. Knowledge Representation (`knowledge/`)

Rich ontological knowledge representation:

- **Ontology Management**: Classes, properties, relations
- **Frame Semantics**: FrameNet-style semantic frames
- **Fact Management**: Triple-based knowledge storage
- **Reasoning Support**: Inheritance, similarity, causation

### 6. Distributed Coordination (`distributed/`)

Transparent distribution across cluster nodes:

- **Cluster Management**: Node discovery, health monitoring
- **Distributed AtomSpace**: Synchronized knowledge across nodes
- **Consensus Protocol**: Paxos-based distributed decisions
- **Load Balancing**: Attention-aware process distribution

### 7. Emergent Intelligence (`kernel/emergence/`)

The highest level of cognitive integration:

- **Cognitive Synergy**: Coordinated interaction between subsystems
- **Goal Management**: Hierarchical goal pursuit
- **Self-Reflection**: Introspection and self-modification
- **Creativity Engine**: Novel idea generation

## Key Features

### Probabilistic Logic Networks (PLN)

Implements uncertain reasoning with:
- Deduction, induction, abduction
- Modus ponens, revision
- Forward and backward chaining
- Attention-guided inference

### Economic Attention Networks (ECAN)

Implements attention as an economic system:
- Atoms have "wealth" (attention value)
- Atoms pay "rent" to stay in memory
- Useful atoms earn "wages"
- Attention spreads through Hebbian links

### MOSES Learning

Meta-Optimizing Semantic Evolutionary Search:
- Genetic programming for program learning
- Fitness-based evolution
- Crossover and mutation operators
- Integration with AtomSpace

### Pattern Recognition

Discovers patterns in the knowledge base:
- Frequent subgraph mining
- Support and confidence metrics
- Attention-guided pattern search
- Pattern-based reasoning

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd opencog-inferno-agi

# No external dependencies required - pure Python implementation
# Python 3.11+ recommended

# Run tests
python3 tests/test_cognitive_kernel.py

# Run the kernel
python3 kernel/cognitive_kernel.py
```

## Usage

### Basic Kernel Usage

```python
from kernel.cognitive_kernel import boot_kernel, KernelState

# Boot the kernel
kernel = boot_kernel(kernel_id="my-kernel")

# Add knowledge
kernel.atomspace.add_node(AtomType.CONCEPT_NODE, "Cat")
kernel.atomspace.add_node(AtomType.CONCEPT_NODE, "Animal")
kernel.atomspace.add_link(
    AtomType.INHERITANCE_LINK,
    [cat_handle, animal_handle],
    tv=TruthValue(1.0, 0.9)
)

# Run cognitive cycles
for _ in range(10):
    kernel.run_cycle()
    result = kernel.think()
    print(f"Focus: {result['focus_size']}, Inferences: {result['inferences']}")

# Create goals
goal_id = kernel.create_goal("Learn about cats", priority=0.8)

# Shutdown
kernel.shutdown()
```

### Using the Cognitive File System

```python
from fs.cogfs.filesystem import CognitiveFileSystem

cogfs = CognitiveFileSystem(atomspace)

# Read directory
entries = cogfs.readdir('/cog')

# Open and read atom file
fd = cogfs.open('/cog/types/concept_node')
data = cogfs.read(fd, 1024)
cogfs.close(fd)

# Query atoms
results = cogfs.query_atoms({'type': 'CONCEPT_NODE'})
```

### Distributed Operation

```python
from distributed.coordination.cluster import DistributedCoordinationService

# Create distributed service
distributed = DistributedCoordinationService(
    atomspace,
    node_id="node-1",
    address="192.168.1.100",
    port=9000
)

# Start coordination
distributed.start()

# Use distributed AtomSpace
distributed.distributed_atomspace.add_node(
    AtomType.CONCEPT_NODE,
    "SharedConcept"
)

# Stop
distributed.stop()
```

## Project Structure

```
opencog-inferno-agi/
├── kernel/
│   ├── cognitive/
│   │   └── types.py           # Core type definitions
│   ├── memory/
│   │   └── manager.py         # Memory management
│   ├── reasoning/
│   │   └── pln.py             # PLN reasoning engine
│   ├── attention/
│   │   └── ecan.py            # ECAN attention system
│   ├── pattern/
│   │   └── recognition.py     # Pattern recognition
│   ├── learning/
│   │   └── moses.py           # MOSES learning
│   ├── emergence/
│   │   └── intelligence.py    # Emergent intelligence
│   └── cognitive_kernel.py    # Main kernel
├── atomspace/
│   ├── hypergraph/
│   │   └── atomspace.py       # AtomSpace implementation
│   └── query/
│       └── pattern_matcher.py # Pattern matching
├── fs/
│   └── cogfs/
│       └── filesystem.py      # Cognitive file system
├── proc/
│   └── scheduler/
│       └── cognitive_scheduler.py  # Process scheduler
├── knowledge/
│   └── ontology/
│       └── representation.py  # Knowledge representation
├── distributed/
│   └── coordination/
│       └── cluster.py         # Distributed coordination
├── tests/
│   └── test_cognitive_kernel.py  # Test suite
└── README.md
```

## Design Philosophy

### From Inferno OS

- **Everything is a file**: Cognitive resources exposed through hierarchical namespace
- **Styx/9P protocol**: Network-transparent resource access
- **Limbo-style concurrency**: Channel-based communication
- **Portable across architectures**: Pure Python implementation

### From OpenCog

- **AtomSpace hypergraph**: Unified knowledge representation
- **PLN reasoning**: Probabilistic uncertain inference
- **ECAN attention**: Economic attention allocation
- **MOSES learning**: Evolutionary program synthesis
- **Cognitive synergy**: Subsystem cooperation

### Novel Contributions

- **Kernel-level cognition**: Intelligence as OS service
- **Attention-based scheduling**: Cognitive resource management
- **Cognitive file system**: File-based cognitive interface
- **Emergent intelligence layer**: Integrated cognitive behavior

## Future Directions

1. **Hardware acceleration**: GPU support for parallel inference
2. **Persistent storage**: Disk-backed AtomSpace
3. **Network protocol**: Full Styx/9P implementation
4. **Language bindings**: C/C++, Rust, JavaScript
5. **Visualization**: Real-time cognitive state visualization
6. **Benchmarks**: Standard AGI benchmarks

## References

- [OpenCog Wiki](https://wiki.opencog.org/)
- [Inferno OS](http://www.vitanuova.com/inferno/)
- [PLN Book](https://wiki.opencog.org/w/PLN_Book)
- [ECAN](https://wiki.opencog.org/w/Attention_Allocation)
- [MOSES](https://wiki.opencog.org/w/MOSES)

## License

This project is released under the MIT License.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

---

*"The question is not whether machines can think, but whether the operating system itself can be intelligent."*
