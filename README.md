# Optimize Dynamic XOR Neural Network

> Thanks for exploring this experimental project. Built as a GPT-assisted idea-to-demo prototype for dynamic, self-growing neural graph architectures.

A browser-based demo that explores a **dynamic, graph-shaped neural network** instead of a fixed hidden-layer architecture.

In this project, the network starts with only:

* **input nodes**
* **a bias node**
* **output nodes**

During training, it can **grow its internal structure automatically** by splitting important connections and inserting new hidden nodes only when necessary.

---

## Why this project exists

Traditional neural networks usually follow a fixed structure like:

`input -> hidden layer(s) -> output`

That works well in practice, but it also forces every signal to pass through the same layer-based pipeline.

This project experiments with a different idea:

* some signals may go **directly** from input to output,
* some may need **one intermediate node**,
* others may need a **longer path**,
* and the internal structure should **form during training**, instead of being fully defined in advance.

The goal is to move from a **layer-based architecture** toward a more flexible **node/path-based topology**.

---

## What this demo does

This demo solves the **XOR problem** with an adaptive network that:

* starts with direct input-to-output connections,
* tries to learn the shortest solution first,
* tracks which connections are most useful,
* adds hidden nodes only when progress slows down,
* prunes weak or low-value connections,
* visualizes the evolving graph in real time.

---

## Main ideas behind the architecture

### 1. Shortest path first

The model first tries to solve the problem using the simplest structure possible.

Instead of creating hidden layers from the beginning, it starts with direct routes:

* `x1 -> y1`
* `x2 -> y1`
* `bias -> y1`

Only if this is not enough does it grow more structure.

### 2. Selective growth

When learning stalls, the network does **not** grow randomly.

It chooses an important connection based on metrics such as:

* gradient activity,
* usage contribution,
* weight magnitude.

Then it **splits that edge** and inserts a new hidden node in between.

So a path can evolve like this:

* `x1 -> y1`
* `x1 -> h1 -> y1`
* `x1 -> h1 -> h3 -> y1`

while another input may still use a short path like:

* `x2 -> y1`

### 3. Controlled complexity

The network is allowed to grow, but only when necessary.

To keep the structure useful and efficient, it also:

* prunes weak connections,
* removes isolated hidden nodes,
* protects important structural edges,
* limits total hidden-node growth.

---

## Features

* **Dynamic topology** instead of fixed hidden layers
* **Graph-based visualization** on HTML canvas
* **Adaptive node insertion** by splitting useful edges
* **Connection pruning** based on importance
* **Adam-style optimization** for more stable training
* **tanh hidden activations** and **sigmoid output activation**
* **Live XOR prediction table**
* **Training stats** such as epoch, loss, growth count, prune count, and active connections

---

## How it works

### Training flow

For each epoch:

1. The network performs a forward pass.
2. It computes loss on XOR samples.
3. It performs backpropagation.
4. It updates weights with an Adam-like optimizer.
5. It tracks connection usefulness over time.
6. If progress stalls, it either:

   * grows by splitting the best edge, or
   * prunes weak edges and removes dead hidden nodes.

### Growth strategy

A hidden node is added by:

1. choosing the most promising edge,
2. disabling that edge,
3. placing a hidden node between source and target,
4. reconnecting the graph through the new node,
5. optionally attaching a few additional useful inputs/targets.

This creates a **deeper but still sparse** structure.

---

## UI overview

The interface contains:

* **1 Epoch** → trains for one epoch
* **100 Epoch** → trains for 100 epochs
* **Auto Train** → runs training automatically until XOR is solved or max epoch is reached
* **Stop** → pauses auto training
* **Reset** → resets the network

### Visual legend

* **Green**: input nodes
* **Yellow**: bias node
* **Purple**: hidden nodes
* **Pink**: output nodes
* **Blue edges**: positive weights
* **Red edges**: negative weights

Edge thickness represents weight magnitude.

---

## Project structure

This demo is currently implemented as a **single HTML file** containing:

* HTML layout
* CSS styling
* JavaScript neural-network implementation
* Canvas-based rendering logic

That makes it easy to clone, open, and run locally without any build tooling.

---

## How to run

### Option 1: Open directly in browser

Just open the HTML file in any modern browser.

### Option 2: Use a local server

If you prefer:

```bash
python -m http.server 8000
```

Then open your browser and go to your local server address.

---

## Example use cases

This project is mainly an **experimental architecture demo**, but the ideas may be useful for exploring:

* adaptive network growth,
* sparse computation,
* learned graph topologies,
* structure search without fully predefined hidden layers,
* biologically inspired alternatives to rigid layer stacks.

---

## Limitations

This is an experimental proof of concept, not a production-ready neural-network framework.

Current limitations include:

* XOR is a very small toy problem,
* the structure search is still heuristic,
* graph-shaped models are harder to optimize than standard dense MLPs,
* this implementation is designed for clarity and visualization rather than raw speed,
* GPU-friendly tensor execution is not used here.

In practice, fixed-layer MLPs are often still faster and easier to train.

---

## Why this can be interesting anyway

Even if fixed hidden layers remain stronger in many real-world tasks, this kind of experiment is useful because it asks a different question:

> What if a neural network did not have to decide only its weights, but also its internal path structure during learning?

That opens the door to architectures where:

* easy signals take short routes,
* difficult signals take deeper routes,
* the network stays compact unless complexity is actually needed.

---

## Future improvements

Possible next steps:

* per-sample adaptive routing
* path explanation panel
* input-to-output influence tracing
* comparison mode vs standard MLP
* node merge / compression logic
* better topology scoring
* support for multiple outputs
* benchmark mode for speed and accuracy comparison
* WebGL or tensor-based acceleration

---

## Suggested repository name

A few good repository names for this project:

* `dynamic-xor-network`
* `adaptive-graph-neural-demo`
* `layerless-neural-network-demo`
* `dynamic-topology-xor`
* `self-growing-neural-graph`

---

## License

You can use a simple MIT License if you want to publish this as an open-source experiment.

---

## Summary

This project is a small but concrete experiment in replacing rigid hidden layers with a **self-growing neural graph**.

Instead of saying:

> “define all hidden layers first, then train weights”

it explores the alternative:

> “define inputs and outputs first, then let internal structure emerge only where needed.”

That is the core idea behind this demo.
