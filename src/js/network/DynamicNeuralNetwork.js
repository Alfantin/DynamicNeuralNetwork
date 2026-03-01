export class DynamicNeuralNetwork {
  constructor(inputCount, outputCount, learningRate = 0.03) {
    this.learningRate = learningRate;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.eps = 1e-8;
    this.nextNodeId = 0;
    this.nodes = {};
    this.edges = [];
    this.inputIds = [];
    this.outputIds = [];
    this.biasId = null;
    this.epoch = 0;
    this.lossHistory = [];
    this.windowLosses = [];
    this.maxHiddenNodes = 8;
    this.maxDepth = 6;
    this.growCount = 0;
    this.pruneCount = 0;

    for (let i = 0; i < inputCount; i++) {
      this.inputIds.push(this.createNode('input', `x${i + 1}`, 0));
    }

    this.biasId = this.createNode('bias', 'bias', 0);

    for (let i = 0; i < outputCount; i++) {
      this.outputIds.push(this.createNode('output', `y${i + 1}`, this.maxDepth));
    }

    for (const from of [...this.inputIds, this.biasId]) {
      for (const to of this.outputIds) {
        this.addEdge(from, to, this.xavierWeight(1, 1), true);
      }
    }
  }

  createNode(type, label = null, depth = 0) {
    const id = this.nextNodeId++;
    this.nodes[id] = {
      id,
      type,
      label: label || `${type}-${id}`,
      value: 0,
      preact: 0,
      delta: 0,
      depth,
      usageEMA: 0
    };
    return id;
  }

  xavierWeight(fanIn, fanOut) {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return (Math.random() * 2 - 1) * limit;
  }

  addEdge(from, to, weight = null, protectedEdge = false) {
    if (this.hasEdge(from, to)) return null;
    const edge = {
      from,
      to,
      weight: weight ?? this.xavierWeight(1, 1),
      enabled: true,
      m: 0,
      v: 0,
      gradEMA: 0,
      useEMA: 0,
      protectedEdge
    };
    this.edges.push(edge);
    return edge;
  }

  hasEdge(from, to) {
    return this.edges.some((edge) => edge.enabled && edge.from === from && edge.to === to);
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  tanh(x) {
    return Math.tanh(x);
  }

  tanhDerivativeFromOutput(y) {
    return 1 - y * y;
  }

  sigmoidDerivativeFromOutput(y) {
    return y * (1 - y);
  }

  incomingEdges(nodeId) {
    return this.edges.filter((edge) => edge.enabled && edge.to === nodeId);
  }

  outgoingEdges(nodeId) {
    return this.edges.filter((edge) => edge.enabled && edge.from === nodeId);
  }

  getHiddenIds() {
    return Object.values(this.nodes)
      .filter((node) => node.type === 'hidden')
      .map((node) => node.id);
  }

  getActiveEdges() {
    return this.edges.filter((edge) => edge.enabled && this.nodes[edge.from] && this.nodes[edge.to]);
  }

  forward(inputArray) {
    for (let i = 0; i < this.inputIds.length; i++) {
      this.nodes[this.inputIds[i]].value = inputArray[i];
    }

    this.nodes[this.biasId].value = 1;

    const computeOrder = Object.values(this.nodes)
      .filter((node) => node.type !== 'input' && node.type !== 'bias')
      .sort((a, b) => a.depth - b.depth || a.id - b.id);

    for (const node of computeOrder) {
      let sum = 0;
      const incoming = this.incomingEdges(node.id);

      for (const edge of incoming) {
        const contrib = this.nodes[edge.from].value * edge.weight;
        sum += contrib;
        edge.useEMA = edge.useEMA * 0.98 + Math.abs(contrib) * 0.02;
      }

      node.preact = sum;
      node.value = node.type === 'output' ? this.sigmoid(sum) : this.tanh(sum);
      node.usageEMA = node.usageEMA * 0.98 + Math.abs(node.value) * 0.02;
    }

    return this.outputIds.map((id) => this.nodes[id].value);
  }

  backward(targetArray) {
    const reverseOrder = Object.values(this.nodes)
      .filter((node) => node.type !== 'input' && node.type !== 'bias')
      .sort((a, b) => b.depth - a.depth || b.id - a.id);

    for (let i = 0; i < this.outputIds.length; i++) {
      const outId = this.outputIds[i];
      const out = this.nodes[outId].value;
      const error = targetArray[i] - out;
      this.nodes[outId].delta = error * this.sigmoidDerivativeFromOutput(out);
    }

    for (const node of reverseOrder) {
      if (node.type === 'output') continue;
      let downstream = 0;
      for (const edge of this.outgoingEdges(node.id)) {
        downstream += edge.weight * this.nodes[edge.to].delta;
      }
      node.delta = downstream * this.tanhDerivativeFromOutput(node.value);
    }

    const t = this.epoch + 1;
    for (const edge of this.edges) {
      if (!edge.enabled) continue;
      const fromVal = this.nodes[edge.from].value;
      const toDelta = this.nodes[edge.to].delta;
      const grad = fromVal * toDelta;
      edge.gradEMA = edge.gradEMA * 0.97 + Math.abs(grad) * 0.03;
      edge.m = this.beta1 * edge.m + (1 - this.beta1) * grad;
      edge.v = this.beta2 * edge.v + (1 - this.beta2) * grad * grad;
      const mHat = edge.m / (1 - Math.pow(this.beta1, t));
      const vHat = edge.v / (1 - Math.pow(this.beta2, t));
      edge.weight += this.learningRate * mHat / (Math.sqrt(vHat) + this.eps);
    }
  }

  trainSample(inputArray, targetArray) {
    const output = this.forward(inputArray);
    this.backward(targetArray);

    let loss = 0;
    for (let i = 0; i < output.length; i++) {
      const diff = targetArray[i] - output[i];
      loss += diff * diff;
    }

    return loss / output.length;
  }

  predict(inputArray) {
    return this.forward(inputArray);
  }

  chooseEdgeToSplit() {
    const candidates = this.getActiveEdges().filter((edge) => {
      const fromNode = this.nodes[edge.from];
      const toNode = this.nodes[edge.to];
      return toNode.depth - fromNode.depth >= 2;
    });

    if (!candidates.length) return null;

    let best = null;
    let bestScore = -Infinity;

    for (const edge of candidates) {
      const score = 0.55 * edge.gradEMA + 0.35 * edge.useEMA + 0.1 * Math.abs(edge.weight);
      if (score > bestScore) {
        bestScore = score;
        best = edge;
      }
    }

    return best;
  }

  addHiddenNodeBySplittingBestEdge() {
    if (this.getHiddenIds().length >= this.maxHiddenNodes) return false;

    const edge = this.chooseEdgeToSplit();
    if (!edge) return false;

    const fromNode = this.nodes[edge.from];
    const toNode = this.nodes[edge.to];
    const newDepth = Math.floor((fromNode.depth + toNode.depth) / 2);

    if (newDepth <= fromNode.depth || newDepth >= toNode.depth) return false;

    edge.enabled = false;
    const hiddenIndex = this.getHiddenIds().length + 1;
    const hiddenId = this.createNode('hidden', `h${hiddenIndex}`, newDepth);

    this.addEdge(edge.from, hiddenId, 1.0, true);
    this.addEdge(hiddenId, edge.to, edge.weight, true);

    const usefulSources = [...this.inputIds, this.biasId, ...this.getHiddenIds()]
      .filter((id) => id !== hiddenId && this.nodes[id] && this.nodes[id].depth < newDepth && !this.hasEdge(id, hiddenId))
      .sort((a, b) => this.nodes[b].usageEMA - this.nodes[a].usageEMA)
      .slice(0, 2);

    for (const src of usefulSources) {
      if (Math.random() < 0.8) this.addEdge(src, hiddenId, this.xavierWeight(2, 2));
    }

    const usefulTargets = [...this.getHiddenIds(), ...this.outputIds]
      .filter((id) => id !== hiddenId && this.nodes[id] && this.nodes[id].depth > newDepth && !this.hasEdge(hiddenId, id))
      .sort((a, b) => this.nodes[b].usageEMA - this.nodes[a].usageEMA)
      .slice(0, 2);

    for (const dst of usefulTargets) {
      if (Math.random() < 0.6) this.addEdge(hiddenId, dst, this.xavierWeight(2, 2));
    }

    this.growCount += 1;
    return true;
  }

  pruneWeakEdges(threshold = 0.008) {
    let pruned = 0;

    for (const edge of this.edges) {
      if (!edge.enabled || edge.protectedEdge) continue;
      const score = 0.6 * edge.useEMA + 0.4 * edge.gradEMA;
      if (score < threshold && Math.abs(edge.weight) < 0.18) {
        edge.enabled = false;
        pruned += 1;
      }
    }

    this.pruneCount += pruned;
  }

  removeIsolatedHiddenNodes() {
    for (const hid of this.getHiddenIds()) {
      const incoming = this.incomingEdges(hid).length;
      const outgoing = this.outgoingEdges(hid).length;

      if (incoming === 0 || outgoing === 0) {
        for (const edge of this.edges) {
          if (edge.from === hid || edge.to === hid) edge.enabled = false;
        }
        delete this.nodes[hid];
      }
    }
  }

  maybeGrowOrPrune(avgLoss) {
    this.windowLosses.push(avgLoss);
    if (this.windowLosses.length > 40) this.windowLosses.shift();
    if (this.windowLosses.length < 40) return;

    const early = this.windowLosses.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const late = this.windowLosses.slice(-10).reduce((a, b) => a + b, 0) / 10;
    const improvement = early - late;

    if (this.epoch > 200 && improvement < 0.0025 && avgLoss > 0.02) {
      const grown = this.addHiddenNodeBySplittingBestEdge();
      if (!grown) {
        this.pruneWeakEdges();
        this.removeIsolatedHiddenNodes();
      }
      this.windowLosses = [];
    } else if (this.epoch % 120 === 0) {
      this.pruneWeakEdges();
      this.removeIsolatedHiddenNodes();
    }
  }

  trainEpoch(dataset) {
    let totalLoss = 0;

    for (const sample of dataset) {
      totalLoss += this.trainSample(sample.input, sample.target);
    }

    this.epoch += 1;
    const avgLoss = totalLoss / dataset.length;
    this.lossHistory.push(avgLoss);
    if (this.lossHistory.length > 300) this.lossHistory.shift();
    this.maybeGrowOrPrune(avgLoss);

    return avgLoss;
  }
}
