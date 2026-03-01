import { DynamicNeuralNetwork } from '../network/DynamicNeuralNetwork.js';
import { GraphRenderer } from './GraphRenderer.js';

export class AppController {
  constructor(elements, dataset) {
    this.elements = elements;
    this.dataset = dataset;
    this.autoTraining = false;
    this.autoHandle = null;
    this.net = new DynamicNeuralNetwork(2, 1, 0.03);
    this.autoStopEpoch = 1000;
    this.stopLossThreshold = 0.008;
    this.refinementWindows = 0;
    this.requiredRefinementWindows = 12;
    this.bestAutoCandidate = null;
    this.renderer = new GraphRenderer(elements.graphCanvas, this.net);
  }

  init() {
    this.bindEvents();
    this.elements.autoBtn.textContent = 'Baslat';
    this.elements.stopBtn.textContent = 'Durdur';
    this.applyTrainingSettings();
    this.updateStats(null);
  }

  bindEvents() {
    this.elements.autoBtn.addEventListener('click', () => this.startAuto());
    this.elements.stopBtn.addEventListener('click', () => this.stopAuto());
    this.elements.resetBtn.addEventListener('click', () => this.resetAll());

    const settingsInputs = [
      this.elements.autoStopEpochInput,
      this.elements.stopLossThresholdInput,
      this.elements.growthStartEpochInput,
      this.elements.splitThresholdInput,
      this.elements.pruneIntervalInput,
      this.elements.pruneThresholdInput
    ];

    for (const input of settingsInputs) {
      input.addEventListener('change', () => this.applyTrainingSettings());
    }
  }

  updateResults() {
    const rows = this.dataset
      .map((sample) => {
        const prediction = this.net.predict(sample.input)[0];
        return `<tr>
          <td class="mono">[${sample.input.join(', ')}]</td>
          <td class="mono">${prediction.toFixed(4)}</td>
          <td class="mono">${sample.target[0]}</td>
        </tr>`;
      })
      .join('');

    this.elements.resultBody.innerHTML = rows;
  }

  updateStats(lastLoss = null) {
    this.elements.epochVal.textContent = String(this.net.epoch);
    this.elements.lossVal.textContent = lastLoss == null ? '-' : lastLoss.toFixed(5);
    this.elements.hiddenVal.textContent = String(this.net.getHiddenIds().length);
    this.elements.edgeVal.textContent = String(this.net.getActiveEdges().length);
    this.elements.growVal.textContent = String(this.net.growCount);
    this.elements.pruneVal.textContent = String(this.net.pruneCount);

    this.updateResults();
    this.renderer.render();
  }

  runEpochs(count) {
    let loss = null;

    for (let i = 0; i < count; i++) {
      loss = this.net.trainEpoch(this.dataset);
    }

    this.updateStats(loss);
    return loss;
  }

  startAuto() {
    if (this.autoTraining) return;

    this.autoTraining = true;
    this.refinementWindows = 0;
    this.bestAutoCandidate = null;
    this.elements.autoBtn.classList.add('hidden-control');
    this.elements.stopBtn.classList.remove('hidden-control');

    this.autoHandle = setInterval(() => {
      const loss = this.runEpochs(8);
      const preds = this.dataset.map((sample) => this.net.predict(sample.input)[0]);
      const learnedXor = preds[0] < 0.08 && preds[1] > 0.92 && preds[2] > 0.92 && preds[3] < 0.08;
      const reachedLossThreshold = loss != null && loss <= this.stopLossThreshold;
      const meetsTarget = learnedXor || reachedLossThreshold;

      if (meetsTarget) {
        this.refinementWindows += 1;
        this.captureBestAutoCandidate(loss);
        const tightenedThreshold = Math.min(0.03, this.net.behavior.pruneThreshold + 0.002);
        const pruned = this.net.pruneWeakEdges(tightenedThreshold);
        if (pruned > 0) this.net.removeIsolatedHiddenNodes();
        this.updateStats(loss);
      } else {
        this.refinementWindows = 0;
      }

      if (this.refinementWindows >= this.requiredRefinementWindows) {
        this.stopAuto(true);
        return;
      }

      if (this.net.epoch >= this.autoStopEpoch) {
        this.stopAuto(true);
      }
    }, 35);
  }

  stopAuto(restoreBest = false) {
    this.autoTraining = false;
    clearInterval(this.autoHandle);
    this.autoHandle = null;

    if (restoreBest && this.bestAutoCandidate) {
      this.net.restoreState(this.bestAutoCandidate.snapshot);
      this.updateStats(this.bestAutoCandidate.loss);
    }

    this.elements.autoBtn.classList.remove('hidden-control');
    this.elements.stopBtn.classList.add('hidden-control');
  }

  resetAll() {
    this.stopAuto();
    this.net = new DynamicNeuralNetwork(2, 1, 0.03);
    this.applyTrainingSettings();
    this.renderer.setNetwork(this.net);
    this.updateStats(null);
  }

  applyTrainingSettings() {
    this.autoStopEpoch = this.getInputNumber(this.elements.autoStopEpochInput, 1000, 50);
    this.stopLossThreshold = this.getInputNumber(this.elements.stopLossThresholdInput, 0.008, 0);
    const growthStartEpoch = this.getInputNumber(this.elements.growthStartEpochInput, 120, 40);

    this.net.updateBehavior({
      growthStartEpoch,
      improvementPruneStartEpoch: Math.max(40, growthStartEpoch - 40),
      growthImprovementThreshold: this.getInputNumber(this.elements.splitThresholdInput, 0.0015, 0),
      pruneInterval: this.getInputNumber(this.elements.pruneIntervalInput, 80, 20),
      pruneThreshold: this.getInputNumber(this.elements.pruneThresholdInput, 0.01, 0.001)
    });
  }

  captureBestAutoCandidate(loss) {
    const candidate = {
      loss,
      complexity: this.net.getComplexityScore(),
      snapshot: this.net.snapshotState()
    };

    if (!this.bestAutoCandidate) {
      this.bestAutoCandidate = candidate;
      return;
    }

    const currentScore = this.bestAutoCandidate.loss + this.bestAutoCandidate.complexity * 0.0006;
    const nextScore = candidate.loss + candidate.complexity * 0.0006;
    if (nextScore < currentScore - 0.00005) {
      this.bestAutoCandidate = candidate;
    }
  }

  getInputNumber(input, fallback, minValue) {
    const value = Number(input.value);
    if (!Number.isFinite(value)) {
      input.value = String(fallback);
      return fallback;
    }

    const normalized = Math.max(minValue, value);
    input.value = String(normalized);
    return normalized;
  }
}
