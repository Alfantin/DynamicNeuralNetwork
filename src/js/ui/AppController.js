import { DynamicNeuralNetwork } from '../network/DynamicNeuralNetwork.js';
import { GraphRenderer } from './GraphRenderer.js';

export class AppController {
  constructor(elements, modelOptions) {
    this.elements = elements;
    this.modelOptions = modelOptions;
    this.currentModel = modelOptions[0];
    this.dataset = this.currentModel.dataset;
    this.autoTraining = false;
    this.autoHandle = null;
    this.net = this.createNetworkForCurrentModel();
    this.autoStopEpoch = 1000;
    this.stopLossThreshold = 0.008;
    this.refinementWindows = 0;
    this.requiredRefinementWindows = 12;
    this.minTargetEpoch = 0;
    this.bestAutoCandidate = null;
    this.renderer = new GraphRenderer(elements.graphCanvas, this.net);
  }

  init() {
    this.populateModelSelect();
    this.bindEvents();
    this.elements.autoBtn.textContent = 'Start';
    this.elements.stopBtn.textContent = 'Stop';
    this.applyTrainingSettings();
    this.updateStats(null);
  }

  bindEvents() {
    this.elements.modelSelect.addEventListener('change', (event) => this.changeModel(event.target.value));
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
        const prediction = this.net.predict(sample.input);
        return `<tr>
          <td class="mono">[${sample.input.join(', ')}]</td>
          <td class="mono">[${prediction.map((value) => value.toFixed(4)).join(', ')}]</td>
          <td class="mono">[${sample.target.join(', ')}]</td>
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
      const learnedPattern = this.hasLearnedCurrentDataset();
      const reachedLossThreshold = loss != null && loss <= this.stopLossThreshold;
      const targetWindowOpen = this.net.epoch >= this.minTargetEpoch;
      const useLossThreshold = this.stopLossThreshold > 0;
      const meetsTarget = targetWindowOpen && (useLossThreshold ? reachedLossThreshold : learnedPattern);

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
    this.net = this.createNetworkForCurrentModel();
    this.applyTrainingSettings();
    this.renderer.setNetwork(this.net);
    this.updateStats(null);
  }

  populateModelSelect() {
    const options = this.modelOptions
      .map((model) => `<option value="${model.key}">${model.label}</option>`)
      .join('');

    this.elements.modelSelect.innerHTML = options;
    this.elements.modelSelect.value = this.currentModel.key;
    this.elements.modelDesc.textContent = this.currentModel.description;
    this.applyModelProfile();
  }

  changeModel(modelKey) {
    const nextModel = this.modelOptions.find((model) => model.key === modelKey);
    if (!nextModel || nextModel.key === this.currentModel.key) return;

    this.currentModel = nextModel;
    this.dataset = nextModel.dataset;
    this.elements.modelDesc.textContent = nextModel.description;
    this.applyModelProfile();
    this.resetAll();
  }

  createNetworkForCurrentModel() {
    return new DynamicNeuralNetwork(this.currentModel.inputCount, this.currentModel.outputCount, 0.03);
  }

  applyTrainingSettings() {
    const profile = this.currentModel.profile;
    this.autoStopEpoch = this.getInputNumber(this.elements.autoStopEpochInput, profile.autoStopEpoch ?? 1000, 50);
    this.stopLossThreshold = this.getInputNumber(this.elements.stopLossThresholdInput, profile.stopLossThreshold ?? 0.008, 0);
    this.requiredRefinementWindows = profile.requiredRefinementWindows ?? 12;
    this.minTargetEpoch = profile.minTargetEpoch ?? 0;
    const growthStartEpoch = this.getInputNumber(this.elements.growthStartEpochInput, profile.growthStartEpoch ?? 120, 40);

    this.net.updateBehavior({
      growthStartEpoch,
      improvementPruneStartEpoch: Math.max(40, growthStartEpoch - 40),
      growthWindowSize: profile.growthWindowSize ?? 40,
      growthImprovementThreshold: this.getInputNumber(this.elements.splitThresholdInput, profile.splitThreshold ?? 0.0015, 0),
      growthLossFloor: profile.growthLossFloor ?? 0.012,
      minHiddenNodes: profile.minHiddenNodes ?? 0,
      forcedGrowthInterval: profile.forcedGrowthInterval ?? 0,
      pruneInterval: this.getInputNumber(this.elements.pruneIntervalInput, profile.pruneInterval ?? 80, 20),
      pruneThreshold: this.getInputNumber(this.elements.pruneThresholdInput, profile.pruneThreshold ?? 0.01, 0.001)
    });
  }

  applyModelProfile() {
    const profile = this.currentModel.profile;
    if (!profile) return;

    this.elements.autoStopEpochInput.value = String(profile.autoStopEpoch ?? this.elements.autoStopEpochInput.value);
    this.elements.stopLossThresholdInput.value = String(profile.stopLossThreshold ?? this.elements.stopLossThresholdInput.value);
    this.elements.growthStartEpochInput.value = String(profile.growthStartEpoch ?? this.elements.growthStartEpochInput.value);
    this.elements.splitThresholdInput.value = String(profile.splitThreshold ?? this.elements.splitThresholdInput.value);
    this.elements.pruneIntervalInput.value = String(profile.pruneInterval ?? this.elements.pruneIntervalInput.value);
    this.elements.pruneThresholdInput.value = String(profile.pruneThreshold ?? this.elements.pruneThresholdInput.value);
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

  hasLearnedCurrentDataset() {
    return this.dataset.every((sample) => {
      const prediction = this.net.predict(sample.input);
      return sample.target.every((targetValue, index) => {
        const predictedValue = prediction[index];
        if (targetValue >= 0.5) return predictedValue > 0.9;
        return predictedValue < 0.1;
      });
    });
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
