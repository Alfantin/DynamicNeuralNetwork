import { DynamicNeuralNetwork } from '../network/DynamicNeuralNetwork.js';
import { GraphRenderer } from './GraphRenderer.js';

export class AppController {
  constructor(elements, dataset) {
    this.elements = elements;
    this.dataset = dataset;
    this.autoTraining = false;
    this.autoHandle = null;
    this.net = new DynamicNeuralNetwork(2, 1, 0.03);
    this.renderer = new GraphRenderer(elements.graphCanvas, this.net);
  }

  init() {
    this.bindEvents();
    this.updateStats(null);
  }

  bindEvents() {
    this.elements.stepBtn.addEventListener('click', () => this.runEpochs(1));
    this.elements.train100Btn.addEventListener('click', () => this.runEpochs(100));
    this.elements.autoBtn.addEventListener('click', () => this.startAuto());
    this.elements.stopBtn.addEventListener('click', () => this.stopAuto());
    this.elements.resetBtn.addEventListener('click', () => this.resetAll());
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
  }

  startAuto() {
    if (this.autoTraining) return;

    this.autoTraining = true;
    this.elements.autoBtn.disabled = true;
    this.elements.stopBtn.disabled = false;
    this.elements.stepBtn.disabled = true;
    this.elements.train100Btn.disabled = true;

    this.autoHandle = setInterval(() => {
      this.runEpochs(8);
      const preds = this.dataset.map((sample) => this.net.predict(sample.input)[0]);
      const learnedXor = preds[0] < 0.08 && preds[1] > 0.92 && preds[2] > 0.92 && preds[3] < 0.08;
      if (learnedXor || this.net.epoch >= 5000) this.stopAuto();
    }, 35);
  }

  stopAuto() {
    this.autoTraining = false;
    clearInterval(this.autoHandle);
    this.autoHandle = null;

    this.elements.autoBtn.disabled = false;
    this.elements.stopBtn.disabled = true;
    this.elements.stepBtn.disabled = false;
    this.elements.train100Btn.disabled = false;
  }

  resetAll() {
    this.stopAuto();
    this.net = new DynamicNeuralNetwork(2, 1, 0.03);
    this.renderer.setNetwork(this.net);
    this.updateStats(null);
  }
}
