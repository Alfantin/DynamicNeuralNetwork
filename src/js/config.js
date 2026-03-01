function createScenario(key, label, description, dataset, profile = {}) {
  return {
    key,
    label,
    description,
    dataset,
    inputCount: dataset[0].input.length,
    outputCount: dataset[0].target.length,
    profile
  };
}

const LOGIC_INPUTS = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];

const THREE_BIT_INPUTS = [
  [0, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [1, 1, 1]
];

function buildBinaryScenario(outputs) {
  return LOGIC_INPUTS.map((input, index) => ({
    input,
    target: [outputs[index]]
  }));
}

function buildThreeBitScenario(outputBuilder) {
  return THREE_BIT_INPUTS.map((input) => ({
    input,
    target: outputBuilder(input)
  }));
}

export const MODEL_OPTIONS = [
  createScenario(
    'xor',
    'XOR Baseline',
    'Baseline reference case: output becomes 1 when exactly one of two inputs is active.',
    buildBinaryScenario([0, 1, 1, 0]),
    {
      autoStopEpoch: 1000,
      stopLossThreshold: 0.008,
      growthStartEpoch: 120,
      growthWindowSize: 40,
      growthLossFloor: 0.012,
      splitThreshold: 0.0015,
      minHiddenNodes: 0,
      forcedGrowthInterval: 0,
      pruneInterval: 80,
      pruneThreshold: 0.01,
      requiredRefinementWindows: 12,
      minTargetEpoch: 0
    }
  ),
  createScenario(
    'smart_lamp',
    'Smart Lamp',
    'Lighting turns on only in specific non-linear combinations of three signals.',
    buildThreeBitScenario(([wallSwitch, motion, daylight]) => {
      const lampOn = (wallSwitch + motion + daylight) % 2;
      return [lampOn];
    }),
    {
      autoStopEpoch: 2200,
      stopLossThreshold: 0.003,
      growthStartEpoch: 24,
      growthWindowSize: 20,
      growthLossFloor: 0.004,
      splitThreshold: 0.005,
      minHiddenNodes: 2,
      forcedGrowthInterval: 20,
      pruneInterval: 36,
      pruneThreshold: 0.012,
      requiredRefinementWindows: 22,
      minTargetEpoch: 260
    }
  ),
  createScenario(
    'double_approval',
    'Dual Approval',
    'A request proceeds only when exactly two of the three approval sources agree.',
    buildThreeBitScenario(([manager, finance, risk]) => {
      const approval = manager + finance + risk === 2 ? 1 : 0;
      return [approval];
    }),
    {
      autoStopEpoch: 2400,
      stopLossThreshold: 0.003,
      growthStartEpoch: 24,
      growthWindowSize: 20,
      growthLossFloor: 0.004,
      splitThreshold: 0.005,
      minHiddenNodes: 2,
      forcedGrowthInterval: 18,
      pruneInterval: 34,
      pruneThreshold: 0.012,
      requiredRefinementWindows: 22,
      minTargetEpoch: 260
    }
  ),
  createScenario(
    'alarm_trigger',
    'Alarm Trigger',
    'Door, motion, and heat signals combine into alarm and lock outputs under risky patterns.',
    buildThreeBitScenario(([door, motion, heat]) => {
      const alarm = (door ^ motion) || (motion && heat) ? 1 : 0;
      const lock = door && heat ? 1 : 0;
      return [alarm, lock];
    }),
    {
      autoStopEpoch: 2600,
      stopLossThreshold: 0.003,
      growthStartEpoch: 20,
      growthWindowSize: 18,
      growthLossFloor: 0.003,
      splitThreshold: 0.0055,
      minHiddenNodes: 3,
      forcedGrowthInterval: 16,
      pruneInterval: 30,
      pruneThreshold: 0.013,
      requiredRefinementWindows: 24,
      minTargetEpoch: 320
    }
  ),
  createScenario(
    'health_risk',
    'Health Risk Alert',
    'Uses temperature, oxygen, and pulse to estimate risk and urgent response.',
    [
      { input: [0, 0, 0], target: [0, 0] },
      { input: [1, 0, 0], target: [1, 0] },
      { input: [1, 1, 0], target: [1, 1] },
      { input: [1, 1, 1], target: [1, 1] },
      { input: [0, 1, 1], target: [1, 1] },
      { input: [0, 0, 1], target: [0, 1] }
    ],
    {
      autoStopEpoch: 1800,
      stopLossThreshold: 0.0035,
      growthStartEpoch: 20,
      growthWindowSize: 18,
      growthLossFloor: 0.003,
      splitThreshold: 0.0055,
      minHiddenNodes: 3,
      forcedGrowthInterval: 16,
      pruneInterval: 28,
      pruneThreshold: 0.013,
      requiredRefinementWindows: 24,
      minTargetEpoch: 320
    }
  ),
  createScenario(
    'weather_signal',
    'Weather Forecast',
    'Uses humidity, pressure, and wind to predict rain and storm signals.',
    [
      { input: [0, 1, 0], target: [0, 0] },
      { input: [1, 1, 0], target: [1, 0] },
      { input: [1, 0, 1], target: [1, 1] },
      { input: [0, 0, 1], target: [0, 1] },
      { input: [1, 1, 1], target: [1, 1] },
      { input: [0, 1, 1], target: [0, 1] }
    ],
    {
      autoStopEpoch: 1800,
      stopLossThreshold: 0.0035,
      growthStartEpoch: 20,
      growthWindowSize: 18,
      growthLossFloor: 0.003,
      splitThreshold: 0.0055,
      minHiddenNodes: 3,
      forcedGrowthInterval: 16,
      pruneInterval: 28,
      pruneThreshold: 0.013,
      requiredRefinementWindows: 24,
      minTargetEpoch: 320
    }
  )
];

export const SELECTORS = {
  modelSelect: 'modelSelect',
  modelDesc: 'modelDesc',
  epochVal: 'epochVal',
  lossVal: 'lossVal',
  hiddenVal: 'hiddenVal',
  edgeVal: 'edgeVal',
  growVal: 'growVal',
  pruneVal: 'pruneVal',
  resultBody: 'resultBody',
  autoStopEpochInput: 'autoStopEpochInput',
  stopLossThresholdInput: 'stopLossThresholdInput',
  growthStartEpochInput: 'growthStartEpochInput',
  splitThresholdInput: 'splitThresholdInput',
  pruneIntervalInput: 'pruneIntervalInput',
  pruneThresholdInput: 'pruneThresholdInput',
  autoBtn: 'autoBtn',
  stopBtn: 'stopBtn',
  resetBtn: 'resetBtn',
  graphCanvas: 'graphCanvas'
};
