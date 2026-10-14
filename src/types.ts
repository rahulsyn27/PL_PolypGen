export type FLMethod = 'fedavg' | 'fedbn' | 'fedrep' | 'fednova' | 'dpfedavg' | 'local_only';

export type DatasetName = 'polypgen' | 'cifar10' | 'cifar100' | 'svhn' | 'usps';

export type PartitionScheme = 'center' | 'dirichlet' | 'iid' | 'shards' | 'classes' | 'semantic';

export type OptimizerName = 'sgd' | 'adam' | 'adamw' | 'rmsprop' | 'adagrad';

export type LrSchedulerName = 'step' | 'cosine' | 'constant' | 'plateau' | 'none';

export interface FLConfig {
  method: FLMethod;
  dataset: {
    name: DatasetName;
    clientNum: number;
    split: 'user' | 'sample';
    valRatio: number;
    testRatio: number;
    partitionScheme: PartitionScheme;
    alpha: number; // for Dirichlet
    iidRatio: number;
    shardsPerClient: number;
    classesPerClient: number;
    superClass?: boolean;
  };
  model: {
    name: 'resnet18' | 'resnet34' | 'lenet5' | 'unet' | 'vgg11';
    usePretrained: boolean;
    allParamsPersonalized: boolean;
  };
  common: {
    seed: number;
    joinRatio: number; // e.g. 1.0 (all 5 centers) or 0.6
    globalEpoch: number; // rounds e.g. 50
    localEpoch: number; // client local epochs
    batchSize: number;
    stragglerRatio: number;
    buffers: 'global' | 'local' | 'drop';
    clientSideEval: boolean;
    systemHeterogeneity: boolean; // computational multipliers [1, 2, 4]
  };
  optimizer: {
    name: OptimizerName;
    lr: number;
    weightDecay: number;
    momentum: number;
  };
  lrScheduler: {
    name: LrSchedulerName;
    stepSize: number;
    gamma: number;
  };
  differentialPrivacy: {
    enabled: boolean;
    epsilon: number;
    delta: number;
    maxGradNorm: number;
    noiseMultiplier: number;
  };
  fedrep?: {
    trainBodyEpoch: number;
  };
  fednova?: {
    aggregateNormalized: boolean;
  };
}

export interface ClientParty {
  id: number;
  name: string;
  hospitalName: string;
  endoscopeModel: string;
  sampleCount: number;
  polypPositiveCount: number;
  polypNegativeCount: number;
  localEpochsMultiplier: number; // 1, 2, or 4 from apply_system_heterogeneity
  isStraggler: boolean;
  role: 'train' | 'test' | 'val' | 'ood_test';
  sensorCharacteristics: {
    colorTone: string;
    lighting: string;
    sharpness: string;
  };
}

export interface PolypSample {
  id: string;
  centerId: number;
  hasPolyp: boolean;
  polypType?: 'Adenoma' | 'Hyperplastic' | 'Serrated' | 'None';
  sizeMm?: number;
  imageColor: string;
  description: string;
  maskBox?: { x: number; y: number; width: number; height: number }; // normalized coords
}

export interface RoundMetrics {
  round: number;
  globalLoss: number;
  globalTrainAccuracy: number;
  globalTestAccuracy: number;
  oodCenter6Accuracy: number; // Center 6 out-of-domain evaluation
  clientAccuracies: Record<number, number>; // client_id -> accuracy %
  clientLosses: Record<number, number>;
  activeClients: number[];
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  microPrecision: number;
  microRecall: number;
  cumulativeCommunicationKb: number;
  privacyEpsilonSpent?: number;
}

export interface ClientPackage {
  clientId: number;
  sampleWeight: number;
  localStepsTau: number;
  trainLoss: number;
  valAccuracy: number;
  personalizedLayers: string[];
  sharedLayers: string[];
}

export interface BenchmarkResult {
  method: FLMethod;
  name: string;
  description: string;
  paperRef: string;
  personalizedAvgAccuracy: number;
  globalTestAccuracy: number;
  oodCenter6Accuracy: number;
  convergenceRound: number;
  commKbPerRound: number;
  privacyBudgetEpsilon: number | null;
  features: string[];
  clientAccuracyBreakdown: Record<number, number>;
  metricsHistory: {
    round: number;
    accuracy: number;
    loss: number;
    oodAccuracy: number;
  }[];
}
