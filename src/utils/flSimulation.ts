import { FLConfig, FLMethod, RoundMetrics, ClientPackage } from '../types';
import { POLYPGEN_CENTERS } from '../data/polypgenData';

/**
 * Approximate Gamma random variable for Dirichlet sampling
 */
function sampleGamma(shape: number, scale: number = 1.0): number {
  if (shape < 1) {
    return sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let z = 0;
    let u = 0;
    let v = 0;
    do {
      // Box-Muller transform for standard normal
      const u1 = Math.random();
      const u2 = Math.random();
      z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      v = 1 + c * z;
    } while (v <= 0);

    v = v * v * v;
    u = Math.random();
    if (u < 1 - 0.0331 * z * z * z * z) {
      return d * v * scale;
    }
    if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) {
      return d * v * scale;
    }
  }
}

/**
 * Sample Dirichlet vector
 */
export function sampleDirichlet(alpha: number, numClients: number): number[] {
  const gammas = Array.from({ length: numClients }, () => Math.max(0.0001, sampleGamma(alpha)));
  const sum = gammas.reduce((a, b) => a + b, 0);
  return gammas.map(g => g / sum);
}

/**
 * Jensen-Shannon Divergence between two probability distributions
 * Matches src/utils/statistical_distances.py
 */
export function computeJSD(p: number[], q: number[]): number {
  const m = p.map((pi, idx) => 0.5 * (pi + q[idx]));
  const kl = (a: number[], b: number[]) => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] > 1e-9 && b[i] > 1e-9) {
        sum += a[i] * Math.log2(a[i] / b[i]);
      }
    }
    return sum;
  };
  const jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m);
  return Math.max(0, Math.min(1, jsd));
}

/**
 * Compute average pairwise Jensen-Shannon Divergence for client distributions
 */
export function calculateAverageJSD(clientDistributions: number[][]): number {
  const n = clientDistributions.length;
  if (n < 2) return 0;
  let totalJsd = 0;
  let pairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      totalJsd += computeJSD(clientDistributions[i], clientDistributions[j]);
      pairs++;
    }
  }
  return pairs > 0 ? totalJsd / pairs : 0;
}

/**
 * Partition simulation for PolypGen and standard datasets
 */
export interface PartitionResult {
  clientDistributions: {
    clientId: number;
    name: string;
    positiveCount: number;
    negativeCount: number;
    totalCount: number;
    positiveRatio: number;
    systemMultiplier: number;
    effectiveEpochs: number;
    isStraggler: boolean;
  }[];
  avgJsd: number;
}

export function simulatePartition(config: FLConfig): PartitionResult {
  const { dataset, common } = config;
  const numClients = dataset.name === 'polypgen' ? 5 : dataset.clientNum; // C1-C5 (C6 held-out)

  if (dataset.partitionScheme === 'center' && dataset.name === 'polypgen') {
    // Natural clinical multi-center distribution
    const clients = POLYPGEN_CENTERS.filter(c => c.role === 'train').map(c => {
      const multiplier = common.systemHeterogeneity ? c.localEpochsMultiplier : 1;
      return {
        clientId: c.id,
        name: c.name,
        positiveCount: c.polypPositiveCount,
        negativeCount: c.polypNegativeCount,
        totalCount: c.sampleCount,
        positiveRatio: c.polypPositiveCount / c.sampleCount,
        systemMultiplier: multiplier,
        effectiveEpochs: common.localEpoch * multiplier,
        isStraggler: common.stragglerRatio > 0 && c.isStraggler,
      };
    });

    const dists = clients.map(c => [c.positiveRatio, 1 - c.positiveRatio]);
    return {
      clientDistributions: clients,
      avgJsd: calculateAverageJSD(dists),
    };
  }

  // Dirichlet or IID synthetic partition
  const totalSamples = dataset.name === 'polypgen' ? 2260 : numClients * 400;
  const basePositiveRatio = 0.52;
  const totalPos = Math.round(totalSamples * basePositiveRatio);
  const totalNeg = totalSamples - totalPos;

  let posShares: number[];
  let negShares: number[];

  if (dataset.partitionScheme === 'iid' || (dataset.iidRatio && dataset.iidRatio >= 0.99)) {
    posShares = Array(numClients).fill(1 / numClients);
    negShares = Array(numClients).fill(1 / numClients);
  } else {
    // Dirichlet skew with alpha
    const alpha = dataset.alpha || 0.5;
    posShares = sampleDirichlet(alpha, numClients);
    negShares = sampleDirichlet(alpha * 1.5, numClients);
  }

  const clients = Array.from({ length: numClients }, (_, i) => {
    const pos = Math.max(10, Math.round(totalPos * posShares[i]));
    const neg = Math.max(10, Math.round(totalNeg * negShares[i]));
    const total = pos + neg;
    const isStraggler = common.stragglerRatio > 0 && i === numClients - 1;
    const multiplier = common.systemHeterogeneity ? (i % 2 === 0 ? 1 : i === 1 ? 2 : 4) : 1;
    return {
      clientId: i + 1,
      name: `Client ${i + 1}`,
      positiveCount: pos,
      negativeCount: neg,
      totalCount: total,
      positiveRatio: pos / total,
      systemMultiplier: multiplier,
      effectiveEpochs: common.localEpoch * multiplier,
      isStraggler,
    };
  });

  const dists = clients.map(c => [c.positiveRatio, 1 - c.positiveRatio]);
  return {
    clientDistributions: clients,
    avgJsd: calculateAverageJSD(dists),
  };
}

/**
 * Step generator for federated learning round
 */
export function simulateFLRound(
  currentRound: number,
  config: FLConfig,
  partition: PartitionResult,
  previousMetrics?: RoundMetrics
): { metrics: RoundMetrics; packages: ClientPackage[] } {
  const { method, common, differentialPrivacy } = config;
  const totalRounds = common.globalEpoch;
  const progress = Math.min(1.0, currentRound / Math.max(1, totalRounds));
  
  // Asymptotic convergence formula with method-specific rate & asymptotes
  const methodProfiles: Record<FLMethod, {
    maxAcc: number;
    rate: number;
    oodMax: number;
    commKb: number;
    noiseLoss: number;
  }> = {
    fedbn: { maxAcc: 92.5, rate: 0.12, oodMax: 86.0, commKb: 1840, noiseLoss: 0.02 },
    fedrep: { maxAcc: 91.2, rate: 0.10, oodMax: 83.2, commKb: 1720, noiseLoss: 0.03 },
    fednova: { maxAcc: 88.0, rate: 0.09, oodMax: 82.5, commKb: 1890, noiseLoss: 0.02 },
    fedavg: { maxAcc: 85.0, rate: 0.08, oodMax: 79.0, commKb: 1890, noiseLoss: 0.04 },
    dpfedavg: { maxAcc: 81.5, rate: 0.07, oodMax: 75.5, commKb: 1890, noiseLoss: 0.08 },
    local_only: { maxAcc: 79.0, rate: 0.15, oodMax: 63.0, commKb: 0, noiseLoss: 0.05 },
  };

  const profile = methodProfiles[method] || methodProfiles.fedavg;

  // Active participating clients this round
  const allClientIds = partition.clientDistributions.map(c => c.clientId);
  const sampleSize = Math.max(1, Math.round(allClientIds.length * common.joinRatio));
  // Deterministic shuffle using round
  const shuffled = [...allClientIds].sort((a, b) => {
    return Math.sin(a * 100 + currentRound) - Math.sin(b * 100 + currentRound);
  });
  const activeClients = shuffled.slice(0, sampleSize);

  // Convergence curves
  const expFactor = 1 - Math.exp(-profile.rate * currentRound);
  const baseAcc = 52.0 + (profile.maxAcc - 52.0) * expFactor;
  // Natural stochastic fluctuation
  const noise = (Math.sin(currentRound * 1.7) * 0.4 + (Math.random() - 0.5) * 0.3) * (1 - progress * 0.7);
  const globalTestAcc = Math.min(99.0, Math.max(50.0, baseAcc + noise));
  const globalTrainAcc = Math.min(99.5, globalTestAcc + 2.5 * (1 - progress * 0.5));

  // OOD Center 6 generalization
  const oodExp = 1 - Math.exp(-profile.rate * 0.85 * currentRound);
  const oodBase = 49.0 + (profile.oodMax - 49.0) * oodExp;
  const oodCenter6Acc = Math.min(98.0, Math.max(48.0, oodBase + noise * 0.8));

  // Global loss
  const startLoss = 0.82;
  const targetLoss = 0.14 + profile.noiseLoss;
  const loss = targetLoss + (startLoss - targetLoss) * Math.exp(-profile.rate * 1.1 * currentRound) + Math.abs(noise) * 0.02;

  // Client-specific accuracies
  const clientAccuracies: Record<number, number> = {};
  const clientLosses: Record<number, number> = {};
  const packages: ClientPackage[] = [];

  for (const client of partition.clientDistributions) {
    const isActive = activeClients.includes(client.clientId);
    // Personalization boost for FedBN / FedRep
    let clientBonus = 0;
    if (method === 'fedbn') {
      clientBonus = 1.8 + Math.cos(client.clientId * 2) * 1.2;
    } else if (method === 'fedrep') {
      clientBonus = 1.0 + Math.sin(client.clientId * 1.5) * 0.9;
    } else if (method === 'local_only') {
      // Local only overfits to local center
      clientBonus = client.totalCount > 450 ? 2.5 : -4.0;
    }

    const cAcc = Math.min(99.0, Math.max(50.0, globalTestAcc + clientBonus + (Math.sin(client.clientId + currentRound) * 0.6)));
    const cLoss = Math.max(0.08, loss - clientBonus * 0.015);
    clientAccuracies[client.clientId] = Number(cAcc.toFixed(1));
    clientLosses[client.clientId] = Number(cLoss.toFixed(3));

    if (isActive) {
      packages.push({
        clientId: client.clientId,
        sampleWeight: client.totalCount,
        localStepsTau: client.effectiveEpochs * Math.ceil(client.totalCount / config.common.batchSize),
        trainLoss: Number(cLoss.toFixed(3)),
        valAccuracy: Number(cAcc.toFixed(1)),
        personalizedLayers: method === 'fedbn' 
          ? ['bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn2.*']
          : method === 'fedrep'
          ? ['linear_head.weight', 'linear_head.bias']
          : [],
        sharedLayers: method === 'fedbn'
          ? ['conv1.*', 'conv2.*', 'fc.*']
          : method === 'fedrep'
          ? ['backbone.conv.*', 'backbone.resblock.*']
          : ['all_layers'],
      });
    }
  }

  // Precision, recall, F1 based on accuracy and medical sensitivity
  const macroRecall = Math.min(99.0, globalTestAcc * 0.98 + (Math.random() - 0.5) * 0.5);
  const macroPrecision = Math.min(99.0, globalTestAcc * 0.96 + (Math.random() - 0.5) * 0.5);
  const macroF1 = (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall);

  // Cumulative communication
  const prevComm = previousMetrics ? previousMetrics.cumulativeCommunicationKb : 0;
  const currentComm = prevComm + profile.commKb * activeClients.length;

  // DP budget accounting
  let epsilonSpent: number | undefined = undefined;
  if (differentialPrivacy.enabled || method === 'dpfedavg') {
    const q = common.batchSize / 500;
    const sigma = differentialPrivacy.noiseMultiplier || 1.1;
    // Standard RDP accountant accumulation approximation
    const prevEps = previousMetrics?.privacyEpsilonSpent || 0.15;
    epsilonSpent = Number((prevEps + (q * Math.sqrt(2 * Math.log(1 / differentialPrivacy.delta))) / sigma).toFixed(2));
  }

  const metrics: RoundMetrics = {
    round: currentRound,
    globalLoss: Number(loss.toFixed(3)),
    globalTrainAccuracy: Number(globalTrainAcc.toFixed(1)),
    globalTestAccuracy: Number(globalTestAcc.toFixed(1)),
    oodCenter6Accuracy: Number(oodCenter6Acc.toFixed(1)),
    clientAccuracies,
    clientLosses,
    activeClients,
    macroPrecision: Number(macroPrecision.toFixed(1)),
    macroRecall: Number(macroRecall.toFixed(1)),
    macroF1: Number(macroF1.toFixed(1)),
    microPrecision: Number((macroPrecision + 0.3).toFixed(1)),
    microRecall: Number((macroRecall + 0.2).toFixed(1)),
    cumulativeCommunicationKb: currentComm,
    privacyEpsilonSpent: epsilonSpent,
  };

  return { metrics, packages };
}

/**
 * Generate Hydra YAML configuration matching config/defaults.yaml
 */
export function generateHydraConfigYaml(config: FLConfig): string {
  return `# Generated Hydra Configuration for FL-bench / PL_PolypGen
# Reproduce via: python main.py method=${config.method} dataset.name=${config.dataset.name}

method: ${config.method}

dataset:
  name: ${config.dataset.name}

model:
  name: ${config.model.name}
  use_torchvision_pretrained_weights: ${config.model.usePretrained}
  external_model_weights_path: null

lr_scheduler:
  name: ${config.lrScheduler.name === 'none' ? 'null' : config.lrScheduler.name}
  step_size: ${config.lrScheduler.stepSize}
  gamma: ${config.lrScheduler.gamma}

optimizer:
  name: ${config.optimizer.name}
  lr: ${config.optimizer.lr}
  weight_decay: ${config.optimizer.weightDecay}
  momentum: ${config.optimizer.momentum}

common:
  seed: ${config.common.seed}
  join_ratio: ${config.common.joinRatio}
  global_epoch: ${config.common.globalEpoch}
  local_epoch: ${config.common.localEpoch}
  batch_size: ${config.common.batchSize}
  reset_optimizer_on_global_epoch: true
  straggler_ratio: ${config.common.stragglerRatio}
  straggler_min_local_epoch: 0
  buffers: ${config.common.buffers}
  client_side_evaluation: ${config.common.clientSideEval}
  system_heterogeneity: ${config.common.systemHeterogeneity}

${config.method === 'fedrep' ? `fedrep:
  train_body_epoch: ${config.fedrep?.trainBodyEpoch || 1}` : ''}
${config.method === 'dpfedavg' || config.differentialPrivacy.enabled ? `dpfedavg:
  epsilon: ${config.differentialPrivacy.epsilon}
  delta: ${config.differentialPrivacy.delta}
  max_grad_norm: ${config.differentialPrivacy.maxGradNorm}
  noise_multiplier: ${config.differentialPrivacy.noiseMultiplier}` : ''}
`;
}
