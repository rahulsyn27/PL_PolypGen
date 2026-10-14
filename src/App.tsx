import React, { useState, useEffect } from 'react';
import { FLConfig, FLMethod, RoundMetrics } from './types';
import { simulatePartition, PartitionResult } from './utils/flSimulation';
import { Header } from './components/Header';
import { SimulationView } from './components/SimulationView';
import { PolypGenCenterExplorer } from './components/PolypGenCenterExplorer';
import { DatasetPartitionView } from './components/DatasetPartitionView';
import { BenchmarkComparisonView } from './components/BenchmarkComparisonView';
import { ConfigModal } from './components/ConfigModal';
import { ArchitectureModal } from './components/ArchitectureModal';
import { ExternalLink, GitBranch, Heart, Shield, CheckCircle2 } from 'lucide-react';

const DEFAULT_CONFIG: FLConfig = {
  method: 'fedbn',
  dataset: {
    name: 'polypgen',
    clientNum: 5,
    split: 'sample',
    valRatio: 0.1,
    testRatio: 0.2,
    partitionScheme: 'center',
    alpha: 0.5,
    iidRatio: 0.0,
    shardsPerClient: 2,
    classesPerClient: 2,
  },
  model: {
    name: 'resnet18',
    usePretrained: true,
    allParamsPersonalized: false,
  },
  common: {
    seed: 42,
    joinRatio: 1.0,
    globalEpoch: 50,
    localEpoch: 5,
    batchSize: 32,
    stragglerRatio: 0.2,
    buffers: 'local',
    clientSideEval: true,
    systemHeterogeneity: true,
  },
  optimizer: {
    name: 'sgd',
    lr: 0.01,
    weightDecay: 1e-4,
    momentum: 0.9,
  },
  lrScheduler: {
    name: 'step',
    stepSize: 15,
    gamma: 0.5,
  },
  differentialPrivacy: {
    enabled: false,
    epsilon: 5.0,
    delta: 1e-5,
    maxGradNorm: 1.0,
    noiseMultiplier: 1.1,
  },
  fedrep: {
    trainBodyEpoch: 1,
  },
  fednova: {
    aggregateNormalized: true,
  },
};

export const App: React.FC = () => {
  const [config, setConfig] = useState<FLConfig>(DEFAULT_CONFIG);
  const [partition, setPartition] = useState<PartitionResult>(() => simulatePartition(DEFAULT_CONFIG));
  const [activeTab, setActiveTab] = useState<'simulation' | 'centers' | 'partition' | 'benchmarks'>('simulation');
  
  // Simulation State
  const [currentRound, setCurrentRound] = useState<number>(0);
  const [history, setHistory] = useState<RoundMetrics[]>([]);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  // Modals
  const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);
  const [isArchOpen, setIsArchOpen] = useState<boolean>(false);

  // Re-run partition whenever dataset settings change
  const handleConfigChange = (newConfig: FLConfig) => {
    setConfig(newConfig);
    setPartition(simulatePartition(newConfig));
  };

  const handleSelectMethod = (newMethod: FLMethod) => {
    const updated: FLConfig = {
      ...config,
      method: newMethod,
      common: {
        ...config.common,
        buffers: newMethod === 'fedbn' ? 'local' : 'global',
      },
      differentialPrivacy: {
        ...config.differentialPrivacy,
        enabled: newMethod === 'dpfedavg',
      },
    };
    setConfig(updated);
    // Reset simulation history for fair comparison
    setIsSimulating(false);
    setCurrentRound(0);
    setHistory([]);
  };

  const handleRegeneratePartition = () => {
    setPartition(simulatePartition(config));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Header */}
      <Header
        currentMethod={config.method}
        onSelectMethod={handleSelectMethod}
        onOpenConfig={() => setIsConfigOpen(true)}
        onOpenArchitecture={() => setIsArchOpen(true)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isSimulating={isSimulating}
        currentRound={currentRound}
        totalRounds={config.common.globalEpoch}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'simulation' && (
          <SimulationView
            config={config}
            partition={partition}
            currentRound={currentRound}
            setCurrentRound={setCurrentRound}
            history={history}
            setHistory={setHistory}
            isSimulating={isSimulating}
            setIsSimulating={setIsSimulating}
          />
        )}

        {activeTab === 'centers' && <PolypGenCenterExplorer />}

        {activeTab === 'partition' && (
          <DatasetPartitionView
            config={config}
            onChangeConfig={handleConfigChange}
            partition={partition}
            onRegeneratePartition={handleRegeneratePartition}
          />
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarkComparisonView
            onSelectMethod={(m) => {
              handleSelectMethod(m);
              setActiveTab('simulation');
            }}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/60 py-6 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-300">PL_PolypGen</span>
            <span>•</span>
            <span>FL-bench Personalized Federated Learning Suite</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span className="flex items-center gap-1 text-slate-500">
              <span>Imported from</span>
              <span className="font-mono text-slate-400">rahulsyn27/PL_PolypGen</span>
            </span>
            <button
              onClick={() => setIsArchOpen(true)}
              className="hover:text-indigo-300 transition-colors"
            >
              Layer Architecture
            </button>
            <button
              onClick={() => setIsConfigOpen(true)}
              className="hover:text-indigo-300 transition-colors"
            >
              Hydra Defaults
            </button>
          </div>
        </div>
      </footer>

      {/* Modals */}
      <ConfigModal
        isOpen={isConfigOpen}
        onClose={() => setIsConfigOpen(false)}
        config={config}
        onChangeConfig={handleConfigChange}
      />

      <ArchitectureModal
        isOpen={isArchOpen}
        onClose={() => setIsArchOpen(false)}
        currentMethod={config.method}
        onSelectMethod={handleSelectMethod}
      />
    </div>
  );
};
