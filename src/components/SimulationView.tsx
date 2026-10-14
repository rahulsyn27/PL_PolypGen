import React, { useState, useEffect, useRef } from 'react';
import { FLConfig, RoundMetrics, ClientPackage, FLMethod } from '../types';
import { PartitionResult, simulateFLRound } from '../utils/flSimulation';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  SkipForward, 
  Activity, 
  TrendingUp, 
  ShieldCheck, 
  Layers, 
  Cpu, 
  CheckCircle2, 
  Clock, 
  Send,
  Terminal,
  Zap,
  BarChart,
  HardDrive
} from 'lucide-react';

interface SimulationViewProps {
  config: FLConfig;
  partition: PartitionResult;
  currentRound: number;
  setCurrentRound: React.Dispatch<React.SetStateAction<number>>;
  history: RoundMetrics[];
  setHistory: React.Dispatch<React.SetStateAction<RoundMetrics[]>>;
  isSimulating: boolean;
  setIsSimulating: React.Dispatch<React.SetStateAction<boolean>>;
}

export const SimulationView: React.FC<SimulationViewProps> = ({
  config,
  partition,
  currentRound,
  setCurrentRound,
  history,
  setHistory,
  isSimulating,
  setIsSimulating,
}) => {
  const [speed, setSpeed] = useState<number>(1);
  const [activePackage, setActivePackage] = useState<ClientPackage | null>(null);
  const [logMessages, setLogMessages] = useState<string[]>([
    `[INFO] FL-bench server initialized with algorithm: ${config.method.toUpperCase()}`,
    `[INFO] Target dataset: PolypGen (5 training centers, 1 held-out test center)`,
    `[INFO] System heterogeneity: ${config.common.systemHeterogeneity ? 'ENABLED (1x, 2x, 4x local epoch multipliers)' : 'DISABLED'}`,
    `[INFO] Ready for round 1 federated communication.`
  ]);

  const logContainerRef = useRef<HTMLDivElement>(null);
  const latestMetrics = history[history.length - 1] || null;

  // Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logMessages]);

  // Simulation execution loop
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isSimulating && currentRound < config.common.globalEpoch) {
      const delay = Math.max(120, 1000 / speed);
      timer = setTimeout(() => {
        executeOneRound();
      }, delay);
    } else if (currentRound >= config.common.globalEpoch && isSimulating) {
      setIsSimulating(false);
      setLogMessages(prev => [
        ...prev,
        `[SUCCESS] Federated experiment completed ${config.common.globalEpoch} rounds. Final Personalized Accuracy: ${latestMetrics?.globalTestAccuracy || 0}%.`
      ]);
    }
    return () => clearTimeout(timer);
  }, [isSimulating, currentRound, speed, config]);

  const executeOneRound = () => {
    const nextRound = currentRound + 1;
    const prevMetrics = history[history.length - 1];
    const { metrics, packages } = simulateFLRound(nextRound, config, partition, prevMetrics);

    setCurrentRound(nextRound);
    setHistory(prev => [...prev, metrics]);
    if (packages.length > 0) {
      setActivePackage(packages[0]);
    }

    // Add log
    const clientListStr = metrics.activeClients.map(c => `C${c}`).join(', ');
    const logText = `[ROUND ${nextRound}/${config.common.globalEpoch}] Selected: [${clientListStr}] | Loss: ${metrics.globalLoss.toFixed(3)} | Acc: ${metrics.globalTestAccuracy.toFixed(1)}% | OOD C6: ${metrics.oodCenter6Accuracy.toFixed(1)}%`;
    setLogMessages(prev => [...prev.slice(-30), logText]);
  };

  const handleReset = () => {
    setIsSimulating(false);
    setCurrentRound(0);
    setHistory([]);
    setActivePackage(null);
    setLogMessages([
      `[INFO] Resetting FL-bench server state.`,
      `[INFO] Algorithm: ${config.method.toUpperCase()} | Global Epochs: ${config.common.globalEpoch}`,
      `[INFO] Ready for new experiment run.`
    ]);
  };

  const methodBadge: Record<FLMethod, { title: string; color: string }> = {
    fedbn: { title: 'FedBN (Local Batch Normalization)', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
    fedrep: { title: 'FedRep (Shared Body, Local Head)', color: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40' },
    fednova: { title: 'FedNova (Normalized τ Aggregation)', color: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40' },
    fedavg: { title: 'FedAvg (Standard Averaging)', color: 'bg-blue-500/20 text-blue-300 border-blue-500/40' },
    dpfedavg: { title: 'DP-FedAvg (Opacus Rényi DP)', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
    local_only: { title: 'Local-Only (No Fed Communication)', color: 'bg-rose-500/20 text-rose-300 border-rose-500/40' },
  };

  return (
    <div className="space-y-6">
      {/* Control Toolbar */}
      <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center gap-3">
          {/* Play / Pause */}
          <button
            onClick={() => setIsSimulating(!isSimulating)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold shadow-lg transition-all ${
              isSimulating
                ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30'
            }`}
          >
            {isSimulating ? (
              <>
                <Pause className="w-4 h-4 fill-current" />
                <span>Pause Run</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>{currentRound === 0 ? 'Start Training' : 'Resume'}</span>
              </>
            )}
          </button>

          {/* Step Round */}
          <button
            onClick={executeOneRound}
            disabled={isSimulating || currentRound >= config.common.globalEpoch}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors"
          >
            <SkipForward className="w-3.5 h-3.5" />
            <span>Step Round</span>
          </button>

          {/* Reset */}
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700/80 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>

          {/* Speed Selector */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
            <span className="text-[10px] text-slate-400 px-2 font-medium">Speed:</span>
            {[1, 2, 5, 10].map(s => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2 py-0.5 rounded-lg text-xs font-semibold transition-colors ${
                  speed === s ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Current Round Indicator & Method Pill */}
        <div className="flex items-center gap-3">
          <span className={`text-xs font-bold px-3 py-1.5 rounded-xl border ${methodBadge[config.method].color}`}>
            {methodBadge[config.method].title}
          </span>
          <div className="text-right">
            <div className="text-[10px] text-slate-400">Communication Round</div>
            <div className="text-sm font-mono font-bold text-white">
              {currentRound} / {config.common.globalEpoch}
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Personalized Test Acc */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Hospital Acc</span>
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-white mt-1">
            {latestMetrics ? `${latestMetrics.globalTestAccuracy.toFixed(1)}%` : '--'}
          </div>
          <div className="text-[10px] text-emerald-400 mt-1">
            {latestMetrics && latestMetrics.globalTestAccuracy > 85 ? 'High Clinical Precision' : 'Training...'}
          </div>
        </div>

        {/* OOD Center 6 Acc */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>OOD (C6) Acc</span>
            <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-300 mt-1">
            {latestMetrics ? `${latestMetrics.oodCenter6Accuracy.toFixed(1)}%` : '--'}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">Unseen Center Generalization</div>
        </div>

        {/* Global Loss */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Global Loss</span>
            <BarChart className="w-3.5 h-3.5 text-rose-400" />
          </div>
          <div className="text-2xl font-bold text-slate-200 mt-1">
            {latestMetrics ? latestMetrics.globalLoss.toFixed(3) : '--'}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">Cross-Entropy</div>
        </div>

        {/* Macro F1 */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Macro F1 Score</span>
            <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-300 mt-1">
            {latestMetrics ? `${latestMetrics.macroF1.toFixed(1)}%` : '--'}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">Balanced Metric</div>
        </div>

        {/* Network Transfer */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Network Comm</span>
            <Send className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-purple-300 mt-1">
            {latestMetrics ? `${(latestMetrics.cumulativeCommunicationKb / 1024).toFixed(1)} MB` : '0 MB'}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">Weights Sent/Recv</div>
        </div>

        {/* Privacy Epsilon */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>DP Budget (ε)</span>
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-300 mt-1">
            {config.method === 'dpfedavg' || config.differentialPrivacy.enabled
              ? latestMetrics?.privacyEpsilonSpent ? `ε=${latestMetrics.privacyEpsilonSpent}` : 'Tracking'
              : 'N/A (No DP)'}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">Opacus RDP Accountant</div>
        </div>
      </div>

      {/* Main Charts & Inspector Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Convergence Chart */}
        <div className="lg:col-span-8 bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" />
                <span>Convergence & Domain Generalization Curves</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Real-time tracking of Personalized Hospital Accuracy vs Held-out Center 6 Generalization
              </p>
            </div>
            {/* Chart Legend */}
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-indigo-400" />
                <span className="text-slate-300">Personalized Test Acc</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-amber-400" />
                <span className="text-slate-300">OOD Center 6 Acc</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-rose-400" />
                <span className="text-slate-300">Loss</span>
              </span>
            </div>
          </div>

          {/* SVG Canvas Plot */}
          <div className="h-64 w-full bg-slate-950/70 rounded-xl border border-slate-800 p-2 relative flex flex-col justify-end">
            {history.length < 2 ? (
              <div className="h-full w-full flex flex-col items-center justify-center text-slate-500 text-xs">
                <Activity className="w-8 h-8 text-slate-600 mb-2 animate-pulse" />
                <span>Start or step rounds to render live convergence trajectory.</span>
              </div>
            ) : (
              <svg className="w-full h-full overflow-visible">
                {/* Horizontal grid lines */}
                {[0.25, 0.5, 0.75].map(ratio => (
                  <line
                    key={ratio}
                    x1="0"
                    y1={240 * ratio}
                    x2="100%"
                    y2={240 * ratio}
                    stroke="#1e293b"
                    strokeDasharray="4 4"
                  />
                ))}

                {/* Accuracy Path */}
                <path
                  d={history.reduce((acc, pt, i) => {
                    const x = (i / Math.max(1, config.common.globalEpoch - 1)) * 100;
                    const y = 240 - ((pt.globalTestAccuracy - 50) / 50) * 220;
                    return `${acc} ${i === 0 ? 'M' : 'L'} ${x}% ${y}`;
                  }, '')}
                  fill="none"
                  stroke="#818cf8"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />

                {/* OOD Center 6 Path */}
                <path
                  d={history.reduce((acc, pt, i) => {
                    const x = (i / Math.max(1, config.common.globalEpoch - 1)) * 100;
                    const y = 240 - ((pt.oodCenter6Accuracy - 50) / 50) * 220;
                    return `${acc} ${i === 0 ? 'M' : 'L'} ${x}% ${y}`;
                  }, '')}
                  fill="none"
                  stroke="#fbbf24"
                  strokeWidth="2"
                  strokeDasharray="4 2"
                  strokeLinecap="round"
                />

                {/* Loss Path (scaled) */}
                <path
                  d={history.reduce((acc, pt, i) => {
                    const x = (i / Math.max(1, config.common.globalEpoch - 1)) * 100;
                    const y = 240 - (1 - Math.min(1, pt.globalLoss)) * 200;
                    return `${acc} ${i === 0 ? 'M' : 'L'} ${x}% ${y}`;
                  }, '')}
                  fill="none"
                  stroke="#fb7185"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            )}

            <div className="flex justify-between text-[10px] text-slate-500 pt-2 px-1 border-t border-slate-800/80">
              <span>Round 1</span>
              <span>Round {Math.round(config.common.globalEpoch / 2)}</span>
              <span>Round {config.common.globalEpoch}</span>
            </div>
          </div>

          {/* Hospital Client Participation Cards */}
          <div className="pt-2">
            <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>Center Participation Status (This Round)</span>
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
              {partition.clientDistributions.map((client) => {
                const isActive = latestMetrics?.activeClients.includes(client.clientId);
                const acc = latestMetrics?.clientAccuracies[client.clientId];
                return (
                  <div
                    key={client.clientId}
                    className={`p-2.5 rounded-xl border text-xs transition-all ${
                      isActive
                        ? 'bg-indigo-950/40 border-indigo-500/50 text-white shadow-sm'
                        : 'bg-slate-950/40 border-slate-800 text-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold">{client.name}</span>
                      <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
                    </div>
                    <div className="text-[11px] font-mono">
                      Acc: {acc ? `${acc}%` : '--'}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      τ = {client.effectiveEpochs * 10} steps
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Weight Package Inspector & Terminal Log */}
        <div className="lg:col-span-4 space-y-4">
          {/* Weight Packaging Inspector */}
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>Parameter Transmission Inspector</span>
              </h3>
              <span className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-cyan-300">
                src/utils/weight_packaging.py
              </span>
            </div>

            {activePackage ? (
              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Reporting Client:</span>
                    <span className="font-semibold text-white">Center {activePackage.clientId}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Local Steps (τ):</span>
                    <span className="font-mono text-indigo-300">{activePackage.localStepsTau} steps</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Aggregation Weight (n_k):</span>
                    <span className="font-mono text-slate-200">{activePackage.sampleWeight} samples</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Client Val Accuracy:</span>
                    <span className="font-bold text-emerald-400">{activePackage.valAccuracy}%</span>
                  </div>
                </div>

                {/* Personalized vs Shared Layer Routing */}
                <div className="space-y-2">
                  <div>
                    <span className="text-[11px] font-semibold text-rose-400 block mb-1">
                      🔒 Kept Local (Personalized):
                    </span>
                    <div className="p-2 rounded-lg bg-rose-950/20 border border-rose-900/40 font-mono text-[10px] text-rose-200">
                      {activePackage.personalizedLayers.length > 0
                        ? activePackage.personalizedLayers.join(', ')
                        : 'None (Full global aggregation)'}
                    </div>
                  </div>

                  <div>
                    <span className="text-[11px] font-semibold text-indigo-400 block mb-1">
                      🌐 Transmitted to Global Server:
                    </span>
                    <div className="p-2 rounded-lg bg-indigo-950/20 border border-indigo-900/40 font-mono text-[10px] text-indigo-200">
                      {activePackage.sharedLayers.join(', ')}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-500 bg-slate-950/50 rounded-xl border border-slate-800">
                Execute a round to inspect client packages and parameter routing.
              </div>
            )}
          </div>

          {/* Rich Console Log */}
          <div className="bg-slate-900/80 p-4 rounded-2xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs font-bold text-white">
              <span className="flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                <span>FL-bench Server Logs</span>
              </span>
              <span className="text-[10px] text-slate-500">Live stdout</span>
            </div>
            <div
              ref={logContainerRef}
              className="h-44 bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-300 overflow-y-auto space-y-1"
            >
              {logMessages.map((msg, i) => (
                <div
                  key={i}
                  className={`${
                    msg.includes('[SUCCESS]')
                      ? 'text-emerald-400 font-bold'
                      : msg.includes('[ROUND')
                      ? 'text-indigo-300'
                      : 'text-slate-400'
                  }`}
                >
                  {msg}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
