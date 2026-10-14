import React from 'react';
import { FLMethod } from '../types';
import { 
  Network, 
  Settings, 
  Layers, 
  HelpCircle, 
  Cpu, 
  Activity, 
  ShieldCheck, 
  FileCode,
  Sparkles
} from 'lucide-react';

interface HeaderProps {
  currentMethod: FLMethod;
  onSelectMethod: (method: FLMethod) => void;
  onOpenConfig: () => void;
  onOpenArchitecture: () => void;
  activeTab: 'simulation' | 'centers' | 'partition' | 'benchmarks';
  setActiveTab: (tab: 'simulation' | 'centers' | 'partition' | 'benchmarks') => void;
  isSimulating: boolean;
  currentRound: number;
  totalRounds: number;
}

export const Header: React.FC<HeaderProps> = ({
  currentMethod,
  onSelectMethod,
  onOpenConfig,
  onOpenArchitecture,
  activeTab,
  setActiveTab,
  isSimulating,
  currentRound,
  totalRounds,
}) => {
  const methodLabels: Record<FLMethod, { name: string; tag: string; color: string }> = {
    fedbn: { name: 'FedBN', tag: 'Local Batch Normalization', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
    fedrep: { name: 'FedRep', tag: 'Shared Representation', color: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40' },
    fednova: { name: 'FedNova', tag: 'Normalized Aggregation', color: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40' },
    fedavg: { name: 'FedAvg', tag: 'Canonical Baseline', color: 'bg-blue-500/20 text-blue-300 border-blue-500/40' },
    dpfedavg: { name: 'DP-FedAvg', tag: 'Differential Privacy', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
    local_only: { name: 'Local-Only', tag: 'No Collaboration', color: 'bg-rose-500/20 text-rose-300 border-rose-500/40' },
  };

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          {/* Logo & Identity */}
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Network className="w-5 h-5 text-indigo-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-white">PL_PolypGen</span>
                <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  FL-bench
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Personalized Federated Learning on Colonoscopy Centers
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-950/70 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('simulation')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'simulation'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Simulation</span>
              {isSimulating && (
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse ml-0.5" />
              )}
            </button>
            <button
              onClick={() => setActiveTab('centers')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'centers'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Centers (C1–C6)</span>
            </button>
            <button
              onClick={() => setActiveTab('partition')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'partition'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Data Partition & JSD</span>
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'benchmarks'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Benchmark Matrix</span>
            </button>
          </nav>

          {/* Action Tools */}
          <div className="flex items-center gap-2">
            {/* Algorithm Selector Dropdown */}
            <div className="relative">
              <select
                value={currentMethod}
                onChange={(e) => onSelectMethod(e.target.value as FLMethod)}
                className="bg-slate-800/90 hover:bg-slate-800 text-slate-100 text-xs font-semibold rounded-lg px-2.5 py-1.5 pr-8 border border-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500 cursor-pointer appearance-none"
              >
                <option value="fedbn">FedBN (Local Batchnorm)</option>
                <option value="fedrep">FedRep (Shared Body, Local Head)</option>
                <option value="fednova">FedNova (Normalized Averaging)</option>
                <option value="fedavg">FedAvg (Standard Averaging)</option>
                <option value="dpfedavg">DP-FedAvg (Opacus DP)</option>
                <option value="local_only">Local-Only (No Federated Comm)</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>

            {/* Architecture Explanation Button */}
            <button
              onClick={onOpenArchitecture}
              title="View Algorithm Architecture & Layer Routing"
              className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-colors"
            >
              <Layers className="w-4 h-4" />
            </button>

            {/* Hydra Config Button */}
            <button
              onClick={onOpenConfig}
              title="View & Export Hydra defaults.yaml"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs font-medium transition-colors"
            >
              <FileCode className="w-3.5 h-3.5 text-indigo-400" />
              <span className="hidden md:inline">Hydra Config</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
