import React, { useState } from 'react';
import { FLMethod } from '../types';
import { 
  X, 
  Layers, 
  Network, 
  ShieldCheck, 
  Cpu, 
  Check, 
  ArrowRight,
  Sparkles,
  Zap
} from 'lucide-react';

interface ArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentMethod: FLMethod;
  onSelectMethod: (method: FLMethod) => void;
}

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({
  isOpen,
  onClose,
  currentMethod,
  onSelectMethod,
}) => {
  const [activeTab, setActiveTab] = useState<FLMethod>(currentMethod);

  if (!isOpen) return null;

  const architectures: Record<FLMethod, {
    title: string;
    sub: string;
    paper: string;
    sharedLayers: string[];
    localLayers: string[];
    summary: string;
    mathFormula: string;
  }> = {
    fedbn: {
      title: 'FedBN: Local Batch Normalization',
      sub: 'Federated Learning On Non-IID Features Via Local Batch Normalization',
      paper: 'Li et al., ICLR 2021 (Cited 750+ times)',
      sharedLayers: ['Conv2d weights & biases', 'Residual shortcuts', 'Linear classification head'],
      localLayers: ['BN running_mean', 'BN running_var', 'BN gamma (scale)', 'BN beta (shift)'],
      summary: 'In colonoscopy imaging, illumination and optical sensors differ drastically between hospitals. FedBN shares representations while keeping BN layers local. This normalizes each hospital’s feature distributions into a canonical space without corrupting the global feature extractor.',
      mathFormula: 'w_{conv}^{global} = \\sum_{k=1}^K \\frac{n_k}{N} w_{conv}^k, \\quad \\theta_{bn}^k \\text{ stays local}',
    },
    fedrep: {
      title: 'FedRep: Representation Learning',
      sub: 'Exploiting Shared Representations for Personalized Federated Learning',
      paper: 'Collins et al., ICML 2021 (Cited 600+ times)',
      sharedLayers: ['Backbone / Trunk CNN layers', 'Feature extractor weights', 'Residual blocks'],
      localLayers: ['Final Linear Classification Head (Weights & Bias)'],
      summary: 'FedRep partitions the deep network into a shared global feature representation body and unique local decision heads. Clients alternately update the local head while holding the body fixed, then update the body while holding the head fixed.',
      mathFormula: 'w_{body}^{global} = \\text{FedAvg}(w_{body}^k), \\quad h^k \\text{ trained locally}',
    },
    fednova: {
      title: 'FedNova: Normalized Aggregation',
      sub: 'Tackling Objective Inconsistency in Heterogeneous Federated Optimization',
      paper: 'Wang et al., NeurIPS 2020 (Cited 800+ times)',
      sharedLayers: ['Full Model (Trunk + Head)'],
      localLayers: ['None (Global aggregation)'],
      summary: 'When hospitals train with different local epochs (e.g. Center 4 with 4x computational delay), standard FedAvg overweighs fast clients and drifts towards their local objective. FedNova scales updates inversely by tau_k = effective local steps.',
      mathFormula: '\\Delta w^{global} = \\sum_{k=1}^K p_k \\frac{\\tau_{eff}}{\\tau_k} \\Delta w^k',
    },
    fedavg: {
      title: 'FedAvg: Canonical Federated Averaging',
      sub: 'Communication-Efficient Learning of Deep Networks from Decentralized Data',
      paper: 'McMahan et al., AISTATS 2017 (Seminal Foundation)',
      sharedLayers: ['All Convolutional Layers', 'All Batch Normalization Layers', 'All Linear Layers'],
      localLayers: ['None (100% parameter synchronization)'],
      summary: 'The universal baseline in federated learning. Every client trains for E epochs locally on private data and transmits its full weight tensor. The central server computes a sample-weighted arithmetic mean.',
      mathFormula: 'W_{t+1} = \\sum_{k=1}^K \\frac{n_k}{N} W_{t+1}^k',
    },
    dpfedavg: {
      title: 'DP-FedAvg: Differential Privacy',
      sub: 'Opacus Rényi DP Accounting for Protected Patient Data',
      paper: 'Dwork et al. / PyTorch Opacus Integration',
      sharedLayers: ['Aggregated Noisy Model Parameters'],
      localLayers: ['Local gradient clip norm buffer'],
      summary: 'Guarantees provable patient differential privacy (epsilon, delta). Gradients are clipped to threshold C, then perturbed with calibrated Gaussian noise sigma before aggregation, preventing membership inference and model inversion attacks.',
      mathFormula: 'w_{DP} = W + \\frac{1}{K}\\sum_k (\\text{clip}(g_k, C) + \\mathcal{N}(0, \\sigma^2 C^2 I))',
    },
    local_only: {
      title: 'Local-Only: Independent Training',
      sub: 'No Communication Isolated Baseline',
      paper: 'Empirical Baseline',
      sharedLayers: ['None (0% Communication)'],
      localLayers: ['100% Local Model'],
      summary: 'Each medical center trains only on its private endoscopic data. Demonstrates the isolation failure mode where small centers overfit and fail on out-of-domain evaluation.',
      mathFormula: 'w^k_{t+1} = w^k_t - \\eta \\nabla L_k(w^k_t)',
    },
  };

  const arch = architectures[activeTab];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Algorithm Architecture & Parameter Routing</h3>
              <p className="text-xs text-slate-400">
                Understanding personalized vs shared model parameters across clinical endoscopy nodes
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Algorithm Tabs */}
        <div className="px-6 pt-4 border-b border-slate-800/80 bg-slate-950/40 flex items-center gap-2 overflow-x-auto">
          {(Object.keys(architectures) as FLMethod[]).map(method => (
            <button
              key={method}
              onClick={() => setActiveTab(method)}
              className={`px-3 py-2 text-xs font-bold border-b-2 transition-all whitespace-nowrap ${
                activeTab === method
                  ? 'border-indigo-500 text-indigo-400 bg-indigo-500/10 rounded-t-lg'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {method.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1">
          <div>
            <div className="flex items-center justify-between">
              <h4 className="text-lg font-bold text-white">{arch.title}</h4>
              <span className="text-xs font-medium text-slate-400 bg-slate-800 px-2.5 py-1 rounded-lg">
                {arch.paper}
              </span>
            </div>
            <p className="text-xs text-slate-300 mt-1">{arch.sub}</p>
          </div>

          {/* Visual Routing Flow Diagram */}
          <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
              Network Weight Routing & Aggregation Topology:
            </span>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              {/* Shared Component */}
              <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-900/50 space-y-2">
                <div className="font-bold text-indigo-400 flex items-center gap-1.5">
                  <Network className="w-4 h-4" />
                  <span>Shared Globally (Transmitted to Central Server)</span>
                </div>
                <ul className="space-y-1.5 text-slate-200">
                  {arch.sharedLayers.map((layer, idx) => (
                    <li key={idx} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                      <span>{layer}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Local / Personalized Component */}
              <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-900/50 space-y-2">
                <div className="font-bold text-rose-400 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Personalized Locally (Never Transmitted)</span>
                </div>
                <ul className="space-y-1.5 text-slate-200">
                  {arch.localLayers.map((layer, idx) => (
                    <li key={idx} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                      <span>{layer}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Mathematical Aggregation Formula */}
            <div className="p-3 bg-black/60 rounded-xl border border-slate-800 font-mono text-xs text-center text-slate-300">
              <span className="text-slate-500 mr-2">Aggregation Rule:</span>
              <span className="text-indigo-300 font-bold">{arch.mathFormula}</span>
            </div>
          </div>

          {/* Detailed Summary */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-300 leading-relaxed space-y-1.5">
            <span className="font-bold text-white block">Clinical Rationale:</span>
            <p>{arch.summary}</p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between">
          <button
            onClick={() => {
              onSelectMethod(activeTab);
              onClose();
            }}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md"
          >
            Apply & Launch {activeTab.toUpperCase()}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
