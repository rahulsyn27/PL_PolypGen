import React, { useState } from 'react';
import { FLConfig, PartitionScheme } from '../types';
import { PartitionResult } from '../utils/flSimulation';
import { 
  Sliders, 
  BarChart2, 
  Flame, 
  Layers, 
  Cpu, 
  CheckCircle, 
  AlertTriangle, 
  FileImage,
  RefreshCw
} from 'lucide-react';

interface DatasetPartitionViewProps {
  config: FLConfig;
  onChangeConfig: (newConfig: FLConfig) => void;
  partition: PartitionResult;
  onRegeneratePartition: () => void;
}

export const DatasetPartitionView: React.FC<DatasetPartitionViewProps> = ({
  config,
  onChangeConfig,
  partition,
  onRegeneratePartition,
}) => {
  const [selectedImageTab, setSelectedImageTab] = useState<'dir0.1' | 'iid' | 'iid0.5' | 'shards' | 'semantic'>('dir0.1');

  const distributionImages: Record<string, { label: string; src: string; desc: string }> = {
    'dir0.1': {
      label: 'Dirichlet(α=0.1)',
      src: '/distributions/dir0.1.png',
      desc: 'Extreme label heterogeneity across clients simulated via Dirichlet distribution with α=0.1.',
    },
    'iid': {
      label: 'IID (Uniform)',
      src: '/distributions/iid.png',
      desc: 'Identical and independently distributed samples across all participating clients.',
    },
    'iid0.5': {
      label: 'IID 50% + Dir(0.1)',
      src: '/distributions/iid0.5-a0.1.png',
      desc: 'Hybrid distribution: 50% data partitioned IID and 50% partitioned via Dirichlet(0.1).',
    },
    'shards': {
      label: 'Shards (2 shards)',
      src: '/distributions/shards2.png',
      desc: 'Pathological partition allocating fixed number of label shards per client.',
    },
    'semantic': {
      label: 'Semantic Partition',
      src: '/distributions/semantic.png',
      desc: 'Feature-based non-IID partitioning via PCA and pretrained EfficientNet embeddings.',
    },
  };

  const handleAlphaChange = (newAlpha: number) => {
    onChangeConfig({
      ...config,
      dataset: {
        ...config.dataset,
        alpha: newAlpha,
        partitionScheme: 'dirichlet',
      },
    });
  };

  const handleSchemeChange = (scheme: PartitionScheme) => {
    onChangeConfig({
      ...config,
      dataset: {
        ...config.dataset,
        partitionScheme: scheme,
      },
    });
  };

  const handleSystemHeterogeneityToggle = (enabled: boolean) => {
    onChangeConfig({
      ...config,
      common: {
        ...config.common,
        systemHeterogeneity: enabled,
      },
    });
  };

  const handleStragglerChange = (ratio: number) => {
    onChangeConfig({
      ...config,
      common: {
        ...config.common,
        stragglerRatio: ratio,
      },
    });
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <Layers className="w-4 h-4" />
              </span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Non-IID Partitioning & Statistical Heterogeneity
              </h2>
            </div>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl">
              Implements <code>generate_data.py</code>, <code>src/utils/heterogeneity.py</code>, and <code>src/utils/statistical_distances.py</code>. Analyze label skew via Dirichlet distribution, compute Jensen-Shannon Divergence ($JSD$), and simulate system speed variations.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="px-3 py-2 rounded-xl bg-slate-950/80 border border-slate-700/80 text-xs">
              <div className="text-slate-400">Mean Pairwise JSD</div>
              <div className="text-base font-bold text-indigo-300">
                {partition.avgJsd.toFixed(4)}
              </div>
            </div>
            <button
              onClick={onRegeneratePartition}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition-all"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Resample Partition</span>
            </button>
          </div>
        </div>
      </div>

      {/* Interactive Controls & Schemes */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls Column */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-5">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <span>Partition Strategy</span>
            </h3>

            {/* Scheme Selector */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => handleSchemeChange('center')}
                className={`px-3 py-2 rounded-xl text-xs font-semibold border transition-all text-left ${
                  config.dataset.partitionScheme === 'center'
                    ? 'bg-indigo-600 text-white border-indigo-400 shadow'
                    : 'bg-slate-800/80 text-slate-300 border-slate-700 hover:bg-slate-800'
                }`}
              >
                <div>Natural Centers</div>
                <div className="text-[10px] font-normal opacity-80">Hospital C1-C6</div>
              </button>

              <button
                onClick={() => handleSchemeChange('dirichlet')}
                className={`px-3 py-2 rounded-xl text-xs font-semibold border transition-all text-left ${
                  config.dataset.partitionScheme === 'dirichlet'
                    ? 'bg-indigo-600 text-white border-indigo-400 shadow'
                    : 'bg-slate-800/80 text-slate-300 border-slate-700 hover:bg-slate-800'
                }`}
              >
                <div>Dirichlet(α)</div>
                <div className="text-[10px] font-normal opacity-80">Continuous Skew</div>
              </button>

              <button
                onClick={() => handleSchemeChange('iid')}
                className={`px-3 py-2 rounded-xl text-xs font-semibold border transition-all text-left ${
                  config.dataset.partitionScheme === 'iid'
                    ? 'bg-indigo-600 text-white border-indigo-400 shadow'
                    : 'bg-slate-800/80 text-slate-300 border-slate-700 hover:bg-slate-800'
                }`}
              >
                <div>IID Uniform</div>
                <div className="text-[10px] font-normal opacity-80">Equal Ratio</div>
              </button>

              <button
                onClick={() => handleSchemeChange('shards')}
                className={`px-3 py-2 rounded-xl text-xs font-semibold border transition-all text-left ${
                  config.dataset.partitionScheme === 'shards'
                    ? 'bg-indigo-600 text-white border-indigo-400 shadow'
                    : 'bg-slate-800/80 text-slate-300 border-slate-700 hover:bg-slate-800'
                }`}
              >
                <div>Shards (2)</div>
                <div className="text-[10px] font-normal opacity-80">Extreme Skew</div>
              </button>
            </div>

            {/* Dirichlet Alpha Slider */}
            {config.dataset.partitionScheme === 'dirichlet' && (
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-200">Dirichlet Concentration (α):</span>
                  <span className="font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded">
                    α = {config.dataset.alpha.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="5.0"
                  step="0.05"
                  value={config.dataset.alpha}
                  onChange={(e) => handleAlphaChange(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>Extreme Non-IID (α=0.05)</span>
                  <span>Balanced IID (α=5.0)</span>
                </div>
              </div>
            )}

            {/* System Heterogeneity Options */}
            <div className="border-t border-slate-800 pt-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-bold text-white flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                    <span>System Heterogeneity</span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Computational multipliers [1x, 2x, 4x]
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.common.systemHeterogeneity}
                    onChange={(e) => handleSystemHeterogeneityToggle(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                </label>
              </div>

              {/* Straggler Ratio */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">Straggler Ratio:</span>
                <select
                  value={config.common.stragglerRatio}
                  onChange={(e) => handleStragglerChange(parseFloat(e.target.value))}
                  className="bg-slate-800 text-slate-200 text-xs rounded-lg px-2 py-1 border border-slate-700"
                >
                  <option value={0.0}>0% (None)</option>
                  <option value={0.2}>20% (1 Client)</option>
                  <option value={0.4}>40% (2 Clients)</option>
                </select>
              </div>
            </div>

            {/* Theoretical insight note */}
            <div className="p-3 rounded-xl bg-indigo-950/20 border border-indigo-900/40 text-xs text-indigo-300/90 leading-relaxed">
              <strong>Heterogeneity Impact:</strong> In traditional <em>FedAvg</em>, stragglers or clients with higher local epochs (τ) cause objective drift towards their local minima. <em>FedNova</em> normalizes local updates by τ to restore the unbiased global gradient.
            </div>
          </div>
        </div>

        {/* Client Distribution Chart Column */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-indigo-400" />
                  <span>Client Sample & Label Skew Distribution</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Visualizing class balance (Polyp Positive vs Clear) and computational capacity across clients
                </p>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-rose-500" />
                  <span className="text-slate-300">Polyp Positive</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-slate-700" />
                  <span className="text-slate-300">Clear Tissue</span>
                </span>
              </div>
            </div>

            {/* Distribution Bars */}
            <div className="space-y-3 pt-2">
              {partition.clientDistributions.map((client) => {
                const posPercent = Math.round(client.positiveRatio * 100);
                const negPercent = 100 - posPercent;
                return (
                  <div key={client.clientId} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-200">{client.name}</span>
                        {client.isStraggler && (
                          <span className="text-[10px] font-bold text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/30 flex items-center gap-1">
                            <AlertTriangle className="w-2.5 h-2.5" />
                            Straggler
                          </span>
                        )}
                        <span className="text-[10px] text-slate-400">
                          ({client.effectiveEpochs} local epochs / {client.systemMultiplier}x speed)
                        </span>
                      </div>
                      <div className="text-slate-300 font-mono text-[11px]">
                        {client.positiveCount} pos / {client.negativeCount} neg ({client.totalCount} total)
                      </div>
                    </div>

                    {/* Stacked bar */}
                    <div className="h-6 w-full bg-slate-950 rounded-lg overflow-hidden flex border border-slate-800">
                      <div
                        style={{ width: `${posPercent}%` }}
                        className="bg-rose-500 h-full flex items-center justify-start px-2 text-[10px] font-bold text-white transition-all duration-300 truncate"
                        title={`${client.positiveCount} Polyps (${posPercent}%)`}
                      >
                        {posPercent > 15 ? `${posPercent}% Polyp` : ''}
                      </div>
                      <div
                        style={{ width: `${negPercent}%` }}
                        className="bg-slate-700 h-full flex items-center justify-end px-2 text-[10px] font-medium text-slate-300 transition-all duration-300 truncate"
                        title={`${client.negativeCount} Clear (${negPercent}%)`}
                      >
                        {negPercent > 15 ? `${negPercent}% Clear` : ''}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Reference Distribution Gallery (From Repository Assets) */}
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <FileImage className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-bold text-white">
                  FL-bench Partition Schemes Reference Library
                </h3>
              </div>
              
              {/* Tab Selector */}
              <div className="flex flex-wrap gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
                {Object.entries(distributionImages).map(([key, data]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedImageTab(key as any)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                      selectedImageTab === key
                        ? 'bg-indigo-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {data.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Display selected image */}
            <div className="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 p-3 flex flex-col items-center">
              <img
                src={distributionImages[selectedImageTab].src}
                alt={distributionImages[selectedImageTab].label}
                className="max-h-56 object-contain rounded-lg shadow"
                onError={(e) => {
                  // Fallback in case of path mismatch
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
              <p className="text-xs text-slate-400 mt-2 text-center max-w-xl">
                {distributionImages[selectedImageTab].desc}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
