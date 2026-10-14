import React, { useState } from 'react';
import { FLConfig } from '../types';
import { generateHydraConfigYaml } from '../utils/flSimulation';
import { 
  X, 
  Copy, 
  Check, 
  Download, 
  Terminal, 
  FileCode, 
  Sliders, 
  ShieldCheck 
} from 'lucide-react';

interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: FLConfig;
  onChangeConfig: (newConfig: FLConfig) => void;
}

export const ConfigModal: React.FC<ConfigModalProps> = ({
  isOpen,
  onClose,
  config,
  onChangeConfig,
}) => {
  const [copied, setCopied] = useState<boolean>(false);
  const yamlContent = generateHydraConfigYaml(config);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(yamlContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([yamlContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `defaults_${config.method}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <FileCode className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Hydra Configuration & CLI Execution</h3>
              <p className="text-xs text-slate-400">
                Matches <code>config/defaults.yaml</code> and Hydra CLI runtime flags
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

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1">
          {/* CLI Command Generator Box */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                <span>Python CLI Command to Reproduce in FL-bench:</span>
              </span>
            </div>
            <div className="p-2.5 bg-black/60 rounded-lg border border-slate-800 font-mono text-xs text-emerald-400 select-all overflow-x-auto">
              python main.py method={config.method} dataset.name={config.dataset.name} common.global_epoch={config.common.globalEpoch} common.local_epoch={config.common.localEpoch} optimizer.lr={config.optimizer.lr}
            </div>
          </div>

          {/* Quick Hyperparameter Tuning */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-white uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-indigo-400" />
              <span>Experiment Hyperparameters</span>
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              {/* Global Rounds */}
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <label className="text-slate-400 block">Global Rounds (Epochs):</label>
                <input
                  type="number"
                  min="5"
                  max="200"
                  value={config.common.globalEpoch}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      common: { ...config.common, globalEpoch: parseInt(e.target.value) || 50 },
                    })
                  }
                  className="w-full bg-slate-800 px-2.5 py-1.5 rounded-lg border border-slate-700 text-white font-mono"
                />
              </div>

              {/* Local Epochs */}
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <label className="text-slate-400 block">Local Epochs (τ):</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={config.common.localEpoch}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      common: { ...config.common, localEpoch: parseInt(e.target.value) || 5 },
                    })
                  }
                  className="w-full bg-slate-800 px-2.5 py-1.5 rounded-lg border border-slate-700 text-white font-mono"
                />
              </div>

              {/* Learning Rate */}
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <label className="text-slate-400 block">Learning Rate (lr):</label>
                <input
                  type="number"
                  step="0.001"
                  value={config.optimizer.lr}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      optimizer: { ...config.optimizer, lr: parseFloat(e.target.value) || 0.01 },
                    })
                  }
                  className="w-full bg-slate-800 px-2.5 py-1.5 rounded-lg border border-slate-700 text-white font-mono"
                />
              </div>

              {/* Join Ratio */}
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <label className="text-slate-400 block">Client Join Ratio:</label>
                <select
                  value={config.common.joinRatio}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      common: { ...config.common, joinRatio: parseFloat(e.target.value) },
                    })
                  }
                  className="w-full bg-slate-800 px-2.5 py-1.5 rounded-lg border border-slate-700 text-white font-mono"
                >
                  <option value={1.0}>1.0 (All 5 Centers)</option>
                  <option value={0.8}>0.8 (4 Centers)</option>
                  <option value={0.6}>0.6 (3 Centers)</option>
                  <option value={0.4}>0.4 (2 Centers)</option>
                </select>
              </div>
            </div>
          </div>

          {/* YAML Viewer */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white">Full Hydra defaults.yaml Preview:</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-all"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy YAML'}</span>
                </button>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow transition-all"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .yaml</span>
                </button>
              </div>
            </div>

            <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-indigo-200 overflow-x-auto max-h-64 selection:bg-indigo-700">
              {yamlContent}
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-end gap-3">
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
