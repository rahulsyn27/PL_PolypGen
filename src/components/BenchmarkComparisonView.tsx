import React, { useState } from 'react';
import { PRECOMPUTED_BENCHMARKS } from '../data/polypgenData';
import { BenchmarkResult, FLMethod } from '../types';
import { 
  Trophy, 
  Sparkles, 
  Download, 
  Check, 
  ArrowUpRight, 
  ShieldCheck, 
  Zap, 
  Eye, 
  Layers, 
  FileSpreadsheet,
  Award,
  HelpCircle
} from 'lucide-react';

interface BenchmarkComparisonViewProps {
  onSelectMethod: (method: FLMethod) => void;
}

export const BenchmarkComparisonView: React.FC<BenchmarkComparisonViewProps> = ({
  onSelectMethod,
}) => {
  const [selectedBenchmark, setSelectedBenchmark] = useState<BenchmarkResult>(PRECOMPUTED_BENCHMARKS[0]);
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);

  const handleExportCSV = () => {
    const headers = 'Method,Paper,Personalized_Acc,Global_Test_Acc,OOD_Center6_Acc,Convergence_Rounds,Comm_KB_Round,DP_Epsilon\n';
    const rows = PRECOMPUTED_BENCHMARKS.map(b => 
      `"${b.name}","${b.paperRef}",${b.personalizedAvgAccuracy},${b.globalTestAccuracy},${b.oodCenter6Accuracy},${b.convergenceRound},${b.commKbPerRound},${b.privacyBudgetEpsilon || 'None'}`
    ).join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'polypgen_federated_benchmark_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(PRECOMPUTED_BENCHMARKS, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'polypgen_federated_benchmark_results.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <Trophy className="w-4 h-4" />
              </span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Empirical Benchmark Matrix on PolypGen
              </h2>
            </div>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl">
              Comparative evaluation of Personalized Federated Learning methods against canonical baselines across 6 clinical endoscopy centers. Evaluated on personalized local accuracy, held-out zero-shot generalization (Center 6), and communication efficiency.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleExportCSV}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-all shadow-sm"
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
              <span>Export CSV</span>
            </button>
            <button
              onClick={handleExportJSON}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export JSON</span>
            </button>
          </div>
        </div>
      </div>

      {/* Leaderboard Table */}
      <div className="bg-slate-900/80 rounded-2xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Award className="w-4 h-4 text-amber-400" />
            <span>Algorithm Performance Leaderboard</span>
          </h3>
          <span className="text-xs text-slate-400">
            Click any row to inspect convergence and launch simulation
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Rank & Algorithm</th>
                <th className="py-3 px-4">Personalized Accuracy</th>
                <th className="py-3 px-4">OOD (Center 6) Acc</th>
                <th className="py-3 px-4">Convergence (Rounds)</th>
                <th className="py-3 px-4">Comm / Round</th>
                <th className="py-3 px-4">Privacy Budget</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {PRECOMPUTED_BENCHMARKS.map((benchmark, index) => {
                const isSelected = selectedBenchmark.method === benchmark.method;
                const isWinner = index === 0;
                return (
                  <tr
                    key={benchmark.method}
                    onClick={() => setSelectedBenchmark(benchmark)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-indigo-950/40 text-white'
                        : 'hover:bg-slate-800/40'
                    }`}
                  >
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2.5">
                        <span className={`w-5 h-5 rounded-full flex items-center justify-center font-bold text-[11px] ${
                          index === 0
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50'
                            : index === 1
                            ? 'bg-slate-400/20 text-slate-200 border border-slate-400/50'
                            : index === 2
                            ? 'bg-amber-700/20 text-amber-500 border border-amber-700/50'
                            : 'text-slate-500'
                        }`}>
                          {index + 1}
                        </span>
                        <div>
                          <div className="font-bold flex items-center gap-1.5">
                            <span>{benchmark.name}</span>
                            {isWinner && (
                              <span className="text-[9px] uppercase px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                                SOTA
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-slate-400">{benchmark.paperRef}</div>
                        </div>
                      </div>
                    </td>

                    <td className="py-3.5 px-4">
                      <div className="font-bold text-sm text-emerald-400">
                        {benchmark.personalizedAvgAccuracy.toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-slate-400">5-hospital avg</div>
                    </td>

                    <td className="py-3.5 px-4">
                      <div className="font-semibold text-amber-300">
                        {benchmark.oodCenter6Accuracy.toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-slate-400">Zero-shot shift</div>
                    </td>

                    <td className="py-3.5 px-4 font-mono">
                      {benchmark.convergenceRound} rounds
                    </td>

                    <td className="py-3.5 px-4 font-mono text-slate-300">
                      {benchmark.commKbPerRound > 0 ? `${benchmark.commKbPerRound} KB` : '0 KB (Local)'}
                    </td>

                    <td className="py-3.5 px-4">
                      {benchmark.privacyBudgetEpsilon ? (
                        <span className="font-mono text-emerald-300 bg-emerald-950/50 px-2 py-0.5 rounded border border-emerald-800/40">
                          ε = {benchmark.privacyBudgetEpsilon}
                        </span>
                      ) : (
                        <span className="text-slate-500">None (Full data)</span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectMethod(benchmark.method);
                        }}
                        className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/40 text-[11px] font-semibold transition-all inline-flex items-center gap-1"
                      >
                        <span>Run</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Benchmark Deep-Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/80 p-6 rounded-2xl border border-slate-800">
        <div className="lg:col-span-5 space-y-4">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
              Method Analysis & Layer Routing
            </span>
            <h3 className="text-lg font-bold text-white mt-0.5">
              {selectedBenchmark.name} ({selectedBenchmark.paperRef})
            </h3>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">
              {selectedBenchmark.description}
            </p>
          </div>

          <div className="space-y-2">
            <span className="text-xs font-semibold text-slate-400">Architectural Key Features:</span>
            <div className="grid grid-cols-1 gap-1.5">
              {selectedBenchmark.features.map((feat, i) => (
                <div key={i} className="flex items-center gap-2 text-xs text-slate-200 p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                  <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span>{feat}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Hospital Breakdown */}
          <div>
            <span className="text-xs font-semibold text-slate-400 mb-2 block">
              Per-Hospital Personalized Accuracy (C1–C5):
            </span>
            <div className="grid grid-cols-5 gap-1.5 text-center text-xs">
              {Object.entries(selectedBenchmark.clientAccuracyBreakdown).map(([cid, acc]) => (
                <div key={cid} className="p-2 rounded-lg bg-slate-950/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">C{cid}</div>
                  <div className="font-bold text-indigo-300 font-mono mt-0.5">{acc}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Key Takeaways & Clinical Impact */}
        <div className="lg:col-span-7 space-y-4">
          <h4 className="text-sm font-bold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span>Key Research Findings & Clinical Takeaways</span>
          </h4>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {/* Takeaway 1 */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
              <div className="font-bold text-emerald-400 flex items-center gap-1.5">
                <Award className="w-3.5 h-3.5" />
                <span>FedBN Wins on Sensor Heterogeneity</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                By decoupling batch normalization statistics from global aggregation, FedBN allows each hospital's local model to align its feature distributions without corrupting others. This yields +7.5% higher accuracy than standard FedAvg.
              </p>
            </div>

            {/* Takeaway 2 */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
              <div className="font-bold text-cyan-400 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                <span>FedRep Enables Representation Sharing</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                Hospitals collaborate strictly on learning general polyp morphology (shared backbone trunk), while keeping the decision boundary (classification head) tailored to local polyp prevalence.
              </p>
            </div>

            {/* Takeaway 3 */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
              <div className="font-bold text-indigo-400 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5" />
                <span>FedNova Mitigates Straggler Drift</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                When hospitals have varying compute hardware (e.g. Center 4 with 4x local epoch delay), FedNova normalizes updates by local steps τ, preventing the global model from over-biasing towards slower clients.
              </p>
            </div>

            {/* Takeaway 4 */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
              <div className="font-bold text-rose-400 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>The Privacy-Utility Tradeoff</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                DP-FedAvg guarantees patient differential privacy (ε=4.82) under Opacus accounting at a mild 3.4% utility drop, proving clinical feasibility for sensitive endoscopic data.
              </p>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-indigo-950/30 border border-indigo-900/50 flex items-center justify-between">
            <span className="text-xs text-indigo-300">
              Ready to reproduce this experiment in real-time?
            </span>
            <button
              onClick={() => onSelectMethod(selectedBenchmark.method)}
              className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-md transition-all"
            >
              Simulate {selectedBenchmark.name}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
