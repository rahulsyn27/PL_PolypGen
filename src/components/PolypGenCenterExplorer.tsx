import React, { useState } from 'react';
import { POLYPGEN_CENTERS, SAMPLE_ENDOSCOPY_FRAMES } from '../data/polypgenData';
import { ClientParty, PolypSample } from '../types';
import { 
  Building2, 
  Eye, 
  Sparkles, 
  Sliders, 
  Activity, 
  ShieldAlert, 
  CheckCircle2, 
  Tag, 
  Layers, 
  Maximize2,
  Camera,
  Info
} from 'lucide-react';

export const PolypGenCenterExplorer: React.FC = () => {
  const [selectedCenterId, setSelectedCenterId] = useState<number>(1);
  const [showMask, setShowMask] = useState<boolean>(true);
  const [activeFrameIndex, setActiveFrameIndex] = useState<number>(0);

  const selectedCenter = POLYPGEN_CENTERS.find(c => c.id === selectedCenterId) || POLYPGEN_CENTERS[0];
  const centerFrames = SAMPLE_ENDOSCOPY_FRAMES.filter(f => f.centerId === selectedCenterId);
  const activeFrame = centerFrames[activeFrameIndex] || centerFrames[0] || SAMPLE_ENDOSCOPY_FRAMES[0];

  return (
    <div className="space-y-6">
      {/* Top Banner Overview */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <Building2 className="w-4 h-4" />
              </span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                PolypGen Multi-Center Clinical Endoscopy Network
              </h2>
            </div>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl">
              Real-world medical federated learning across 6 distinct hospital endoscopy suites. Centers 1–5 collaborate as federated training clients with heterogeneous optical sensors, while Center 6 serves as a strictly held-out out-of-domain (OOD) test facility.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="px-3 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700 text-xs">
              <div className="text-slate-400">Total PolypGen Data</div>
              <div className="font-semibold text-white">2,810 Frames (6 Centers)</div>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs">
              <div className="text-emerald-400">Holdout OOD Center</div>
              <div className="font-semibold text-emerald-300">Center 6 (EVIS X1)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Grid of 6 Centers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {POLYPGEN_CENTERS.map((center) => {
          const isSelected = center.id === selectedCenterId;
          const isOOD = center.role === 'ood_test';
          return (
            <button
              key={center.id}
              onClick={() => {
                setSelectedCenterId(center.id);
                setActiveFrameIndex(0);
              }}
              className={`text-left p-4 rounded-xl border transition-all duration-200 relative overflow-hidden group ${
                isSelected
                  ? 'bg-slate-800/90 border-indigo-500 shadow-lg shadow-indigo-500/10 ring-1 ring-indigo-500/50'
                  : 'bg-slate-900/60 border-slate-800/80 hover:bg-slate-800/50 hover:border-slate-700'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs ${
                    isOOD 
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' 
                      : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                  }`}>
                    C{center.id}
                  </span>
                  <div>
                    <h3 className="font-semibold text-sm text-white group-hover:text-indigo-300 transition-colors">
                      {center.name}
                    </h3>
                    <p className="text-[11px] text-slate-400 truncate max-w-[170px]">{center.hospitalName}</p>
                  </div>
                </div>
                {isOOD ? (
                  <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    OOD Test
                  </span>
                ) : (
                  <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                    FL Client
                  </span>
                )}
              </div>

              {/* Hardware Spec */}
              <div className="mt-3 p-2 rounded-lg bg-slate-950/50 border border-slate-800/60 text-xs">
                <div className="flex items-center gap-1.5 text-slate-300 font-medium truncate">
                  <Camera className="w-3 h-3 text-indigo-400 shrink-0" />
                  <span className="truncate">{center.endoscopeModel}</span>
                </div>
              </div>

              {/* Stats & Multipliers */}
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="bg-slate-950/40 p-1.5 rounded-lg border border-slate-800/50">
                  <div className="text-[10px] text-slate-400">Samples</div>
                  <div className="font-bold text-slate-200">{center.sampleCount}</div>
                </div>
                <div className="bg-slate-950/40 p-1.5 rounded-lg border border-slate-800/50">
                  <div className="text-[10px] text-slate-400">Polyp Rate</div>
                  <div className="font-bold text-indigo-300">
                    {Math.round((center.polypPositiveCount / center.sampleCount) * 100)}%
                  </div>
                </div>
                <div className="bg-slate-950/40 p-1.5 rounded-lg border border-slate-800/50">
                  <div className="text-[10px] text-slate-400">Compute Speed</div>
                  <div className={`font-bold ${center.isStraggler ? 'text-amber-400' : 'text-emerald-400'}`}>
                    {center.localEpochsMultiplier}x {center.isStraggler ? '(Straggler)' : ''}
                  </div>
                </div>
              </div>

              {/* Selection indicator bar */}
              {isSelected && (
                <div className="absolute bottom-0 inset-x-0 h-0.5 bg-indigo-500" />
              )}
            </button>
          );
        })}
      </div>

      {/* Detailed Center Profile & Colonoscopy Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900/80 p-6 rounded-2xl border border-slate-800">
        {/* Left Column: Center Details & Sensor Shift */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
                Hospital Profile & Optical Specs
              </span>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                {selectedCenter.name} - {selectedCenter.hospitalName}
              </h3>
            </div>
            {selectedCenter.role === 'ood_test' ? (
              <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-amber-500/20 text-amber-300 border border-amber-500/40">
                Held-out Generalization
              </span>
            ) : (
              <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                Active Federated Participant
              </span>
            )}
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3">
            <div>
              <div className="text-xs font-medium text-slate-400">Endoscopy System</div>
              <div className="text-sm font-semibold text-slate-200 mt-0.5">{selectedCenter.endoscopeModel}</div>
            </div>

            <div className="border-t border-slate-800/80 pt-3">
              <div className="text-xs font-medium text-slate-400 mb-1.5">Sensor Domain Shift Signature</div>
              <ul className="text-xs space-y-1.5 text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 shrink-0" />
                  <span><strong>Color Profile:</strong> {selectedCenter.sensorCharacteristics.colorTone}</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                  <span><strong>Illumination:</strong> {selectedCenter.sensorCharacteristics.lighting}</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <span><strong>Resolution & Texture:</strong> {selectedCenter.sensorCharacteristics.sharpness}</span>
                </li>
              </ul>
            </div>

            <div className="border-t border-slate-800/80 pt-3 flex items-center justify-between text-xs">
              <span className="text-slate-400">Heterogeneity Multiplier:</span>
              <span className="font-bold text-slate-200">
                {selectedCenter.localEpochsMultiplier}x local epochs (τ = {selectedCenter.localEpochsMultiplier * 5} steps/round)
              </span>
            </div>
          </div>

          {/* Educational Note: Why FedBN works here */}
          <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-900/50 text-xs text-emerald-200/90 space-y-1.5">
            <div className="flex items-center gap-1.5 font-semibold text-emerald-300">
              <Info className="w-4 h-4 text-emerald-400" />
              <span>Domain Adaptation Advantage</span>
            </div>
            <p>
              In <strong>FedBN</strong>, while convolutional feature extractors are shared globally across hospitals, Batch Normalization layers are kept private to {selectedCenter.name}. This normalizes {selectedCenter.name}'s unique sensor glare, dye tint, and optical magnification without destabilizing other centers!
            </p>
          </div>
        </div>

        {/* Right Column: Colonoscopy Frame & Mask Visualizer */}
        <div className="lg:col-span-7 flex flex-col space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-white">Clinical Frame Visualizer</span>
              <span className="text-xs text-slate-400">(224×224 Endoscopy Input)</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowMask(!showMask)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold border transition-all ${
                  showMask
                    ? 'bg-indigo-600/30 text-indigo-300 border-indigo-500/50'
                    : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Mask Overlay {showMask ? 'ON' : 'OFF'}</span>
              </button>
            </div>
          </div>

          {/* Frame Display Area */}
          <div className="relative aspect-video sm:aspect-[4/3] max-h-[340px] w-full rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center shadow-inner">
            {/* Synthetic realistic colonoscopy canvas simulation */}
            <div 
              className="w-full h-full relative flex items-center justify-center"
              style={{
                backgroundColor: activeFrame.imageColor,
                backgroundImage: `radial-gradient(circle at 50% 50%, ${activeFrame.imageColor}dd 20%, #150608 95%)`,
              }}
            >
              {/* Simulated mucosal vascular textures */}
              <div className="absolute inset-0 opacity-40 mix-blend-overlay bg-[radial-gradient(#ff8899_1px,transparent_1px)] [background-size:16px_16px]" />
              
              {/* Lumen deep dark center */}
              <div className="absolute w-36 h-36 rounded-full bg-black/60 blur-xl pointer-events-none" />

              {/* Specular glare typical in wet endoscopy */}
              <div className="absolute top-1/4 left-1/3 w-8 h-4 rounded-full bg-white/70 blur-[1px] rotate-12 pointer-events-none" />
              <div className="absolute bottom-1/3 right-1/4 w-5 h-3 rounded-full bg-white/60 blur-[1px] -rotate-12 pointer-events-none" />

              {/* Polyp Lesion Rendering */}
              {activeFrame.hasPolyp && activeFrame.maskBox && (
                <div
                  className="absolute rounded-[40%_60%_70%_30%/40%_50%_60%_50%] transition-all"
                  style={{
                    left: `${activeFrame.maskBox.x}%`,
                    top: `${activeFrame.maskBox.y}%`,
                    width: `${activeFrame.maskBox.width}%`,
                    height: `${activeFrame.maskBox.height}%`,
                    backgroundColor: '#b34757',
                    boxShadow: 'inset 2px 2px 8px rgba(255,255,255,0.4), 0 4px 15px rgba(0,0,0,0.6)',
                  }}
                >
                  {/* Subtle pit pattern */}
                  <div className="w-full h-full rounded-full opacity-30 bg-[radial-gradient(#4d111d_2px,transparent_2px)] [background-size:6px_6px]" />
                </div>
              )}

              {/* Ground Truth Segmentation Mask Overlay */}
              {showMask && activeFrame.hasPolyp && activeFrame.maskBox && (
                <div
                  className="absolute border-2 border-dashed border-emerald-400 bg-emerald-500/20 rounded-lg flex items-start justify-end p-1 transition-all"
                  style={{
                    left: `${Math.max(2, activeFrame.maskBox.x - 2)}%`,
                    top: `${Math.max(2, activeFrame.maskBox.y - 2)}%`,
                    width: `${activeFrame.maskBox.width + 4}%`,
                    height: `${activeFrame.maskBox.height + 4}%`,
                  }}
                >
                  <span className="text-[10px] font-bold bg-emerald-500 text-slate-950 px-1 py-0.2 rounded shadow">
                    Polyp ({activeFrame.sizeMm || '7.5'}mm)
                  </span>
                </div>
              )}

              {/* Watermark / Sensor Overlay */}
              <div className="absolute top-3 left-3 text-[10px] font-mono text-white/70 bg-black/60 px-2 py-0.5 rounded backdrop-blur-sm">
                SRC: {selectedCenter.name} | SENSOR: {selectedCenter.endoscopeModel.split(' ')[0]} | STATUS: {activeFrame.hasPolyp ? 'POLYP DETECTED' : 'CLEAR MUCOSA'}
              </div>
            </div>
          </div>

          {/* Frame Description & Thumbnails */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div>
              <div className="font-semibold text-white flex items-center gap-1.5">
                <Tag className="w-3.5 h-3.5 text-indigo-400" />
                <span>Classification: {activeFrame.hasPolyp ? `Positive (${activeFrame.polypType})` : 'Negative (No Dysplasia)'}</span>
              </div>
              <p className="text-slate-400 text-[11px] mt-0.5">{activeFrame.description}</p>
            </div>
            
            {/* Center sample selector thumbnails */}
            <div className="flex items-center gap-1.5 shrink-0">
              {centerFrames.map((frame, idx) => (
                <button
                  key={frame.id}
                  onClick={() => setActiveFrameIndex(idx)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${
                    idx === activeFrameIndex
                      ? 'bg-indigo-600 text-white border-indigo-400'
                      : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-white'
                  }`}
                >
                  Sample #{idx + 1}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
