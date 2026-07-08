import React, { useState } from 'react';
import { ApiClient, LaneDetectionResponse } from '../services/ApiClient';
import { DragDropUpload, LoadingState, ErrorState } from '../components/Common';
import { Route, Image as ImageIcon } from 'lucide-react';

const LaneDetection: React.FC = () => {
  const [result, setResult] = useState<LaneDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFilesSelected = (files: File[]) => {
    setLoading(true);
    setError(null);
    setResult(null);

    ApiClient.detectLanes(files)
      .then(res => {
        setResult(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to run lane detection.');
        setLoading(false);
      });
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
          <Route className="w-8 h-8 text-brand-500" /> Lane Detection
        </h1>
        <p className="text-gray-400 text-sm mt-1">Canny Edge Profiler + Probabilistic Hough Lane Tracing.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Column */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6 space-y-4">
            <h3 className="text-lg font-semibold text-white">Source Road Images</h3>
            <p className="text-xs text-gray-400">Upload one or more dashboard camera frames. The system evaluates sensitive, balanced, and strict Canny presets dynamically.</p>
            <DragDropUpload
              onFilesSelected={handleFilesSelected}
              multiple={true}
              maxFiles={5}
              label="Select Road Frame"
            />
          </div>

          {result && (
            <div className="glass-card p-6 space-y-4">
              <h3 className="text-base font-semibold text-white">Preset Score Summary</h3>
              <div className="space-y-3">
                {Object.entries(result.preset_mean_scores).map(([preset, score]) => (
                  <div key={preset} className="flex justify-between items-center bg-dark-800/40 p-3 rounded-xl border border-gray-800/60">
                    <span className="text-sm font-medium capitalize text-gray-300">{preset} Preset</span>
                    <span className="text-sm font-bold text-white">{score.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Column */}
        <div className="lg:col-span-2 space-y-6">
          {loading && <div className="glass-card p-12"><LoadingState message="Detecting lines and filtering angles..." /></div>}
          {error && <ErrorState message={error} />}

          {!loading && !error && !result && (
            <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
              <ImageIcon className="w-12 h-12 text-gray-600 mb-3" />
              <h3 className="text-base font-semibold text-gray-300">No Image Selected</h3>
              <p className="text-xs text-gray-500 mt-1 max-w-sm">Upload road images on the left pane to analyze Hough lines and Lane Quality scores.</p>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              {/* Overlay Visualization */}
              <div className="glass-card p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-base font-semibold text-white">Line Angle Filter Overlay</h3>
                  <span className="bg-brand-500/10 text-brand-400 text-xs px-2.5 py-1 rounded-full font-semibold border border-brand-500/20">
                    Mean Quality Score: {result.overall_mean_quality_score.toFixed(4)}
                  </span>
                </div>
                <div className="relative rounded-xl overflow-hidden border border-gray-800">
                  <img
                    src={`data:image/png;base64,${result.overlay_base64}`}
                    alt="Lanes Overlay"
                    className="w-full h-auto object-contain max-h-[500px]"
                  />
                </div>
                <div className="flex justify-center gap-6 text-xs font-semibold">
                  <span className="flex items-center gap-1 text-green-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" /> Passed Angle [25°-75°] or [105°-155°]
                  </span>
                  <span className="flex items-center gap-1 text-red-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" /> Failed Angle (Out of bounds/noise)
                  </span>
                </div>
              </div>

              {/* Preset Breakdown Table */}
              <div className="glass-card p-6">
                <h3 className="text-base font-semibold text-white mb-4">Preset Detail Breakdown</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-gray-300">
                    <thead className="bg-dark-800 text-gray-400 text-[10px] uppercase tracking-wider border-b border-gray-800">
                      <tr>
                        <th className="p-3">Preset</th>
                        <th className="p-3">File Name</th>
                        <th className="p-3">Lines Count</th>
                        <th className="p-3 text-right">Quality Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/40">
                      {Object.entries(result.results_by_preset).flatMap(([preset, list]) =>
                        list.map((d, index) => (
                          <tr key={`${preset}-${index}`} className="hover:bg-dark-800/30 transition-colors">
                            <td className="p-3 font-semibold capitalize text-brand-400">{preset}</td>
                            <td className="p-3 text-gray-400">{d.image_name}</td>
                            <td className="p-3">{d.num_lines}</td>
                            <td className="p-3 text-right font-bold text-white">{d.quality_score.toFixed(4)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LaneDetection;
