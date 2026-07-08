import React, { useState } from 'react';
import { ApiClient, MatchingResponse } from '../services/ApiClient';
import { DragDropUpload, LoadingState, ErrorState } from '../components/Common';
import { Layers, Image as ImageIcon } from 'lucide-react';

const FeatureMatching: React.FC = () => {
  const [result, setResult] = useState<MatchingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [img1, setImg1] = useState<File | null>(null);
  const [img2, setImg2] = useState<File | null>(null);

  const handleMatch = () => {
    if (!img1 || !img2) return;
    setLoading(true);
    setError(null);
    setResult(null);

    ApiClient.matchFeatures(img1, img2)
      .then(res => {
        setResult(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed SIFT matching.');
        setLoading(false);
      });
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
          <Layers className="w-8 h-8 text-brand-500" /> SIFT Descriptor Matching
        </h1>
        <p className="text-gray-400 text-sm mt-1">SIFT feature description with Lowe's ratio test sweep.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Column */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6 space-y-6">
            <h3 className="text-lg font-semibold text-white">Source Images</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-400 font-semibold mb-2 block">Image A</label>
                <DragDropUpload
                  onFilesSelected={files => setImg1(files[0] || null)}
                  label="Select Image A"
                />
              </div>

              <div>
                <label className="text-xs text-gray-400 font-semibold mb-2 block">Image B</label>
                <DragDropUpload
                  onFilesSelected={files => setImg2(files[0] || null)}
                  label="Select Image B"
                />
              </div>
            </div>

            <button
              onClick={handleMatch}
              disabled={!img1 || !img2 || loading}
              className="btn-primary w-full"
            >
              Run SIFT Matching
            </button>
          </div>

          {result && (
            <div className="glass-card p-6 space-y-4">
              <h3 className="text-base font-semibold text-white font-sans">Ratio Sweep Summary</h3>
              <div className="overflow-hidden rounded-xl border border-gray-800 text-xs">
                <table className="w-full text-left text-gray-300">
                  <thead className="bg-dark-800 text-gray-400 uppercase text-[9px] tracking-wider border-b border-gray-800">
                    <tr>
                      <th className="p-3">Ratio</th>
                      <th className="p-3">Matches</th>
                      <th className="p-3">Inliers</th>
                      <th className="p-3 text-right">Inlier Ratio</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800/40">
                    {Object.entries(result.ratios_sweep).map(([ratio, stats]) => (
                      <tr key={ratio} className={`hover:bg-dark-800/30 transition-colors ${ratio === '0.75' ? 'bg-brand-500/5 text-white font-bold' : ''}`}>
                        <td className="p-3">{ratio}</td>
                        <td className="p-3">{stats.match_count}</td>
                        <td className="p-3">{stats.ransac_inliers}</td>
                        <td className="p-3 text-right">{(stats.inlier_ratio * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Results Column */}
        <div className="lg:col-span-2 space-y-6">
          {loading && <div className="glass-card p-12"><LoadingState message="Extracting descriptors and fitting homography..." /></div>}
          {error && <ErrorState message={error} />}

          {!loading && !error && !result && (
            <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
              <ImageIcon className="w-12 h-12 text-gray-600 mb-3" />
              <h3 className="text-base font-semibold text-gray-300">No Match Resolved</h3>
              <p className="text-xs text-gray-500 mt-1 max-w-sm">Upload dual overlapping images on the left pane, then trigger matching solver.</p>
            </div>
          )}

          {result && (
            <div className="glass-card p-6 space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-base font-semibold text-white">Recommended Match Visualization</h3>
                <span className="bg-brand-500/10 text-brand-400 text-xs px-2.5 py-1 rounded-full font-semibold border border-brand-500/20">
                  Ratio: 0.75 | Inliers: {(result.inlier_ratio_at_0_75 * 100).toFixed(1)}%
                </span>
              </div>
              <div className="relative rounded-xl overflow-hidden border border-gray-800">
                <img
                  src={`data:image/png;base64,${result.visualization_base64}`}
                  alt="SIFT Matches"
                  className="w-full h-auto object-contain max-h-[500px]"
                />
              </div>
              <div className="flex justify-center gap-6 text-xs font-semibold">
                <span className="flex items-center gap-1 text-green-400">
                  <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" /> RANSAC Inlier Matches
                </span>
                <span className="flex items-center gap-1 text-red-400">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" /> Rejected Matches (Outliers)
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FeatureMatching;
