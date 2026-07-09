import React, { useEffect, useState } from 'react';
import { ApiClient, DashboardResponse } from '../services/ApiClient';
import { LoadingState, ErrorState } from '../components/Common';
import { LayoutDashboard, Award, ArrowUpDown, ExternalLink, HelpCircle, Activity } from 'lucide-react';

const Overview: React.FC = () => {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<string>('accuracy');
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    fetchDashboard();
  }, []);

  const fetchDashboard = () => {
    setLoading(true);
    setError(null);
    ApiClient.getDashboard()
      .then(res => {
        setData(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load dashboard.');
        setLoading(false);
      });
  };

  if (loading) return <LoadingState message="Retrieving dashboard statistics..." />;
  if (error) return <ErrorState message={error} onRetry={fetchDashboard} />;
  if (!data) return null;

  const { baseline_metrics, backbone_comparison } = data;

  // Compile comparison rows dynamically based on what backbones are returned by the backend
  const rows = Object.keys(backbone_comparison)
    .filter(key => backbone_comparison[key as keyof typeof backbone_comparison] !== undefined)
    .map(key => {
      const data = backbone_comparison[key as keyof typeof backbone_comparison];
      return {
        name: key === 'resnet18' ? 'resnet18 (Active)' : key,
        accuracy: data?.accuracy || 0,
        f1_score: data?.f1_score || 0,
        params_m: data?.params_m || 0,
        size_mb: data?.size_mb || 0,
        speed_fps: data?.speed_fps || 0,
        train_time_sec: data?.train_time_sec || 0
      };
    });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedRows = [...rows].sort((a: any, b: any) => {
    let valA = a[sortField];
    let valB = b[sortField];
    if (typeof valA === 'string') {
      return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return sortAsc ? valA - valB : valB - valA;
  });

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <LayoutDashboard className="w-8 h-8 text-brand-500" /> Pipeline Overview
          </h1>
          <p className="text-gray-400 text-sm mt-1">Live metrics and benchmark parameters for the PerCV pipeline.</p>
        </div>
        <div className="flex gap-3">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="btn-secondary text-xs px-3 py-2 flex items-center gap-1.5"
          >
            <ExternalLink className="w-4 h-4" /> Notebook Reference
          </a>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Task 1: Lanes */}
        <div className="glass-card glass-card-hover p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">Lane Detection</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {(baseline_metrics.task1.mean_quality_score * 100).toFixed(1)}%
              </h3>
            </div>
            <div className="p-2.5 bg-brand-500/10 rounded-xl text-brand-500">
              <Award className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-xs text-gray-400">
            <span>Score: {baseline_metrics.task1.mean_quality_score.toFixed(4)}</span>
            <span className="mx-2">•</span>
            <span>{baseline_metrics.task1.processed_frames} frames evaluated</span>
          </div>
        </div>

        {/* Task 2: Matching */}
        <div className="glass-card glass-card-hover p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">SIFT Matching</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {baseline_metrics.task2.matches_at_0_75}
              </h3>
            </div>
            <div className="p-2.5 bg-brand-500/10 rounded-xl text-brand-500">
              <Activity className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-xs text-gray-400">
            <span>Inlier ratio: {(baseline_metrics.task2.inlier_ratio_at_0_75 * 100).toFixed(1)}%</span>
            <span className="mx-2">•</span>
            <span>Lowe ratio = 0.75</span>
          </div>
        </div>

        {/* Task 3: Panorama */}
        <div className="glass-card glass-card-hover p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">Stitch Average</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {(baseline_metrics.task3.average_inlier_ratio * 100).toFixed(1)}%
              </h3>
            </div>
            <div className="p-2.5 bg-brand-500/10 rounded-xl text-brand-500">
              <ExternalLink className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-xs text-gray-400">
            <span>L: {(baseline_metrics.task3.left_to_middle.inlier_ratio * 100).toFixed(1)}%</span>
            <span className="mx-2">•</span>
            <span>R: {(baseline_metrics.task3.right_to_middle.inlier_ratio * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Task 4: Classifier */}
        <div className="glass-card glass-card-hover p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">CNN Accuracy</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {(baseline_metrics.task4.test_accuracy * 100).toFixed(1)}%
              </h3>
            </div>
            <div className="p-2.5 bg-brand-500/10 rounded-xl text-brand-500">
              <Award className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-xs text-gray-400">
            <span>F1: {baseline_metrics.task4.macro_f1.toFixed(4)}</span>
            <span className="mx-2">•</span>
            <span>{baseline_metrics.task4.inference_fps} FPS</span>
          </div>
        </div>
      </div>

      {/* Backbone Comparison Dashboard Table */}
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold text-white mb-4">Neural Backbone Comparison</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-300">
            <thead className="bg-dark-800 text-gray-400 text-xs uppercase tracking-wider border-b border-gray-700">
              <tr>
                <th className="p-4 font-semibold text-gray-200">Backbone Model</th>
                <th className="p-4 font-semibold text-gray-200 cursor-pointer hover:text-white" onClick={() => handleSort('accuracy')}>
                  Accuracy <ArrowUpDown className="inline w-3 h-3 ml-1" />
                </th>
                <th className="p-4 font-semibold text-gray-200 cursor-pointer hover:text-white" onClick={() => handleSort('f1_score')}>
                  F1-Score <ArrowUpDown className="inline w-3 h-3 ml-1" />
                </th>
                <th className="p-4 font-semibold text-gray-200 cursor-pointer hover:text-white" onClick={() => handleSort('params_m')}>
                  Params (M) <ArrowUpDown className="inline w-3 h-3 ml-1" />
                </th>
                <th className="p-4 font-semibold text-gray-200 cursor-pointer hover:text-white" onClick={() => handleSort('size_mb')}>
                  Size (MB) <ArrowUpDown className="inline w-3 h-3 ml-1" />
                </th>
                <th className="p-4 font-semibold text-gray-200 cursor-pointer hover:text-white" onClick={() => handleSort('speed_fps')}>
                  Speed (FPS) <ArrowUpDown className="inline w-3 h-3 ml-1" />
                </th>
                <th className="p-4 font-semibold text-gray-200 cursor-pointer hover:text-white" onClick={() => handleSort('train_time_sec')}>
                  Train Time (s) <ArrowUpDown className="inline w-3 h-3 ml-1" />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60">
              {sortedRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-dark-800/40 transition-colors">
                  <td className="p-4 font-semibold text-white">{row.name}</td>
                  <td className="p-4">{(row.accuracy * 100).toFixed(2)}%</td>
                  <td className="p-4">{(row.f1_score * 100).toFixed(2)}%</td>
                  <td className="p-4">{row.params_m.toFixed(2)}</td>
                  <td className="p-4">{row.size_mb.toFixed(1)}</td>
                  <td className="p-4 text-brand-400 font-semibold">{row.speed_fps.toFixed(1)}</td>
                  <td className="p-4 text-gray-400">{row.train_time_sec.toFixed(1)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 text-xs text-gray-400 italic">
          * Measured on Intel test split, Kaggle T4 GPU, backbone frozen / linear-probe only — see notebooks/percv_kaggle.ipynb cells 14–18.
        </div>
      </div>

      {/* Critical Info Banner */}
      <div className="glass-card p-6 border-brand-500/10 flex items-start gap-4">
        <HelpCircle className="w-8 h-8 text-brand-500 flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-semibold text-white">Task Analysis & Failure Mode Reflections</h4>
          <p className="text-xs text-gray-400 mt-1 leading-relaxed">
            Lane detection operates at a baseline quality score of ~18.1% due to dense urban background clutter (signs, shadows, poles) producing false-positive line slope directions. Conversely, SIFT matches demonstrate high precision-recall alignment under a Lowe's ratio of 0.75, which acts as the optimal point on the ROC curve.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Overview;
