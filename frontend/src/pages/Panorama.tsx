import React, { useState, useEffect, useRef } from 'react';
import { ApiClient, PanoramaPollResponse } from '../services/ApiClient';
import { DragDropUpload, LoadingState, ErrorState } from '../components/Common';
import { Columns, Play, Image as ImageIcon } from 'lucide-react';

const Panorama: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [anchor, setAnchor] = useState<string>('middle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [pollResult, setPollResult] = useState<PanoramaPollResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const handleFilesSelected = (selectedFiles: File[]) => {
    setFiles(selectedFiles);
    setJobId(null);
    setJobStatus(null);
    setPollResult(null);
    setError(null);
  };

  const startStitching = () => {
    if (files.length !== 2 && files.length !== 3) {
      setError("Stitching requires exactly 2 or 3 overlapping images.");
      return;
    }

    setLoading(true);
    setError(null);
    setResultNull();

    ApiClient.submitPanoramaJob(files, anchor)
      .then(res => {
        setJobId(res.job_id);
        setJobStatus(res.status);
      })
      .catch(err => {
        setError(err.message || 'Stitching submission failed.');
        setLoading(false);
      });
  };

  const setResultNull = () => {
    setPollResult(null);
  };

  // Polling loop logic
  useEffect(() => {
    if (!jobId) return;

    const poll = async () => {
      try {
        const res = await ApiClient.pollPanoramaJob(jobId);
        setJobStatus(res.status);
        
        if (res.status === 'completed') {
          setPollResult(res);
          setLoading(false);
          setJobId(null); // Stop polling
        } else if (res.status === 'failed') {
          setError(res.error || 'Background stitching job execution failed.');
          setLoading(false);
          setJobId(null); // Stop polling
        }
      } catch (err: any) {
        setError(err.message || 'Polling background job status failed.');
        setLoading(false);
        setJobId(null);
      }
    };

    pollTimerRef.current = window.setInterval(poll, 1500);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, [jobId]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
          <Columns className="w-8 h-8 text-brand-500" /> Panorama Stitching
        </h1>
        <p className="text-gray-400 text-sm mt-1">Multi-frame planar warp stitching with auto-crop boundaries.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Column */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6 space-y-6">
            <h3 className="text-lg font-semibold text-white">Stitch Canvas Setup</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-400 font-semibold mb-2 block">Upload Overlapping Frames</label>
                <DragDropUpload
                  onFilesSelected={handleFilesSelected}
                  multiple={true}
                  maxFiles={3}
                  label="Select 2 or 3 Images"
                />
              </div>

              <div>
                <label className="text-xs text-gray-400 font-semibold mb-2 block">Select Anchor Frame</label>
                <select
                  value={anchor}
                  onChange={e => setAnchor(e.target.value)}
                  className="w-full bg-dark-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-brand-500"
                >
                  <option value="left">Left Frame</option>
                  <option value="middle">Middle Frame (Recommended for 3 images)</option>
                  <option value="right">Right Frame (Recommended for 2 images)</option>
                </select>
              </div>
            </div>

            <button
              onClick={startStitching}
              disabled={files.length < 2 || loading}
              className="btn-primary w-full"
            >
              <Play className="w-4 h-4 fill-current" /> Execute Stitching
            </button>
          </div>

          {pollResult && pollResult.pair_metrics && (
            <div className="glass-card p-6 space-y-4">
              <h3 className="text-base font-semibold text-white">Warp Pair Inliers</h3>
              <div className="space-y-3">
                {Object.entries(pollResult.pair_metrics).map(([pair, stats]) => (
                  <div key={pair} className="bg-dark-800/40 p-3.5 rounded-xl border border-gray-800/60 space-y-1.5">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-semibold uppercase text-brand-400">{pair.replace(/_/g, ' ')}</span>
                      <span className="text-xs font-bold text-white">{(stats.inlier_ratio * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-[10px] text-gray-400">
                      <span>Matches: {stats.total_matches}</span>
                      <span>Inliers: {stats.inliers}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Column */}
        <div className="lg:col-span-2 space-y-6">
          {loading && (
            <div className="glass-card p-12">
              <LoadingState
                message={jobStatus === 'pending' ? 'Queuing background job...' : 'Stitching canvas and auto-cropping...'}
                submessage={`Job ID: ${jobId || 'Allocating...'}`}
              />
            </div>
          )}

          {error && <ErrorState message={error} />}

          {!loading && !error && !pollResult && (
            <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
              <ImageIcon className="w-12 h-12 text-gray-600 mb-3" />
              <h3 className="text-base font-semibold text-gray-300">No Stitched Output</h3>
              <p className="text-xs text-gray-500 mt-1 max-w-sm">Provide 2 or 3 side-by-side frames, select anchor, then run stitching.</p>
            </div>
          )}

          {pollResult && pollResult.panorama_base64 && (
            <div className="glass-card p-6 space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-base font-semibold text-white">Stitched Panorama Canvas</h3>
                {pollResult.average_inlier_ratio && (
                  <span className="bg-brand-500/10 text-brand-400 text-xs px-2.5 py-1 rounded-full font-semibold border border-brand-500/20">
                    Average Inliers: {(pollResult.average_inlier_ratio * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-dark-950 p-2">
                <img
                  src={`data:image/png;base64,${pollResult.panorama_base64}`}
                  alt="Stitched Panorama"
                  className="w-full h-auto object-contain max-h-[500px]"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Panorama;
