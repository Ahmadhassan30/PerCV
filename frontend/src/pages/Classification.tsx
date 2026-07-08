import React, { useState } from 'react';
import { ApiClient, ClassifyResponse } from '../services/ApiClient';
import { DragDropUpload, LoadingState, ErrorState } from '../components/Common';
import { Cpu, Image as ImageIcon } from 'lucide-react';

const Classification: React.FC = () => {
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backbone, setBackbone] = useState<string>('resnet18');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);

  const handleFileSelected = (files: File[]) => {
    const file = files[0] || null;
    setSelectedFile(file);
    setResult(null);
    setError(null);
    if (file) {
      setPreviewSrc(URL.createObjectURL(file));
    } else {
      setPreviewSrc(null);
    }
  };

  const handlePredict = () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    ApiClient.classifyImage(selectedFile, backbone)
      .then(res => {
        setResult(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed model classification.');
        setLoading(false);
      });
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-2">
          <Cpu className="w-8 h-8 text-brand-500" /> CNN Classification & Grad-CAM
        </h1>
        <p className="text-gray-400 text-sm mt-1">Deep learning CNN transfer learning classification with spatial Grad-CAM overlays.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Column */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6 space-y-6">
            <h3 className="text-lg font-semibold text-white">Inference Setup</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-400 font-semibold mb-2 block">Upload Scene Image</label>
                <DragDropUpload
                  onFilesSelected={handleFileSelected}
                  label="Select Image"
                />
              </div>

              <div>
                <label className="text-xs text-gray-400 font-semibold mb-2 block">Select CNN Backbone</label>
                <select
                  value={backbone}
                  onChange={e => setBackbone(e.target.value)}
                  className="w-full bg-dark-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-brand-500"
                >
                  <option value="resnet18">ResNet18 (Active — 92.77% Acc)</option>
                  <option value="mobilenetv2">MobileNetV2 (Baseline — 92.80% Acc)</option>
                </select>
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={!selectedFile || loading}
              className="btn-primary w-full"
            >
              Run CNN Prediction
            </button>
          </div>

          {result && (
            <div className="glass-card p-6 space-y-4">
              <h3 className="text-base font-semibold text-white font-sans">Class Probabilities</h3>
              <div className="space-y-4">
                {Object.entries(result.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className={`capitalize ${cls === result.predicted_class ? 'text-brand-400' : 'text-gray-300'}`}>
                        {cls}
                      </span>
                      <span className="text-white">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-dark-850 h-2 rounded-full overflow-hidden border border-gray-800">
                      <div
                        className={`h-full transition-all duration-500 ${cls === result.predicted_class ? 'bg-brand-500' : 'bg-gray-700'}`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Column */}
        <div className="lg:col-span-2 space-y-6">
          {loading && <div className="glass-card p-12"><LoadingState message="Running CNN forward feed & backward hook passes..." /></div>}
          {error && <ErrorState message={error} />}

          {!loading && !error && !result && (
            <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
              <ImageIcon className="w-12 h-12 text-gray-600 mb-3" />
              <h3 className="text-base font-semibold text-gray-300">No Inference Executed</h3>
              <p className="text-xs text-gray-500 mt-1 max-w-sm">Provide a scene image (buildings, forest, mountain, street), select backbone, and predict.</p>
            </div>
          )}

          {result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Original Preview */}
              {previewSrc && (
                <div className="glass-card p-6 space-y-4">
                  <h3 className="text-base font-semibold text-white">Original Input Backdrop</h3>
                  <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-dark-950">
                    <img
                      src={previewSrc}
                      alt="Original Input"
                      className="w-full h-auto object-contain max-h-[400px]"
                    />
                  </div>
                </div>
              )}

              {/* Grad-CAM Heatmap */}
              <div className="glass-card p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-base font-semibold text-white">Grad-CAM Overlay Map</h3>
                  <span className="bg-brand-500/10 text-brand-400 text-xs px-2.5 py-1 rounded-full font-semibold border border-brand-500/20">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-dark-950">
                  <img
                    src={`data:image/png;base64,${result.gradcam_base64}`}
                    alt="Grad-CAM Overlay"
                    className="w-full h-auto object-contain max-h-[400px]"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Classification;
