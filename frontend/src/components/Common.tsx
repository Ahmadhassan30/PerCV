import React, { useRef, useState } from 'react';
import { AlertCircle, Loader2, Upload, X } from 'lucide-react';

// Shared Loading State Component
export const LoadingState: React.FC<{
  message?: string;
  progress?: number;
  submessage?: string;
}> = ({ message = 'Processing pipeline...', progress, submessage }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center">
      <Loader2 className="w-12 h-12 text-brand-500 animate-spin mb-4" />
      <h3 className="text-lg font-semibold text-gray-100">{message}</h3>
      {submessage && <p className="text-sm text-gray-400 mt-1">{submessage}</p>}
      {progress !== undefined && (
        <div className="w-64 bg-dark-700 h-2 rounded-full overflow-hidden mt-4">
          <div
            className="bg-brand-500 h-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
};

// Shared Error State Component
export const ErrorState: React.FC<{
  message: string;
  onRetry?: () => void;
}> = ({ message, onRetry }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 glass-card border-red-500/20 text-center max-w-md mx-auto my-6">
      <AlertCircle className="w-10 h-10 text-red-500 mb-3" />
      <h4 className="text-base font-semibold text-gray-100 mb-2">Pipeline Execution Failed</h4>
      <p className="text-sm text-gray-400 mb-4">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="btn-secondary text-sm">
          Retry Operation
        </button>
      )}
    </div>
  );
};

// Reusable Drag & Drop Image Uploader Component
export const DragDropUpload: React.FC<{
  onFilesSelected: (files: File[]) => void;
  multiple?: boolean;
  maxFiles?: number;
  label?: string;
}> = ({ onFilesSelected, multiple = false, maxFiles = 1, label = "Upload Image" }) => {
  const [dragActive, setDragActive] = useState(false);
  const [previews, setPreviews] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const filesList = Array.from(e.dataTransfer.files);
      const imageFiles = filesList.filter(f => f.type.startsWith("image/"));
      if (imageFiles.length > 0) {
        processSelectedFiles(imageFiles.slice(0, maxFiles));
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const filesList = Array.from(e.target.files);
      processSelectedFiles(filesList.slice(0, maxFiles));
    }
  };

  const processSelectedFiles = (files: File[]) => {
    onFilesSelected(files);
    // Create local URLs for preview
    const filePreviews = files.map(file => URL.createObjectURL(file));
    setPreviews(filePreviews);
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  const clearPreviews = () => {
    previews.forEach(url => URL.revokeObjectURL(url));
    setPreviews([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        className={`w-full p-8 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-all duration-200 ${
          dragActive
            ? "border-brand-500 bg-brand-500/5"
            : "border-gray-700 bg-dark-900/30 hover:border-gray-600"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple={multiple}
          accept="image/png, image/jpeg, image/jpg"
          onChange={handleChange}
        />
        
        {previews.length > 0 ? (
          <div className="flex flex-col items-center w-full">
            <div className="flex flex-wrap justify-center gap-4 mb-4">
              {previews.map((src, i) => (
                <div key={i} className="relative group w-24 h-24 rounded-lg overflow-hidden border border-gray-800">
                  <img src={src} alt="preview" className="w-full h-full object-cover" />
                </div>
              ))}
            </div>
            <button
              onClick={clearPreviews}
              className="text-xs text-gray-400 hover:text-red-400 transition-colors flex items-center gap-1"
            >
              <X className="w-3.5 h-3.5" /> Remove Image
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center">
            <div className="p-3 bg-brand-500/10 rounded-full mb-3 text-brand-500">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-gray-200 mb-1">{label}</p>
            <p className="text-xs text-gray-400 mb-4">Drag and drop, or click to browse</p>
            <button onClick={onButtonClick} className="btn-secondary text-xs px-4 py-2">
              Select Files
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
