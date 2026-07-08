// Shared API client service wrapping all v1 endpoints
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface DashboardResponse {
  baseline_metrics: {
    task1: {
      mean_quality_score: number;
      processed_frames: number;
      duration_sec: number;
    };
    task2: {
      sequence: string;
      matches_at_0_75: number;
      inlier_ratio_at_0_75: number;
      duration_sec: number;
    };
    task3: {
      scene: string;
      left_to_middle: {
        total_matches_after_lowe: number;
        ransac_inliers: number;
        inlier_ratio: number;
      };
      right_to_middle: {
        total_matches_after_lowe: number;
        ransac_inliers: number;
        inlier_ratio: number;
      };
      average_inlier_ratio: number;
    };
    task4: {
      active_backbone: string;
      test_accuracy: number;
      macro_f1: number;
      inference_fps: number;
      inference_duration_sec: number;
    };
  };
  backbone_comparison: {
    resnet18: {
      accuracy: number;
      f1_score: number;
      params_m: number;
      size_mb: number;
      speed_fps: number;
      train_time_sec: number;
    };
    mobilenetv2: {
      accuracy: number;
      f1_score: number;
      params_m: number;
      size_mb: number;
      speed_fps: number;
      train_time_sec: number;
    };
  };
}

export interface LanePresetDetail {
  image_name: string;
  num_lines: number;
  quality_score: number;
}

export interface LaneDetectionResponse {
  preset_mean_scores: Record<string, number>;
  overall_mean_quality_score: number;
  overlay_base64: string;
  results_by_preset: Record<string, LanePresetDetail[]>;
}

export interface MatchStats {
  match_count: number;
  ransac_inliers: number;
  inlier_ratio: number;
}

export interface MatchingResponse {
  ratios_sweep: Record<string, MatchStats>;
  matches_at_0_75: number;
  inlier_ratio_at_0_75: number;
  visualization_base64: string;
}

export interface PanoramaSubmitResponse {
  job_id: string;
  status: string;
}

export interface PanoramaPollResponse {
  job_id: string;
  status: string; // "pending" | "running" | "completed" | "failed"
  error?: string;
  panorama_base64?: string;
  pair_metrics?: Record<string, {
    total_matches: number;
    inliers: number;
    inlier_ratio: number;
    status: string;
  }>;
  average_inlier_ratio?: number;
}

export interface ClassifyResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcam_base64: string;
}

export const ApiClient = {
  async getDashboard(): Promise<DashboardResponse> {
    const res = await fetch(`${API_BASE}/api/v1/dashboard`);
    if (!res.ok) throw new Error('Failed to retrieve dashboard metrics.');
    return res.json();
  },

  async detectLanes(files: File[]): Promise<LaneDetectionResponse> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const res = await fetch(`${API_BASE}/api/v1/lanes`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to run lane detection' }));
      throw new Error(err.detail || 'Unsupported type or upload error.');
    }
    return res.json();
  },

  async matchFeatures(img1: File, img2: File): Promise<MatchingResponse> {
    const formData = new FormData();
    formData.append('img1_file', img1);
    formData.append('img2_file', img2);

    const res = await fetch(`${API_BASE}/api/v1/match`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed SIFT matching' }));
      throw new Error(err.detail || 'Feature matching parameter error.');
    }
    return res.json();
  },

  async submitPanoramaJob(files: File[], anchor: string): Promise<PanoramaSubmitResponse> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('anchor', anchor);

    const res = await fetch(`${API_BASE}/api/v1/panorama`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Stitch job submission failed' }));
      throw new Error(err.detail || 'Panorama upload image count error.');
    }
    return res.json();
  },

  async pollPanoramaJob(jobId: string): Promise<PanoramaPollResponse> {
    const res = await fetch(`${API_BASE}/api/v1/panorama/jobs/${jobId}`);
    if (!res.ok) throw new Error('Polling request failed.');
    return res.json();
  },

  async classifyImage(file: File, backbone: string): Promise<ClassifyResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/api/v1/classify?backbone=${backbone}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Classification error' }));
      throw new Error(err.detail || 'Selected backbone checkpoint not loaded.');
    }
    return res.json();
  }
};
