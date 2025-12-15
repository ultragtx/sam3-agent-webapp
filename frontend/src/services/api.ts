import axios from 'axios';
import type { MLLMConfig, AgentResult, SAM3Result, HealthStatus } from '../types';

// Use relative path to go through Vite proxy, avoiding CORS issues
const API_BASE = '';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 minutes for long agent runs
});

export const healthCheck = async (): Promise<HealthStatus> => {
  const response = await api.get('/api/health');
  return response.data;
};

export const loadModels = async (): Promise<any> => {
  const response = await api.post('/api/models/load');
  return response.data;
};

export const uploadImage = async (file: File): Promise<{ filepath: string }> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
  return response.data;
};

export const runAgent = async (
  imagePath: string,
  textPrompt: string,
  mllmConfig?: Partial<MLLMConfig>,
  debug: boolean = true
): Promise<AgentResult> => {
  const response = await api.post('/api/agent/run', {
    image_path: imagePath,
    text_prompt: textPrompt,
    mllm_config: mllmConfig,
    debug
  });
  
  return response.data;
};

export const sam3Segment = async (
  imagePath: string,
  textPrompt: string
): Promise<SAM3Result> => {
  const response = await api.post('/api/sam3/segment', {
    image_path: imagePath,
    text_prompt: textPrompt
  });
  
  return response.data;
};

export const getConfig = async (): Promise<any> => {
  const response = await api.get('/api/config');
  return response.data;
};

export const getImageUrl = (path: string): string => {
  // Convert backend path to URL
  if (path.startsWith('/')) {
    return `/api${path}`;
  }
  return `/api/outputs/${path}`;
};

export default api;
