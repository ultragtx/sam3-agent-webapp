import axios from 'axios';
import type { MLLMConfig, AgentResult, SAM3Result, HealthStatus, StreamEvent } from '../types';

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

export const runAgentStream = (
  imagePath: string,
  textPrompt: string,
  mllmConfig?: Partial<MLLMConfig>,
  debug: boolean = true,
  onEvent?: (event: StreamEvent) => void,
  onComplete?: (result: AgentResult) => void,
  onError?: (error: string) => void
): EventSource => {
  // Use EventSource for SSE
  const params = new URLSearchParams({
    image_path: imagePath,
    text_prompt: textPrompt,
    debug: debug.toString(),
  });
  
  if (mllmConfig) {
    params.append('mllm_config', JSON.stringify(mllmConfig));
  }
  
  // For POST with SSE, we need to use fetch instead of EventSource
  // Create a custom implementation
  const controller = new AbortController();
  
  fetch('/api/agent/run-stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_path: imagePath,
      text_prompt: textPrompt,
      mllm_config: mllmConfig,
      debug,
    }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error('No response body');
      }
      
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const event: StreamEvent = JSON.parse(data);
              
              if (onEvent) {
                onEvent(event);
              }
              
              if (event.type === 'agent_complete' && onComplete) {
                onComplete(event.data as AgentResult);
              } else if (event.type === 'error' && onError) {
                onError(event.data.message);
              }
            } catch (e) {
              console.error('Failed to parse SSE message:', e);
            }
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== 'AbortError' && onError) {
        onError(error.message);
      }
    });
  
  // Return a mock EventSource-like object with close method
  return {
    close: () => controller.abort(),
  } as EventSource;
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
