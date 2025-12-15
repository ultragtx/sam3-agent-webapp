export interface MLLMConfig {
  api_base: string;
  api_key: string;
  model: string;
  max_tokens: number;
}

export interface AgentRound {
  round: number;
  messages: any[];
  generated_text: string;
  llm_chunks?: string[];
  sam3_calls?: SAM3Call[];
  status: 'pending' | 'llm_running' | 'sam3_running' | 'complete';
}

export interface SAM3Call {
  text_prompt: string;
  num_masks: number;
  json_path?: string;
  image_path?: string;
  status: 'pending' | 'running' | 'complete';
}

export interface AgentResult {
  status: string;
  history?: AgentRound[];
  final_output?: {
    json_path: string;
    image_path: string;
    num_masks: number;
    outputs: any;
  };
  message?: string;
}

export interface StreamEvent {
  type: 'agent_start' | 'round_start' | 'llm_start' | 'llm_chunk' | 'llm_complete' 
       | 'sam3_start' | 'sam3_complete' | 'agent_complete' | 'error';
  data: any;
}

export interface SAM3Result {
  status: string;
  result: {
    json_path: string;
    image_path: string;
    num_masks: number;
    outputs: any;
  };
}

export interface HealthStatus {
  status: string;
  sam3_loaded: boolean;
  backend: string;
}
