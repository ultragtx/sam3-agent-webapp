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
}

export interface AgentResult {
  status: string;
  history: AgentRound[];
  final_output?: {
    json_path: string;
    image_path: string;
    num_masks: number;
    outputs: any;
  };
  message?: string;
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
