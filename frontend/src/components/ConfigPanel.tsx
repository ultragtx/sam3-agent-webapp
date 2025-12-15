import type { MLLMConfig } from '../types';

interface ConfigPanelProps {
  config: Partial<MLLMConfig>;
  onChange: (config: Partial<MLLMConfig>) => void;
}

export default function ConfigPanel({ config, onChange }: ConfigPanelProps) {
  const handleChange = (key: keyof MLLMConfig, value: string | number) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          API Base URL
        </label>
        <input
          type="text"
          value={config.api_base || ''}
          onChange={(e) => handleChange('api_base', e.target.value)}
          placeholder="http://localhost:8000/v1"
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          API Key
        </label>
        <input
          type="password"
          value={config.api_key || ''}
          onChange={(e) => handleChange('api_key', e.target.value)}
          placeholder="Your API key"
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Model Name
        </label>
        <input
          type="text"
          value={config.model || ''}
          onChange={(e) => handleChange('model', e.target.value)}
          placeholder="Qwen/Qwen2-VL-7B-Instruct"
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Max Tokens
        </label>
        <input
          type="number"
          value={config.max_tokens || 4096}
          onChange={(e) => handleChange('max_tokens', parseInt(e.target.value))}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
        />
      </div>
    </div>
  );
}
