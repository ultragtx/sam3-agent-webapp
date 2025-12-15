import { useState, useEffect, useRef } from 'react';
import { Upload, Settings, Play, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { uploadImage, runAgentStream, healthCheck, loadModels, getConfig } from './services/api';
import type { MLLMConfig, AgentResult, StreamEvent, AgentRound, SAM3Call } from './types';
import ImageUpload from './components/ImageUpload';
import ConfigPanel from './components/ConfigPanel';
import AgentViewer from './components/AgentViewer';

function App() {
  const [imagePath, setImagePath] = useState<string>('');
  const [imagePreview, setImagePreview] = useState<string>('');
  const [textPrompt, setTextPrompt] = useState<string>('');
  const [mllmConfig, setMllmConfig] = useState<Partial<MLLMConfig>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [agentResult, setAgentResult] = useState<AgentResult | null>(null);
  const [error, setError] = useState<string>('');
  const [backendStatus, setBackendStatus] = useState<string>('checking');
  const [sam3Loaded, setSam3Loaded] = useState(false);
  
  // Real-time streaming state
  const [currentRounds, setCurrentRounds] = useState<AgentRound[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    checkBackend();
    fetchConfig();
  }, []);

  const checkBackend = async () => {
    try {
      const status = await healthCheck();
      setBackendStatus(status.backend);
      setSam3Loaded(status.sam3_loaded);
      
      if (!status.sam3_loaded) {
        // Try to load models
        await loadModels();
        const newStatus = await healthCheck();
        setSam3Loaded(newStatus.sam3_loaded);
      }
    } catch (err) {
      setBackendStatus('error');
      setError('Backend connection failed');
    }
  };

  const fetchConfig = async () => {
    try {
      const config = await getConfig();
      setMllmConfig(config.mllm);
    } catch (err) {
      console.error('Failed to fetch config:', err);
    }
  };

  const handleImageUpload = async (file: File) => {
    try {
      setError('');
      const result = await uploadImage(file);
      setImagePath(result.filepath);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError('Failed to upload image');
      console.error(err);
    }
  };

  const handleRunAgent = async () => {
    if (!imagePath || !textPrompt) {
      setError('Please upload an image and enter a prompt');
      return;
    }

    setIsRunning(true);
    setError('');
    setAgentResult(null);
    setCurrentRounds([]);

    try {
      // Close any existing event source
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      eventSourceRef.current = runAgentStream(
        imagePath,
        textPrompt,
        mllmConfig,
        true,
        (event: StreamEvent) => {
          handleStreamEvent(event);
        },
        (result: AgentResult) => {
          setAgentResult(result);
          setIsRunning(false);
        },
        (error: string) => {
          setError(error);
          setIsRunning(false);
        }
      );
    } catch (err: any) {
      setError(err.response?.data?.message || 'Agent execution failed');
      console.error(err);
      setIsRunning(false);
    }
  };

  const handleStreamEvent = (event: StreamEvent) => {
    console.log('Stream event:', event);
    
    switch (event.type) {
      case 'round_start':
        // Initialize new round
        setCurrentRounds(prev => [
          ...prev,
          {
            round: event.data.round,
            messages: [],
            generated_text: '',
            llm_chunks: [],
            sam3_calls: [],
            status: 'pending'
          }
        ]);
        break;
        
      case 'llm_start':
        // Mark LLM as running for current round
        setCurrentRounds(prev => prev.map(r => 
          r.round === event.data.round 
            ? { ...r, status: 'llm_running' }
            : r
        ));
        break;
        
      case 'llm_chunk':
        // Append LLM chunk
        setCurrentRounds(prev => prev.map(r => 
          r.round === event.data.round 
            ? { 
                ...r, 
                llm_chunks: [...(r.llm_chunks || []), event.data.chunk],
                generated_text: event.data.accumulated
              }
            : r
        ));
        break;
        
      case 'llm_complete':
        // Mark LLM as complete
        setCurrentRounds(prev => prev.map(r => 
          r.round === event.data.round 
            ? { ...r, generated_text: event.data.text, status: 'sam3_running' }
            : r
        ));
        break;
        
      case 'sam3_start':
        // Add SAM3 call
        setCurrentRounds(prev => prev.map(r => {
          if (r.status === 'sam3_running') {
            return {
              ...r,
              sam3_calls: [
                ...(r.sam3_calls || []),
                {
                  text_prompt: event.data.text_prompt,
                  num_masks: 0,
                  status: 'running'
                }
              ]
            };
          }
          return r;
        }));
        break;
        
      case 'sam3_complete':
        // Update SAM3 call with results
        setCurrentRounds(prev => prev.map(r => {
          if (r.status === 'sam3_running') {
            const sam3Calls = r.sam3_calls || [];
            const lastIndex = sam3Calls.length - 1;
            const updatedCalls = [...sam3Calls];
            if (lastIndex >= 0) {
              updatedCalls[lastIndex] = {
                ...updatedCalls[lastIndex],
                num_masks: event.data.num_masks,
                json_path: event.data.json_path,
                image_path: event.data.image_path,
                status: 'complete'
              };
            }
            return {
              ...r,
              sam3_calls: updatedCalls,
              status: 'complete'
            };
          }
          return r;
        }));
        break;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            SAM3 Agent Web Application
          </h1>
          <p className="text-gray-400">
            Intelligent visual segmentation with iterative MLLM reasoning
          </p>
          
          {/* Status Indicators */}
          <div className="flex gap-4 mt-4">
            <div className="flex items-center gap-2 px-3 py-1 bg-gray-800 rounded-full">
              {backendStatus === 'running' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <XCircle className="w-4 h-4 text-red-400" />
              )}
              <span className="text-sm">Backend: {backendStatus}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1 bg-gray-800 rounded-full">
              {sam3Loaded ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />
              )}
              <span className="text-sm">SAM3: {sam3Loaded ? 'Loaded' : 'Loading...'}</span>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel: Input */}
          <div className="lg:col-span-1 space-y-6">
            {/* Image Upload */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Image
              </h2>
              <ImageUpload onImageUpload={handleImageUpload} preview={imagePreview} />
            </div>

            {/* Text Prompt */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <h2 className="text-xl font-semibold mb-4">Query</h2>
              <textarea
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                placeholder="Enter your segmentation query (e.g., 'person holding a phone')"
                className="w-full h-32 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
            </div>

            {/* MLLM Configuration */}
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                MLLM Configuration
              </h2>
              <ConfigPanel config={mllmConfig} onChange={setMllmConfig} />
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunAgent}
              disabled={isRunning || !imagePath || !textPrompt || !sam3Loaded}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-all duration-200 flex items-center justify-center gap-2"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Running Agent...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Start Agent
                </>
              )}
            </button>

            {/* Error Display */}
            {error && (
              <div className="bg-red-900/50 border border-red-500 rounded-lg p-4">
                <p className="text-red-200">{error}</p>
              </div>
            )}
          </div>

          {/* Right Panel: Results */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl min-h-[600px]">
              <h2 className="text-xl font-semibold mb-4">Agent Execution</h2>
              {(agentResult || currentRounds.length > 0) ? (
                <AgentViewer 
                  result={agentResult} 
                  liveRounds={currentRounds}
                  isRunning={isRunning}
                />
              ) : (
                <div className="flex items-center justify-center h-[500px] text-gray-500">
                  <div className="text-center">
                    <Play className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>Start the agent to see results</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
