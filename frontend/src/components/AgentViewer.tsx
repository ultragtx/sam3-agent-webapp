import { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, CheckCircle, XCircle, AlertCircle, Loader2, Brain, Image as ImageIcon } from 'lucide-react';
import type { AgentResult, AgentRound } from '../types';

interface AgentViewerProps {
  result?: AgentResult | null;
  liveRounds?: AgentRound[];
  isRunning?: boolean;
}

export default function AgentViewer({ result, liveRounds = [], isRunning = false }: AgentViewerProps) {
  const [expandedRounds, setExpandedRounds] = useState<Set<number>>(new Set([1]));

  // Use live rounds if agent is running, otherwise use final result
  const displayRounds = isRunning ? liveRounds : (result?.history || []);

  useEffect(() => {
    // Auto-expand the latest round
    if (displayRounds.length > 0) {
      const latestRound = displayRounds[displayRounds.length - 1].round;
      setExpandedRounds(prev => new Set([...prev, latestRound]));
    }
  }, [displayRounds.length]);

  const toggleRound = (round: number) => {
    const newExpanded = new Set(expandedRounds);
    if (newExpanded.has(round)) {
      newExpanded.delete(round);
    } else {
      newExpanded.add(round);
    }
    setExpandedRounds(newExpanded);
  };

  const getStatusIcon = () => {
    if (isRunning) {
      return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />;
    }
    
    if (!result) return null;
    
    switch (result.status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'no_masks':
        return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      default:
        return <XCircle className="w-5 h-5 text-red-400" />;
    }
  };

  const getStatusText = () => {
    if (isRunning) {
      return 'Agent running...';
    }
    
    if (!result) return 'No result';
    
    switch (result.status) {
      case 'success':
        return 'Agent completed successfully';
      case 'no_masks':
        return 'No masks found';
      case 'incomplete':
        return 'Agent stopped (incomplete)';
      default:
        return 'Error';
    }
  };

  const getRoundStatusIcon = (round: AgentRound) => {
    switch (round.status) {
      case 'llm_running':
        return <Brain className="w-4 h-4 text-blue-400 animate-pulse" />;
      case 'sam3_running':
        return <ImageIcon className="w-4 h-4 text-purple-400 animate-pulse" />;
      case 'complete':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      default:
        return <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Summary */}
      <div className="bg-gray-700 rounded-lg p-4 flex items-center gap-3">
        {getStatusIcon()}
        <div className="flex-1">
          <h3 className="font-semibold">{getStatusText()}</h3>
          {result?.message && (
            <p className="text-sm text-gray-400">{result.message}</p>
          )}
        </div>
        {result?.final_output && (
          <div className="text-right">
            <p className="text-sm text-gray-400">Masks Found</p>
            <p className="text-2xl font-bold text-blue-400">
              {result.final_output.num_masks}
            </p>
          </div>
        )}
      </div>

      {/* Final Result */}
      {result?.final_output && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Final Result</h3>
          <img
            src={`/api/outputs/${result.final_output.image_path}`}
            alt="Final segmentation"
            className="w-full rounded-lg border border-gray-600"
          />
        </div>
      )}

      {/* Iteration History */}
      {displayRounds.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-lg font-semibold mb-3">Iteration History ({displayRounds.length} rounds)</h3>
          {displayRounds.map((round) => (
          <div key={round.round} className="bg-gray-700 rounded-lg overflow-hidden">
            <button
              onClick={() => toggleRound(round.round)}
              className="w-full px-4 py-3 flex items-center gap-2 hover:bg-gray-600 transition-colors"
            >
              {expandedRounds.has(round.round) ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              <span className="font-medium">Round {round.round}</span>
              <div className="ml-auto flex items-center gap-2">
                {getRoundStatusIcon(round)}
              </div>
            </button>

            {expandedRounds.has(round.round) && (
              <div className="px-4 pb-4 space-y-3">
                {/* LLM Response Section */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-blue-400" />
                    <h4 className="text-sm font-semibold text-gray-400">
                      MLLM Response
                      {round.status === 'llm_running' && (
                        <span className="ml-2 text-blue-400">(streaming...)</span>
                      )}
                    </h4>
                  </div>
                  <div className="bg-gray-800 rounded p-3 text-sm">
                    {round.generated_text ? (
                      <pre className="whitespace-pre-wrap text-gray-300 font-mono text-xs">
                        {round.generated_text}
                        {round.status === 'llm_running' && (
                          <span className="animate-pulse">▋</span>
                        )}
                      </pre>
                    ) : (
                      <div className="flex items-center gap-2 text-gray-500">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Waiting for response...</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* SAM3 Calls Section */}
                {round.sam3_calls && round.sam3_calls.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <ImageIcon className="w-4 h-4 text-purple-400" />
                      <h4 className="text-sm font-semibold text-gray-400">
                        SAM3 Segmentation Calls
                      </h4>
                    </div>
                    <div className="space-y-2">
                      {round.sam3_calls.map((call, idx) => (
                        <div key={idx} className="bg-gray-800 rounded p-3 text-sm">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-purple-400 font-medium">
                              Prompt: "{call.text_prompt}"
                            </span>
                            {call.status === 'running' ? (
                              <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
                            ) : (
                              <CheckCircle className="w-4 h-4 text-green-400" />
                            )}
                          </div>
                          {(call.status === 'complete' || !call.status) && (
                            <>
                              <div className="text-gray-400">
                                Generated {call.num_masks} mask{call.num_masks !== 1 ? 's' : ''}
                              </div>
                              {call.image_path && (
                                <img
                                  src={`/api/outputs/${call.image_path}`}
                                  alt={`SAM3 result for ${call.text_prompt}`}
                                  className="mt-2 w-full rounded border border-gray-600"
                                />
                              )}
                            </>
                          )}
                          {call.status === 'running' && (
                            <div className="text-gray-500">Segmenting...</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      )}
    </div>
  );
}
