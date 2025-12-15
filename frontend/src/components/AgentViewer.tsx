import { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import type { AgentResult } from '../types';

interface AgentViewerProps {
  result: AgentResult;
}

export default function AgentViewer({ result }: AgentViewerProps) {
  const [expandedRounds, setExpandedRounds] = useState<Set<number>>(new Set([1]));

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

  return (
    <div className="space-y-6">
      {/* Status Summary */}
      <div className="bg-gray-700 rounded-lg p-4 flex items-center gap-3">
        {getStatusIcon()}
        <div className="flex-1">
          <h3 className="font-semibold">{getStatusText()}</h3>
          {result.message && (
            <p className="text-sm text-gray-400">{result.message}</p>
          )}
        </div>
        {result.final_output && (
          <div className="text-right">
            <p className="text-sm text-gray-400">Masks Found</p>
            <p className="text-2xl font-bold text-blue-400">
              {result.final_output.num_masks}
            </p>
          </div>
        )}
      </div>

      {/* Final Result */}
      {result.final_output && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Final Result</h3>
          <img
            src={`/api/outputs/${result.final_output.image_path.split('/').pop()}`}
            alt="Final segmentation"
            className="w-full rounded-lg border border-gray-600"
          />
        </div>
      )}

      {/* Iteration History */}
      <div className="space-y-2">
        <h3 className="text-lg font-semibold mb-3">Iteration History ({result.history.length} rounds)</h3>
        {result.history.map((round) => (
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
            </button>

            {expandedRounds.has(round.round) && (
              <div className="px-4 pb-4 space-y-3">
                {/* MLLM Messages */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Messages</h4>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {round.messages.map((msg, idx) => (
                      <div key={idx} className="bg-gray-800 rounded p-3 text-sm">
                        <div className="font-semibold text-blue-400 mb-1">
                          {msg.role}
                        </div>
                        {Array.isArray(msg.content) ? (
                          <div className="space-y-1">
                            {msg.content.map((item: any, i: number) => (
                              <div key={i}>
                                {item.type === 'text' && (
                                  <p className="text-gray-300 whitespace-pre-wrap">
                                    {item.text}
                                  </p>
                                )}
                                {item.type === 'image' && (
                                  <p className="text-gray-500 italic">
                                    [Image: {item.image}]
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-gray-300">{msg.content}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Generated Response */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">
                    MLLM Response
                  </h4>
                  <div className="bg-gray-800 rounded p-3 text-sm">
                    <pre className="whitespace-pre-wrap text-gray-300 font-mono text-xs">
                      {round.generated_text}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
