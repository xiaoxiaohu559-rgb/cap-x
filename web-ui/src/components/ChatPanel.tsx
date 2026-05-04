import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, SessionState } from '../types/messages';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';

interface ChatPanelProps {
  messages: ChatMessage[];
  state: SessionState;
  onSendMessage: (text: string) => void;
  onResume: () => void;
  onRetry?: (hint: string) => void;
  onAutoEvolve?: (count: number, hint: string) => void;
  onStopAutoEvolve?: () => void;
  autoEvolveRemaining?: number;
  autoEvolveTotal?: number;
  taskPrompt: string | null;
}

export function ChatPanel({
  messages,
  state,
  onSendMessage,
  onResume,
  onRetry,
  onAutoEvolve,
  onStopAutoEvolve,
  autoEvolveRemaining = 0,
  autoEvolveTotal = 0,
  taskPrompt,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [taskExpanded, setTaskExpanded] = useState(false);
  const [retryHint, setRetryHint] = useState('');
  const [evolveCount, setEvolveCount] = useState(3);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  const canInput = state === 'awaiting_user_input';
  const trialDone = state === 'complete' || state === 'error';

  const taskLines = taskPrompt?.split('\n') || [];
  const charCount = taskPrompt?.length || 0;
  const needsExpansion = taskLines.length > 3 || charCount > 200;

  // Extract last stderr from the most recent code_execution_result message
  const lastError = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.type === 'code_execution' && !m.isExecuting && m.stderr) {
        return m.stderr;
      }
    }
    return null;
  })();

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Task prompt display */}
      {taskPrompt && (
        <div className="flex-shrink-0 bg-surface-raised border-b border-surface-border">
          <button
            onClick={() => needsExpansion && setTaskExpanded(!taskExpanded)}
            className="w-full px-5 py-3 flex items-center justify-between hover:bg-surface-overlay/50 transition-colors"
          >
            <div className="flex items-center gap-2.5">
              <div className="w-6 h-6 rounded bg-accent/10 border border-accent/20 flex items-center justify-center">
                <svg className="w-3.5 h-3.5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>
              <span className="text-xs font-display font-semibold text-accent uppercase tracking-wide">Task</span>
            </div>
            {needsExpansion && (
              <span className="flex items-center gap-1 text-xs font-display text-text-tertiary">
                {taskExpanded ? 'Collapse' : 'Expand'}
                <svg className={`w-3.5 h-3.5 transition-transform ${taskExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </span>
            )}
          </button>

          {/* Preview (collapsed) */}
          {!taskExpanded && (
            <div className="px-5 pb-3 relative">
              <div className="text-sm text-text-primary whitespace-pre-wrap overflow-hidden leading-relaxed" style={{ maxHeight: '4.5em' }}>
                {taskLines.slice(0, 3).join('\n')}
              </div>
              {needsExpansion && (
                <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-surface-raised to-transparent pointer-events-none" />
              )}
            </div>
          )}

          {/* Full content (expanded) */}
          {taskExpanded && (
            <div className="px-5 pb-4 max-h-64 overflow-y-auto">
              <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed">{taskPrompt}</p>
            </div>
          )}
        </div>
      )}

      {/* Auto-evolve progress banner */}
      {autoEvolveRemaining > 0 && (
        <div className="flex-shrink-0 bg-accent/10 border-b border-accent/20 px-5 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-5 h-5 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
            <span className="text-xs font-display font-semibold text-accent tracking-wide">
              持续进化 Round {autoEvolveTotal - autoEvolveRemaining}/{autoEvolveTotal}
            </span>
            <span className="text-xs text-text-tertiary">
              剩余 {autoEvolveRemaining} 轮
            </span>
          </div>
          <button
            onClick={onStopAutoEvolve}
            className="flex items-center gap-1 px-2.5 py-1 rounded text-xs font-display font-medium bg-red-600/20 text-red-400 border border-red-600/30 hover:bg-red-600/30 transition-colors"
          >
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
            </svg>
            停止
          </button>
        </div>
      )}

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-5 py-5 space-y-4 bg-surface"
        role="log"
        aria-live="polite"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            {/* Large logo with subtle glow */}
            <div className="relative mb-8">
              <div className="absolute inset-0 blur-2xl bg-accent/5 rounded-full scale-150" />
              <img src="/capx_logo.svg" alt="CaP-X" className="w-16 h-16 relative" />
            </div>

            {/* Title — large, tracked, uppercase */}
            <h2 className="text-display font-bold font-display text-text-primary tracking-widest uppercase mb-2">CaP-X</h2>
            <div className="gold-rule w-16 mx-auto mb-4" />
            <p className="text-sm font-display text-text-tertiary tracking-wide mb-12">Code-as-Policy Agent Framework</p>

            {/* Steps — minimal, spaced */}
            <div className="flex flex-col gap-4 text-left">
              <div className="flex items-center gap-4 group">
                <span className="text-[11px] font-mono font-bold text-accent/50 group-hover:text-accent transition-colors w-6">01</span>
                <span className="text-sm font-display text-text-secondary group-hover:text-text-primary transition-colors">Select a configuration</span>
              </div>
              <div className="flex items-center gap-4 group">
                <span className="text-[11px] font-mono font-bold text-accent/50 group-hover:text-accent transition-colors w-6">02</span>
                <span className="text-sm font-display text-text-secondary group-hover:text-text-primary transition-colors">Start a trial</span>
              </div>
              <div className="flex items-center gap-4 group">
                <span className="text-[11px] font-mono font-bold text-accent/50 group-hover:text-accent transition-colors w-6">03</span>
                <span className="text-sm font-display text-text-secondary group-hover:text-text-primary transition-colors">Watch the agent write code</span>
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
      </div>

      {/* Retry panel — visible after trial completes or errors */}
      {trialDone && onRetry && (
        <div className="flex-shrink-0 border-t border-surface-border bg-surface-raised px-4 py-3 space-y-2">
          {/* Error context */}
          {lastError && (
            <div className="rounded-md border border-red-800/30 bg-red-950/30 overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-red-950/50 border-b border-red-800/20">
                <svg className="w-3.5 h-3.5 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-xs font-display font-medium text-red-400">上一轮报错 / Last Error</span>
              </div>
              <pre className="px-3 py-2 text-xs text-red-300/80 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto leading-relaxed">
                {lastError}
              </pre>
            </div>
          )}
          {/* Input + Regenerate button */}
          <div className="flex gap-2 items-end">
            <textarea
              value={retryHint}
              onChange={(e) => setRetryHint(e.target.value)}
              placeholder="输入修改建议（可选），然后点击重新生成 / Enter correction hints, then click Regenerate"
              rows={1}
              className="flex-1 px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary placeholder-text-tertiary font-sans resize-none focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent transition-all"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  onRetry(retryHint.trim());
                  setRetryHint('');
                }
              }}
            />
            <button
              onClick={() => {
                onRetry(retryHint.trim());
                setRetryHint('');
              }}
              className="flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-display font-bold bg-amber-600 text-white hover:bg-amber-500 active:scale-[0.98] transition-all shadow-sm whitespace-nowrap"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              重新生成
            </button>
          </div>

          {/* Auto-evolve controls */}
          {onAutoEvolve && (
            <div className="flex items-center gap-2 pt-1">
              <span className="text-xs text-text-tertiary font-display">持续进化</span>
              <input
                type="number"
                value={evolveCount}
                onChange={(e) => setEvolveCount(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
                min={1}
                max={20}
                className="w-14 px-2 py-1 bg-surface-sunken border border-surface-border rounded text-xs text-text-primary font-mono text-center focus:outline-none focus:ring-1 focus:ring-accent/40"
              />
              <span className="text-xs text-text-tertiary font-display">轮</span>
              <button
                onClick={() => {
                  onAutoEvolve(evolveCount, retryHint.trim());
                  setRetryHint('');
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-display font-bold bg-accent text-black hover:bg-accent-light active:scale-[0.98] transition-all shadow-sm whitespace-nowrap"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                开始进化
              </button>
            </div>
          )}
        </div>
      )}

      {/* Input area — for mid-trial feedback (awaiting_user_input) */}
      {!trialDone && (
        <div className="flex-shrink-0 border-t border-surface-border bg-surface-raised px-4 py-4">
          <ChatInput
            onSend={onSendMessage}
            onSkip={onResume}
            disabled={!canInput}
            lastError={canInput ? lastError : null}
            placeholder={
              canInput
                ? 'Type your feedback...'
                : state === 'running'
                ? 'Model is generating...'
                : 'Waiting for trial to start...'
            }
          />
        </div>
      )}
    </div>
  );
}
