import { useState, useRef, useEffect, KeyboardEvent } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  onSkip: () => void;
  disabled?: boolean;
  placeholder?: string;
  lastError?: string | null;
}

export function ChatInput({ onSend, onSkip, disabled, placeholder, lastError }: ChatInputProps) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea up to 4 lines
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const lineHeight = 20;
    const maxHeight = lineHeight * 4 + 24; // 4 lines + padding
    textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + 'px';
  }, [text]);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (trimmed) {
      onSend(trimmed);
      setText('');
    } else {
      // Empty submit = skip/continue
      onSkip();
    }
  };

  const handleRegenerate = () => {
    const trimmed = text.trim();
    const parts: string[] = [];
    if (trimmed) {
      parts.push(trimmed);
    }
    parts.push('请根据以上报错信息重新生成代码。 Please REGENERATE the code based on the error above.');
    onSend(parts.join('\n'));
    setText('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const hasText = text.trim().length > 0;
  const isAwaitingInput = !disabled;
  const hasError = !!lastError;

  return (
    <div className="space-y-2">
      {/* Error context display */}
      {isAwaitingInput && hasError && (
        <div className="rounded-md border border-red-800/30 bg-red-950/30 overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-red-950/50 border-b border-red-800/20">
            <svg className="w-3.5 h-3.5 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-xs font-display font-medium text-red-400">上一轮执行报错 / Last Execution Error</span>
          </div>
          <pre className="px-3 py-2 text-xs text-red-300/80 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto leading-relaxed">
            {lastError}
          </pre>
        </div>
      )}

      {/* Input area */}
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={
            hasError
              ? '输入修改建议（可选），然后点击重新生成 / Enter correction hints, then click Regenerate'
              : placeholder
          }
          rows={1}
          className="w-full pl-4 pr-48 py-3 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary placeholder-text-tertiary resize-none overflow-y-hidden focus:outline-none focus:ring-1 focus:ring-accent/30 focus:border-accent/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all leading-5"
        />
        {/* Buttons — right-aligned */}
        {isAwaitingInput && (
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
            {/* Regenerate button — shown when there's an error */}
            {hasError && (
              <button
                onClick={handleRegenerate}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-amber-600/80 text-white hover:bg-amber-500 transition-all"
                title="重新生成代码 / Regenerate code with error context"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                重新生成
              </button>
            )}
            {/* Send / Continue button */}
            <button
              onClick={handleSubmit}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                hasText
                  ? 'bg-accent text-black hover:bg-accent-light'
                  : 'bg-surface-overlay text-text-secondary hover:bg-surface-border-light hover:text-text-primary border border-surface-border'
              }`}
              title={hasText ? 'Send feedback (Enter)' : 'Continue without feedback (Enter)'}
            >
              {hasText ? (
                <>
                  Send
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
                  </svg>
                </>
              ) : (
                <>
                  Continue
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                  </svg>
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
