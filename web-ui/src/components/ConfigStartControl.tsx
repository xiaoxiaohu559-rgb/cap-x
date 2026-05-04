import { useState, useEffect, useMemo } from 'react';
import type { SessionState, StartTrialRequest } from '../types/messages';

interface ConfigEntry {
  path: string;
  label?: string | null;
  available: boolean;
  reason?: string | null;
}

/** Group configs by their parent directory for a cleaner dropdown. */
function groupConfigs(configs: ConfigEntry[]): Map<string, ConfigEntry[]> {
  const groups = new Map<string, ConfigEntry[]>();
  for (const config of configs) {
    const parts = config.path.split('/');
    const category = parts.length >= 3 ? parts[1] : 'other';
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category)!.push(config);
  }
  return groups;
}

/** Human-readable category label. */
function categoryLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export type InputMode = 'config' | 'text';

interface ConfigStartControlProps {
  state: SessionState;
  configPath: string | null;
  error: string | null;
  loadConfig: (path: string) => Promise<unknown>;
  startTrial: (request: StartTrialRequest) => Promise<boolean>;
  stopTrial: () => void;
  reset: () => void;
  model: string;
  serverUrl: string;
  temperature: number;
  awaitUserInput: boolean;
  inputMode: InputMode;
  onInputModeChange: (mode: InputMode) => void;
}

export function ConfigStartControl({
  state,
  configPath,
  error,
  loadConfig,
  startTrial,
  stopTrial,
  reset,
  model,
  serverUrl,
  temperature,
  awaitUserInput,
  inputMode,
  onInputModeChange,
}: ConfigStartControlProps) {
  const [configs, setConfigs] = useState<ConfigEntry[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const grouped = useMemo(() => groupConfigs(configs), [configs]);

  const isRunning = state === 'running' || state === 'awaiting_user_input';
  const canStart = state === 'idle' || state === 'complete' || state === 'error';

  useEffect(() => {
    fetch('/api/configs')
      .then((res) => res.json())
      .then((data) => {
        const raw = data.configs || [];
        const entries: ConfigEntry[] = raw.map((c: string | ConfigEntry) =>
          typeof c === 'string' ? { path: c, available: true } : c
        );
        setConfigs(entries);
      })
      .catch((err) => console.error('Failed to fetch configs:', err));
  }, []);

  useEffect(() => {
    if (configPath && selectedConfig !== configPath) {
      setSelectedConfig(configPath);
    }
  }, [configPath]);

  const handleConfigChange = async (path: string) => {
    if (!path) return;
    setSelectedConfig(path);
    setLoading(true);
    try {
      await loadConfig(path);
    } catch (err) {
      console.error('Failed to load config:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStart = async () => {
    if (!configPath) return;
    await startTrial({
      config_path: configPath,
      model,
      server_url: serverUrl,
      temperature,
      await_user_input_each_turn: awaitUserInput,
    });
  };

  return (
    <div className="flex items-center gap-2">
      {/* Mode Toggle */}
      <div className="flex rounded-md border border-surface-border overflow-hidden flex-shrink-0">
        <button
          onClick={() => onInputModeChange('text')}
          disabled={isRunning}
          className={`px-2.5 py-1 text-xs font-display font-medium transition-colors ${
            inputMode === 'text'
              ? 'bg-accent text-black'
              : 'bg-surface-sunken text-text-tertiary hover:text-text-secondary'
          } disabled:opacity-50`}
        >
          文本指令
        </button>
        <button
          onClick={() => onInputModeChange('config')}
          disabled={isRunning}
          className={`px-2.5 py-1 text-xs font-display font-medium transition-colors border-l border-surface-border ${
            inputMode === 'config'
              ? 'bg-accent text-black'
              : 'bg-surface-sunken text-text-tertiary hover:text-text-secondary'
          } disabled:opacity-50`}
        >
          配置选择
        </button>
      </div>

      {/* Config Selector — only in config mode */}
      {inputMode === 'config' && (
        <>
          <div className="relative">
            <select
              value={selectedConfig}
              onChange={(e) => handleConfigChange(e.target.value)}
              disabled={isRunning || loading}
              className="appearance-none pl-3 pr-8 py-1.5 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary min-w-0 sm:min-w-[280px] w-full sm:w-auto focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent disabled:opacity-50 transition-colors cursor-pointer"
            >
              <option value="">Select a config...</option>
              {[...grouped.entries()].map(([category, entries]) => (
                <optgroup key={category} label={categoryLabel(category)}>
                  {entries.map((config) => (
                    <option
                      key={config.path}
                      value={config.path}
                      disabled={!config.available}
                    >
                      {config.label
                        ? `${config.label} — ${config.path.split('/').pop()?.replace('.yaml', '')}`
                        : config.path.split('/').pop()?.replace('.yaml', '')}
                      {!config.available && config.reason ? ` (${config.reason})` : ''}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-text-tertiary">
              {loading ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              )}
            </div>
          </div>

          {/* Start / New Trial / Retry — config mode only */}
          {canStart && configPath && !loading && (
            <button
              onClick={state === 'complete' || state === 'error' ? reset : handleStart}
              className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-accent text-black rounded-md text-sm font-display font-bold tracking-wide hover:bg-accent-light hover:shadow-accent/30 active:scale-[0.98] transform transition-colors shadow-lg shadow-accent/20"
            >
              {state === 'complete' ? (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  New Trial
                </>
              ) : state === 'error' ? (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Retry
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                  Start Trial
                </>
              )}
            </button>
          )}
        </>
      )}

      {/* Stop button — both modes */}
      {isRunning && (
        <button
          onClick={stopTrial}
          className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-red-600/80 text-white border border-red-500/30 rounded-md text-sm font-display font-medium hover:bg-red-600 transition-colors shadow-sm"
        >
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
          </svg>
          Stop
        </button>
      )}

      {/* Clear button after completion — config mode only */}
      {inputMode === 'config' && (state === 'complete' || state === 'error') && (
        <button
          onClick={reset}
          className="p-1.5 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
          title="Clear"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      )}

      {/* Error indicator */}
      {error && (
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-950/50 border border-red-800/30">
          <svg className="w-3.5 h-3.5 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-xs font-display text-red-400 max-w-[200px] truncate">{error}</span>
        </div>
      )}
    </div>
  );
}
