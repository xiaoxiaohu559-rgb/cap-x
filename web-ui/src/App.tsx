import { useState, useEffect, useRef, useCallback } from 'react';
import { useTrialState } from './hooks/useTrialState';
import { ConfigStartControl } from './components/ConfigStartControl';
import type { InputMode } from './components/ConfigStartControl';
import { ChatPanel } from './components/ChatPanel';
import { VisualizationPanel } from './components/VisualizationPanel';

const FALLBACK_CONFIG = 'env_configs/cube_stack/franka_robosuite_cube_stack.yaml';

const ROBOT_ARMS = [
  { value: 'franka_panda', label: 'Franka Emika Panda', labelZh: 'Franka Panda 机械臂', available: true },
  { value: 'ur5e', label: 'UR5e', labelZh: 'UR5e 协作机械臂', available: false },
  { value: 'kinova_gen3', label: 'Kinova Gen3', labelZh: 'Kinova Gen3', available: false },
  { value: 'r1pro', label: 'R1Pro Humanoid', labelZh: 'R1Pro 人形机器人', available: false },
];

const GRIPPERS = [
  { value: 'parallel_jaw', label: 'Parallel Jaw Gripper', labelZh: '平行夹爪', available: true },
  { value: 'robotiq_2f85', label: 'Robotiq 2F-85', labelZh: 'Robotiq 2F-85', available: false },
  { value: 'dexterous_hand', label: 'Dexterous Hand', labelZh: '灵巧手', available: false },
];

function App() {
  const trial = useTrialState();
  const [model, setModel] = useState('anthropic/claude-sonnet-4-6');
  const [serverUrl, setServerUrl] = useState('http://127.0.0.1:8110/chat/completions');
  const [temperature, setTemperature] = useState(1.0);
  const [awaitUserInput, setAwaitUserInput] = useState(true);
  const [executionTimeout, setExecutionTimeout] = useState(180);
  const [hasCheckedSession, setHasCheckedSession] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [userInstruction, setUserInstruction] = useState('');
  const [robotArm, setRobotArm] = useState('franka_panda');
  const [gripper, setGripper] = useState('parallel_jaw');
  const [envDescription, setEnvDescription] = useState('');
  const [codeHint, setCodeHint] = useState('');

  // Auto-evolve loop state
  const [autoEvolveRemaining, setAutoEvolveRemaining] = useState(0);
  const [autoEvolveTotal, setAutoEvolveTotal] = useState(0);
  const autoEvolveRef = useRef(0); // ref to avoid stale closures

  // Resizable panel
  const [splitPercent, setSplitPercent] = useState(60);
  const draggingRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close settings popover when clicking outside
  const settingsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setShowSettings(false);
      }
    }
    if (showSettings) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showSettings]);

  // Check for active session on mount, then load default config
  useEffect(() => {
    const init = async () => {
      const activeSession = await trial.checkActiveSession();
      if (activeSession?.session_id) {
        trial.reconnectToSession(
          activeSession.session_id,
          activeSession.config_path || FALLBACK_CONFIG
        );
        if (activeSession.config_path) {
          trial.loadConfig(activeSession.config_path);
        }
        setHasCheckedSession(true);
        return;
      }

      try {
        const resp = await fetch('/api/default-config');
        const data = await resp.json();
        const configPath = data.config_path || FALLBACK_CONFIG;

        await trial.loadConfig(configPath);
      } catch {
        trial.loadConfig(FALLBACK_CONFIG);
      }
      setHasCheckedSession(true);
    };
    init();
  }, []);

  const isRunning = trial.state === 'running' || trial.state === 'awaiting_user_input';

  // Build combined user_instruction from env setup fields
  const buildUserInstruction = useCallback((): string | undefined => {
    if (inputMode !== 'text') return undefined;
    const armInfo = ROBOT_ARMS.find(r => r.value === robotArm);
    const gripperInfo = GRIPPERS.find(g => g.value === gripper);
    const parts: string[] = [];
    parts.push(`[环境配置] 机械臂: ${armInfo?.label || robotArm}, 末端执行器: ${gripperInfo?.labelZh || gripper}`);
    if (envDescription.trim()) {
      parts.push(`[环境描述] ${envDescription.trim()}`);
    }
    if (userInstruction.trim()) {
      parts.push(`[任务指令] ${userInstruction.trim()}`);
    }
    const combined = parts.join('\n');
    return combined.length > 0 ? combined : undefined;
  }, [inputMode, robotArm, gripper, envDescription, userInstruction]);

  // Wrap startTrial to inject combined prompt
  const handleStartTrial: typeof trial.startTrial = useCallback(async (request) => {
    const instruction = buildUserInstruction();
    console.log('[CaP-X] startTrial user_instruction:', instruction);
    const fullRequest = {
      ...request,
      ...(instruction ? { user_instruction: instruction } : {}),
      ...(codeHint.trim() ? { code_hint: codeHint.trim() } : {}),
    };
    console.log('[CaP-X] full request:', JSON.stringify(fullRequest, null, 2));
    return trial.startTrial(fullRequest);
  }, [trial.startTrial, buildUserInstruction, codeHint]);

  // Retry: reset + start a new trial with the error context as code_hint
  const startNewTrial = useCallback((hint: string, forceNoAwait = false) => {
    // Extract last error from messages
    let lastErr = '';
    for (let i = trial.messages.length - 1; i >= 0; i--) {
      const m = trial.messages[i];
      if (m.type === 'code_execution' && !m.isExecuting && m.stderr) {
        lastErr = m.stderr;
        break;
      }
    }

    const parts: string[] = [];
    if (lastErr) parts.push(`上一轮执行报错:\n${lastErr}`);
    if (hint) parts.push(hint);
    const combinedHint = parts.join('\n\n');

    const cfgPath = trial.configPath;
    if (!cfgPath) return;

    const instruction = buildUserInstruction();

    trial.reset();
    setTimeout(() => {
      const request = {
        config_path: cfgPath,
        model,
        server_url: serverUrl,
        temperature,
        await_user_input_each_turn: forceNoAwait ? false : awaitUserInput,
        execution_timeout: executionTimeout,
        ...(instruction ? { user_instruction: instruction } : {}),
        code_hint: combinedHint || undefined,
      };
      trial.startTrial(request);
    }, 100);
  }, [trial, buildUserInstruction, model, serverUrl, temperature, awaitUserInput, executionTimeout]);

  const handleRetry = useCallback((hint: string) => {
    startNewTrial(hint);
  }, [startNewTrial]);

  // Auto-evolve: start N rounds that run automatically
  const handleAutoEvolve = useCallback((count: number, hint: string) => {
    const remaining = count - 1; // first round starts now
    setAutoEvolveTotal(count);
    setAutoEvolveRemaining(remaining);
    autoEvolveRef.current = remaining;
    startNewTrial(hint, true);
  }, [startNewTrial]);

  const stopAutoEvolve = useCallback(() => {
    setAutoEvolveRemaining(0);
    autoEvolveRef.current = 0;
  }, []);

  // Auto-continue effect: when a trial completes and autoEvolveRemaining > 0, start next
  useEffect(() => {
    if (trial.state === 'complete' && autoEvolveRef.current > 0) {
      const next = autoEvolveRef.current - 1;
      setAutoEvolveRemaining(next);
      autoEvolveRef.current = next;
      const timer = setTimeout(() => {
        startNewTrial('', true);
      }, 2000);
      return () => clearTimeout(timer);
    }
    if (trial.state === 'error' && autoEvolveRef.current > 0) {
      // Stop on error
      setAutoEvolveRemaining(0);
      autoEvolveRef.current = 0;
    }
  }, [trial.state, startNewTrial]);

  // Drag handler for resizable split
  const [isDragging, setIsDragging] = useState(false);
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    setIsDragging(true);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current || !containerRef.current) return;
      e.preventDefault();
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setSplitPercent(Math.min(70, Math.max(30, pct)));
    };

    const handleMouseUp = () => {
      draggingRef.current = false;
      setIsDragging(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, []);

  // Readable model name for the header badge
  const modelDisplayName = (() => {
    const slug = model.split('/').pop() || model;
    const nameMap: Record<string, string> = {
      'claude-sonnet-4-6': 'Sonnet 4.6',
      'claude-opus-4-6': 'Opus 4.6',
      'deepseek-chat': 'DeepSeek V3',
      'deepseek-reasoner': 'DeepSeek R1',
    };
    return nameMap[slug] || slug;
  })();

  if (!hasCheckedSession) {
    return (
      <div className="h-full flex items-center justify-center bg-surface">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
          <span className="text-sm font-medium text-text-secondary tracking-wide">Initializing</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-surface">
      {/* Header */}
      <header className="flex-shrink-0 bg-surface-raised border-b border-surface-border header-glow">
        <div className="flex items-center h-14 px-6 gap-3">
          {/* Logo */}
          <div className="flex items-center gap-2.5 flex-shrink-0">
            <img src="/capx_logo.svg" alt="CaP-X" className="w-7 h-7" />
            <h1 className="text-base font-bold font-display text-text-primary tracking-widest uppercase">CaP-X</h1>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-surface-border flex-shrink-0" />

          {/* Config + Start Control (center) — min-w-0 prevents overflow */}
          <div className="flex-1 min-w-0 flex justify-center">
            <ConfigStartControl
              state={trial.state}
              configPath={trial.configPath}
              error={trial.error}
              loadConfig={trial.loadConfig}
              startTrial={handleStartTrial}
              stopTrial={trial.stopTrial}
              reset={trial.reset}
              model={model}
              serverUrl={serverUrl}
              temperature={temperature}
              awaitUserInput={awaitUserInput}
              inputMode={inputMode}
              onInputModeChange={setInputMode}
            />
          </div>

          {/* Right side: model badge + status + settings */}
          <div className="flex items-center gap-2.5 flex-shrink-0">
            {/* Model badge (click opens settings) */}
            <button
              onClick={() => setShowSettings(true)}
              className="px-2.5 py-1 rounded-md bg-surface-sunken border border-surface-border text-xs font-display text-text-secondary hover:text-text-primary hover:border-accent/30 transition-colors"
              title="Click to change model"
            >
              {modelDisplayName}
            </button>

            {/* Status indicator */}
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-surface-overlay border border-surface-border">
              <div className={`w-1.5 h-1.5 rounded-full ${
                trial.isConnected
                  ? isRunning
                    ? 'bg-accent animate-pulse'
                    : 'bg-nv-green'
                  : 'bg-text-tertiary'
              }`} />
              <span className="text-xs text-text-secondary font-display font-medium tracking-wide">
                {isRunning ? 'Running' : trial.isConnected ? 'Ready' : 'Offline'}
              </span>
              {isRunning && <div className="w-10 h-0.5 rounded-full bg-accent/30 overflow-hidden"><div className="h-full w-1/2 bg-accent rounded-full animate-[slideIn_1s_ease-in-out_infinite_alternate]" /></div>}
            </div>

            {/* Settings Gear */}
            <div className="relative" ref={settingsRef}>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-md transition-all duration-150 ${
                showSettings
                  ? 'bg-surface-overlay text-accent border border-accent/20'
                  : 'text-text-tertiary hover:text-text-primary hover:bg-surface-overlay border border-transparent'
              }`}
              title="Settings"
              aria-label="Settings"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>

            {/* Settings Popover */}
            {showSettings && (
              <div className="absolute right-0 top-full mt-2 w-[calc(100vw-2rem)] sm:w-96 bg-surface-raised rounded-lg shadow-2xl border border-surface-border-light z-50 animate-scale-in">
                <div className="px-5 py-3.5 border-b border-surface-border flex items-center justify-between">
                  <h3 className="text-sm font-bold font-display text-text-primary tracking-wide">Settings</h3>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="p-2 text-text-tertiary hover:text-text-primary rounded transition-colors"
                    aria-label="Close settings"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="p-5 space-y-5">
                  {/* Model Selector */}
                  <div>
                    <label htmlFor="settings-model" className="block text-xs font-display font-medium text-text-secondary mb-1.5 tracking-wide uppercase">Model</label>
                    <select
                      id="settings-model"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      disabled={isRunning}
                      className="w-full appearance-none px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm font-display text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all cursor-pointer"
                    >
                      <optgroup label="Anthropic">
                        <option value="anthropic/claude-sonnet-4-6">Claude Sonnet 4.6</option>
                        <option value="anthropic/claude-opus-4-6">Claude Opus 4.6</option>
                        <option value="anthropic/claude-haiku-4-5" disabled>Claude Haiku 4.5</option>
                        <option value="anthropic/claude-sonnet-4" disabled>Claude Sonnet 4</option>
                        <option value="anthropic/claude-opus-4-5" disabled>Claude Opus 4.5</option>
                      </optgroup>
                      <optgroup label="DeepSeek">
                        <option value="deepseek/deepseek-chat">DeepSeek V3 (Latest)</option>
                        <option value="deepseek/deepseek-reasoner">DeepSeek R1 (Reasoner)</option>
                      </optgroup>
                      <optgroup label="Google (unavailable)">
                        <option value="google/gemini-3.1-pro-preview" disabled>Gemini 3.1 Pro Preview</option>
                        <option value="google/gemini-3.1-pro" disabled>Gemini 3.1 Pro</option>
                        <option value="google/gemini-2.5-flash-lite" disabled>Gemini 2.5 Flash Lite</option>
                      </optgroup>
                      <optgroup label="OpenAI (unavailable)">
                        <option value="openai/gpt-5.2" disabled>GPT 5.2</option>
                        <option value="openai/gpt-5.1" disabled>GPT 5.1</option>
                        <option value="openai/o4-mini" disabled>O4 Mini</option>
                        <option value="openai/o1" disabled>O1</option>
                      </optgroup>
                      <optgroup label="Open Source (unavailable)">
                        <option value="qwen/qwen-235b-a22b" disabled>Qwen 235B</option>
                        <option value="moonshotai/kimi-k2-instruct" disabled>Kimi K2</option>
                      </optgroup>
                    </select>
                  </div>

                  {/* Server URL */}
                  <div>
                    <label htmlFor="settings-server-url" className="block text-xs font-display font-medium text-text-secondary mb-1.5 tracking-wide uppercase">Server URL</label>
                    <input
                      id="settings-server-url"
                      type="text"
                      value={serverUrl}
                      onChange={(e) => setServerUrl(e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary font-mono placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all"
                    />
                  </div>

                  {/* Temperature */}
                  <div>
                    <label htmlFor="settings-temperature" className="block text-xs font-display font-medium text-text-secondary mb-1.5 tracking-wide uppercase">Temperature</label>
                    <input
                      id="settings-temperature"
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
                      disabled={isRunning}
                      min="0"
                      max="2"
                      step="0.1"
                      className="w-24 px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary font-mono focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all"
                    />
                  </div>

                  {/* Execution Timeout */}
                  <div>
                    <label htmlFor="settings-timeout" className="block text-xs font-medium font-display text-text-secondary mb-1.5 tracking-wide uppercase">Execution Timeout (s)</label>
                    <input
                      id="settings-timeout"
                      type="number"
                      value={executionTimeout}
                      onChange={(e) => setExecutionTimeout(parseInt(e.target.value) || 180)}
                      disabled={isRunning}
                      min="30"
                      max="600"
                      step="30"
                      className="w-24 px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary font-mono focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all"
                    />
                  </div>

                  {/* Pause each turn */}
                  <label className="flex items-center gap-2.5 cursor-pointer group py-1">
                    <div className="relative">
                      <input
                        type="checkbox"
                        checked={awaitUserInput}
                        onChange={(e) => {
                          const newValue = e.target.checked;
                          setAwaitUserInput(newValue);
                          if (isRunning) {
                            trial.updateSettings({ await_user_input_each_turn: newValue });
                          }
                        }}
                        className="sr-only peer"
                      />
                      <div className="w-8 h-4.5 bg-surface-border rounded-full peer-checked:bg-accent/80 transition-colors" />
                      <div className="absolute top-0.5 left-0.5 w-3.5 h-3.5 bg-text-secondary rounded-full peer-checked:translate-x-3.5 peer-checked:bg-white transition-all" />
                    </div>
                    <span className="text-sm font-display text-text-secondary group-hover:text-text-primary transition-colors">
                      Pause each turn for feedback
                    </span>
                  </label>
                </div>
              </div>
            )}
          </div>
          </div>
        </div>
      </header>

      {/* Environment Setup + Instruction Panel — visible when text mode is active */}
      {inputMode === 'text' && (
        <div className="flex-shrink-0 bg-surface-raised border-b border-surface-border px-6 py-4 space-y-4">
          {/* Environment Config Section */}
          <div>
            <h4 className="text-xs font-display font-bold text-text-secondary tracking-wide uppercase mb-2.5">
              环境配置 / Environment Setup
            </h4>
            <div className="flex flex-wrap items-center gap-3 mb-3">
              {/* Robot Arm */}
              <div className="flex items-center gap-2">
                <label className="text-xs font-display text-text-tertiary whitespace-nowrap">机械臂</label>
                <select
                  value={robotArm}
                  onChange={(e) => setRobotArm(e.target.value)}
                  disabled={isRunning}
                  className="appearance-none px-2.5 py-1.5 bg-surface-sunken border border-surface-border rounded-md text-xs font-display text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40 disabled:opacity-50 transition-colors cursor-pointer"
                >
                  {ROBOT_ARMS.map(arm => (
                    <option key={arm.value} value={arm.value} disabled={!arm.available}>
                      {arm.labelZh}{!arm.available ? ' (暂不可用)' : ''}
                    </option>
                  ))}
                </select>
              </div>
              {/* Gripper */}
              <div className="flex items-center gap-2">
                <label className="text-xs font-display text-text-tertiary whitespace-nowrap">末端执行器</label>
                <select
                  value={gripper}
                  onChange={(e) => setGripper(e.target.value)}
                  disabled={isRunning}
                  className="appearance-none px-2.5 py-1.5 bg-surface-sunken border border-surface-border rounded-md text-xs font-display text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40 disabled:opacity-50 transition-colors cursor-pointer"
                >
                  {GRIPPERS.map(g => (
                    <option key={g.value} value={g.value} disabled={!g.available}>
                      {g.labelZh}{!g.available ? ' (暂不可用)' : ''}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {/* Environment Description */}
            <textarea
              value={envDescription}
              onChange={(e) => setEnvDescription(e.target.value)}
              disabled={isRunning}
              placeholder="描述仿真环境中的物体和场景，例如：桌上有五个不同颜色的积木块 / Describe objects and scene, e.g.: 5 colored blocks on the table"
              rows={2}
              className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary placeholder-text-tertiary font-sans resize-none focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent disabled:opacity-50 transition-all"
            />
          </div>

          {/* Divider */}
          <div className="border-t border-surface-border" />

          {/* Task Instruction Section */}
          <div>
            <h4 className="text-xs font-display font-bold text-text-secondary tracking-wide uppercase mb-2.5">
              任务指令 / Task Instruction
            </h4>
            <div className="flex gap-3 items-end">
              <div className="flex-1 space-y-2">
                <textarea
                  value={userInstruction}
                  onChange={(e) => setUserInstruction(e.target.value)}
                  disabled={isRunning}
                  placeholder="输入具体操作指令，例如：把红色方块堆叠到蓝色方块上面 / Enter task instructions, e.g.: Stack the red block on top of the blue block"
                  rows={2}
                  className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary placeholder-text-tertiary font-sans resize-y focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent disabled:opacity-50 transition-all"
                />
                <textarea
                  value={codeHint}
                  onChange={(e) => setCodeHint(e.target.value)}
                  disabled={isRunning}
                  placeholder="额外约束（可选）：限制或引导代码生成，例如：只操作环境中已有的物体，不要生成额外物体 / Additional constraints (optional): guide code generation"
                  rows={1}
                  className="w-full px-3 py-1.5 bg-surface-sunken border border-dashed border-surface-border rounded-md text-xs text-text-secondary placeholder-text-tertiary font-sans resize-y focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent disabled:opacity-50 transition-all"
                />
              </div>
              {/* Start / New Trial / Stop — text mode */}
              <div className="flex flex-col gap-2 flex-shrink-0 pb-0.5">
                {!isRunning && (trial.state === 'idle' || trial.state === 'complete' || trial.state === 'error') && (
                  <button
                    onClick={() => {
                      if (trial.state === 'complete' || trial.state === 'error') {
                        trial.reset();
                      } else if (trial.configPath) {
                        handleStartTrial({
                          config_path: trial.configPath,
                          model,
                          server_url: serverUrl,
                          temperature,
                          await_user_input_each_turn: awaitUserInput,
                          execution_timeout: executionTimeout,
                        });
                      }
                    }}
                    disabled={!(envDescription.trim() || userInstruction.trim()) || !trial.configPath}
                    className="inline-flex items-center gap-1.5 px-5 py-2.5 bg-accent text-black rounded-md text-sm font-display font-bold tracking-wide hover:bg-accent-light hover:shadow-accent/30 active:scale-[0.98] transform disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-lg shadow-accent/20 whitespace-nowrap"
                  >
                    {trial.state === 'complete' || trial.state === 'error' ? (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        New Trial
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
                {isRunning && (
                  <button
                    onClick={trial.stopTrial}
                    className="inline-flex items-center gap-1.5 px-5 py-2.5 bg-red-600/80 text-white border border-red-500/30 rounded-md text-sm font-display font-medium hover:bg-red-600 transition-colors shadow-sm whitespace-nowrap"
                  >
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                    </svg>
                    Stop
                  </button>
                )}
              </div>
            </div>
            <p className="mt-1.5 text-xs text-text-tertiary font-display">
              支持中英文。环境描述 + 任务指令将组合后发送给大模型生成代码。
            </p>
          </div>
        </div>
      )}

      {/* Main Content — Resizable Split */}
      <main ref={containerRef} className="flex-1 flex overflow-hidden relative">
        {/* Overlay to capture mouse during drag */}
        {isDragging && <div className="absolute inset-0 z-20" />}

        {/* Left Panel - Chat */}
        <div
          className="flex flex-col bg-surface overflow-hidden"
          style={{ width: `${splitPercent}%` }}
        >
          <ChatPanel
            messages={trial.messages}
            state={trial.state}
            onSendMessage={trial.injectPrompt}
            onResume={trial.resumeTrial}
            onRetry={handleRetry}
            onAutoEvolve={handleAutoEvolve}
            onStopAutoEvolve={stopAutoEvolve}
            autoEvolveRemaining={autoEvolveRemaining}
            autoEvolveTotal={autoEvolveTotal}
            taskPrompt={trial.taskPrompt}
          />
        </div>

        {/* Draggable Divider */}
        <div
          className={`flex-shrink-0 w-0.5 cursor-col-resize relative group z-30 transition-colors duration-150 ${
            isDragging ? 'bg-accent gold-rule' : 'bg-surface-border hover:bg-accent/50'
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Wider invisible hit target */}
          <div className="absolute inset-y-0 -left-2 -right-2" />
          {/* Visual grip indicator */}
          <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-8 rounded-full flex items-center justify-center transition-all ${
            isDragging ? 'bg-accent/20' : 'bg-transparent group-hover:bg-surface-overlay'
          }`}>
            <div className="flex flex-col gap-0.5">
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
            </div>
          </div>
        </div>

        {/* Right Panel - Visualization */}
        <div
          className="flex flex-col bg-surface-raised overflow-hidden"
          style={{ width: `${100 - splitPercent}%` }}
        >
          <VisualizationPanel trialState={trial.state} videoUrl={trial.videoUrl} />
        </div>
      </main>
    </div>
  );
}

export default App;
