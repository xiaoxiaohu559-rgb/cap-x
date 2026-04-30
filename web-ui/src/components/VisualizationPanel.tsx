import { useState, useEffect, useRef } from 'react';
import type { SessionState } from '../types/messages';

const DEFAULT_VISER_URL = '/viser-proxy/';

interface VisualizationPanelProps {
  trialState?: SessionState;
  videoUrl?: string | null;
}

export function VisualizationPanel({ trialState, videoUrl }: VisualizationPanelProps) {
  const [viserUrl, setViserUrl] = useState(DEFAULT_VISER_URL);
  const [isEditing, setIsEditing] = useState(false);
  const [tempUrl, setTempUrl] = useState(viserUrl);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [hasEverConnected, setHasEverConnected] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [viserAvailable, setViserAvailable] = useState<boolean | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const handleSaveUrl = () => {
    setViserUrl(tempUrl);
    setIsEditing(false);
    setConnectionStatus('connecting');
    setHasEverConnected(false);
    hasEverConnectedRef.current = false;
    setRetryCount(0);
    setViserAvailable(null);
  };

  const prevTrialState = useRef(trialState);
  useEffect(() => {
    if (trialState === 'running' && prevTrialState.current !== 'running') {
      wasConnectedRef.current = false;
      hasEverConnectedRef.current = false;
      setConnectionStatus('connecting');
      setHasEverConnected(false);
      setViserAvailable(null);
      setTimeout(() => setRetryCount(c => c + 1), 1500);
    }
    prevTrialState.current = trialState;
  }, [trialState]);

  const wasConnectedRef = useRef(false);
  const hasEverConnectedRef = useRef(false);
  const failCountRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const ctrl = new AbortController();
          const tid = setTimeout(() => ctrl.abort(), 3000);
          const resp = await fetch(viserUrl, { method: 'HEAD', signal: ctrl.signal });
          clearTimeout(tid);

          if (resp.ok) {
            failCountRef.current = 0;
            if (!wasConnectedRef.current) {
              wasConnectedRef.current = true;
              if (!hasEverConnectedRef.current) {
                hasEverConnectedRef.current = true;
                setRetryCount(c => c + 1);
              }
            }
            setConnectionStatus('connected');
            setHasEverConnected(true);
            setViserAvailable(true);
          } else {
            wasConnectedRef.current = false;
            failCountRef.current++;
            if (failCountRef.current >= 5) {
              setViserAvailable(false);
            }
            setConnectionStatus(prev => prev === 'connected' ? 'disconnected' : prev);
          }
        } catch {
          wasConnectedRef.current = false;
          failCountRef.current++;
          if (failCountRef.current >= 5) {
            setViserAvailable(false);
          }
          setConnectionStatus(prev =>
            prev === 'connected' ? 'disconnected' : prev === 'connecting' ? 'connecting' : 'disconnected'
          );
        }
        await new Promise(r => setTimeout(r, wasConnectedRef.current ? 5000 : 3000));
      }
    }

    poll();
    return () => { cancelled = true; };
  }, [viserUrl]);

  const handleIframeLoad = () => {
    setConnectionStatus('connected');
    setHasEverConnected(true);
  };

  const handleIframeError = () => {
    if (!hasEverConnected) {
      setConnectionStatus('connecting');
    }
  };

  const handleManualRefresh = () => {
    wasConnectedRef.current = false;
    hasEverConnectedRef.current = false;
    failCountRef.current = 0;
    setRetryCount(c => c + 1);
    setConnectionStatus('connecting');
    setHasEverConnected(false);
    setViserAvailable(null);
  };

  const showVideo = viserAvailable === false || (videoUrl && !hasEverConnected);

  const statusDotClass = connectionStatus === 'connected'
    ? 'bg-nv-green'
    : connectionStatus === 'disconnected'
    ? 'bg-text-tertiary'
    : 'bg-accent animate-pulse';

  const statusText = showVideo
    ? (videoUrl ? 'Video' : 'No 3D')
    : connectionStatus === 'connected'
    ? 'Connected'
    : connectionStatus === 'disconnected'
    ? 'Idle'
    : 'Connecting...';

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 px-5 py-3.5 bg-surface-raised border-b border-surface-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-accent/10 border border-accent/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
            </svg>
          </div>
          <span className="text-sm font-bold font-display tracking-wide uppercase text-text-primary">
            {showVideo ? 'Simulation Video' : '3D Visualization'}
          </span>

          <div className="flex items-center gap-1.5 ml-2">
            <div className={`w-2 h-2 rounded-full ${showVideo ? (videoUrl ? 'bg-nv-green' : 'bg-text-tertiary') : statusDotClass}`} />
            <span className={`text-xs font-display ${showVideo ? (videoUrl ? 'text-nv-green' : 'text-text-tertiary') : connectionStatus === 'connected' ? 'text-nv-green' : 'text-text-tertiary'}`}>
              {statusText}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!showVideo && (isEditing ? (
            <>
              <input
                type="text"
                value={tempUrl}
                onChange={(e) => setTempUrl(e.target.value)}
                className="px-3 py-1.5 text-xs bg-surface-sunken text-text-primary border border-surface-border rounded-md w-full sm:w-64 focus:ring-1 focus:ring-accent/40 focus:border-accent/40"
                onKeyDown={(e) => e.key === 'Enter' && handleSaveUrl()}
                autoFocus
              />
              <button
                onClick={handleSaveUrl}
                className="px-3 py-1.5 text-xs font-display bg-accent text-black rounded-md hover:bg-accent-light transition-colors"
              >
                Save
              </button>
              <button
                onClick={() => {
                  setTempUrl(viserUrl);
                  setIsEditing(false);
                }}
                className="px-3 py-1.5 text-xs bg-surface-overlay text-text-secondary rounded-md hover:bg-surface-border transition-colors"
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              {!showVideo && (
                <span className="text-xs text-text-tertiary max-w-[200px] truncate" title={viserUrl}>{viserUrl}</span>
              )}
              <button
                onClick={handleManualRefresh}
                className="p-2 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
                title="Refresh connection"
                aria-label="Refresh connection"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
              {!showVideo && (
                <>
                  <button
                    onClick={() => setIsEditing(true)}
                    className="p-2 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
                    title="Edit URL"
                    aria-label="Edit URL"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>
                  <button
                    onClick={() => window.open(viserUrl, '_blank')}
                    className="p-2 text-text-tertiary hover:text-accent hover:bg-surface-overlay rounded-md transition-colors"
                    title="Open in new tab"
                    aria-label="Open in new tab"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </button>
                </>
              )}
            </>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 relative bg-surface-sunken">
        {showVideo ? (
          // Video playback mode (macOS / no Viser)
          videoUrl ? (
            <div className="absolute inset-0 flex items-center justify-center p-4">
              <video
                src={videoUrl}
                controls
                autoPlay
                loop
                className="max-w-full max-h-full rounded-lg shadow-lg"
                style={{ background: '#000' }}
              />
            </div>
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              {trialState === 'running' ? (
                <>
                  <div className="flex items-center gap-3 text-text-secondary mb-2">
                    <div className="w-6 h-6 border-2 border-accent/20 border-t-accent rounded-full animate-spin" />
                    <span className="font-display text-sm">Running simulation...</span>
                  </div>
                  <span className="text-xs text-text-tertiary">Video will appear when the trial completes</span>
                </>
              ) : (
                <>
                  <svg className="w-12 h-12 text-text-tertiary mb-3 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <span className="font-display text-sm text-text-tertiary">Start a trial to generate simulation video</span>
                </>
              )}
            </div>
          )
        ) : (
          // Viser 3D mode
          <>
            {!hasEverConnected && connectionStatus === 'connecting' && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-surface-sunken">
                <div className="flex items-center gap-3 text-text-secondary mb-2">
                  <div className="w-6 h-6 border-2 border-accent/20 border-t-accent rounded-full animate-spin" />
                  <span className="font-display text-sm">Waiting for Viser server...</span>
                </div>
                <span className="text-xs text-text-tertiary">Start a trial to initialize the 3D view</span>
              </div>
            )}
            <iframe
              ref={iframeRef}
              key={`viser-${retryCount}`}
              src={viserUrl}
              title="Viser 3D Visualization"
              className="absolute inset-0 w-full h-full border-0"
              allow="autoplay; fullscreen"
              onLoad={handleIframeLoad}
              onError={handleIframeError}
            />
          </>
        )}
      </div>
    </div>
  );
}
