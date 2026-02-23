import { useState, useRef, useEffect } from 'react';

const MODELS = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' },
];

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'tool'; // 'tool' = agent tool-call activity row
  content: string;
  error?: boolean;
  // tool-specific fields
  toolName?: string;
  toolStatus?: 'running' | 'done';
}

interface SseEvent {
  type: 'text' | 'done' | 'error' | 'tool_call' | 'tool_result';
  text?: string;
  message?: string;
  tool?: string;
  input?: Record<string, unknown>;
}

const API_URL = 'http://localhost:3001';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [model, setModel] = useState('claude-opus-4-6');
  const [system, setSystem] = useState('You are a helpful assistant.');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Refs to track the current assistant bubble and last tool row
  // across multiple agentic loop iterations.
  const currentAssistantIdRef = useRef<string | null>(null);
  const lastToolIdRef = useRef<string | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.max(Math.min(el.scrollHeight, 200), 46)}px`;
  }, [input]);

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
    };

    const snapshot = [...messages, userMessage];
    setMessages(snapshot);
    setInput('');
    setIsStreaming(true);

    // Reset per-request refs
    currentAssistantIdRef.current = null;
    lastToolIdRef.current = null;

    abortRef.current = new AbortController();

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: snapshot.map((m) => ({ role: m.role === 'tool' ? 'user' : m.role, content: m.content }))
            .filter((m) => m.role === 'user' || m.role === 'assistant'), // only send real roles to API
          model,
          system: system.trim() || undefined,
        }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          try {
            const event = JSON.parse(raw) as SseEvent;

            if (event.type === 'text' && event.text) {
              // ── Text delta ───────────────────────────────────────────────
              // If no current assistant bubble exists (start of response, or
              // after a tool call), create a new one. Otherwise append.
              if (!currentAssistantIdRef.current) {
                const newId = crypto.randomUUID();
                currentAssistantIdRef.current = newId;
                setMessages((prev) => [
                  ...prev,
                  { id: newId, role: 'assistant', content: event.text! },
                ]);
              } else {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === currentAssistantIdRef.current
                      ? { ...m, content: m.content + event.text! }
                      : m,
                  ),
                );
              }
            } else if (event.type === 'tool_call' && event.tool) {
              // ── Tool call started ─────────────────────────────────────────
              // Reset assistant bubble so the next text chunk gets a fresh bubble
              // positioned AFTER this tool-activity row.
              currentAssistantIdRef.current = null;

              const toolId = crypto.randomUUID();
              lastToolIdRef.current = toolId;

              const url =
                (event.input as { url?: string })?.url ?? event.tool;

              setMessages((prev) => [
                ...prev,
                {
                  id: toolId,
                  role: 'tool',
                  content: url,
                  toolName: event.tool,
                  toolStatus: 'running',
                },
              ]);
            } else if (event.type === 'tool_result') {
              // ── Tool call finished ────────────────────────────────────────
              if (lastToolIdRef.current) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === lastToolIdRef.current
                      ? { ...m, toolStatus: 'done' }
                      : m,
                  ),
                );
              }
            } else if (event.type === 'error') {
              // Show error in a new assistant bubble
              const errId = crypto.randomUUID();
              setMessages((prev) => [
                ...prev,
                {
                  id: errId,
                  role: 'assistant',
                  content: `Error: ${event.message ?? 'Unknown error'}`,
                  error: true,
                },
              ]);
            }
          } catch {
            // skip malformed SSE line
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        // user stopped — keep partial content
      } else {
        const msg =
          err instanceof Error ? err.message : 'Failed to connect to server';
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: 'assistant', content: `Error: ${msg}`, error: true },
        ]);
      }
    } finally {
      // Ensure any stuck tool row shows as done
      setMessages((prev) =>
        prev.map((m) => (m.toolStatus === 'running' ? { ...m, toolStatus: 'done' } : m)),
      );
      setIsStreaming(false);
      currentAssistantIdRef.current = null;
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const stopStreaming = () => abortRef.current?.abort();

  const clearChat = () => {
    if (!isStreaming) setMessages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  const currentModelName = MODELS.find((m) => m.id === model)?.name ?? model;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <h1 className="logo">SudChat</h1>
        <div className="header-controls">
          <select
            className="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={isStreaming}
            aria-label="Select model"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
          <button
            className={`icon-btn${showSettings ? ' active' : ''}`}
            onClick={() => setShowSettings((s) => !s)}
            title="System prompt"
          >⚙</button>
          <button
            className="icon-btn"
            onClick={clearChat}
            disabled={isStreaming || messages.length === 0}
            title="Clear conversation"
          >✕</button>
        </div>
      </header>

      {/* ── Settings ── */}
      {showSettings && (
        <div className="settings-panel">
          <label htmlFor="system-prompt" className="settings-label">System Prompt</label>
          <textarea
            id="system-prompt"
            className="settings-textarea"
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            placeholder="Enter a system prompt…"
            rows={3}
          />
        </div>
      )}

      {/* ── Messages ── */}
      <main className="messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <p className="empty-title">SudChat</p>
            <p className="empty-sub">
              Start chatting with {currentModelName}
              <br />
              <span className="empty-hint">
                Try: <em>"Summarize https://en.wikipedia.org/wiki/Artificial_intelligence"</em>
              </span>
            </p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            // ── Tool activity row ──
            if (msg.role === 'tool') {
              return (
                <div key={msg.id} className={`tool-row ${msg.toolStatus ?? ''}`}>
                  {msg.toolStatus === 'running' ? (
                    <div className="tool-spinner" />
                  ) : (
                    <span className="tool-check">✓</span>
                  )}
                  <span className="tool-verb">
                    {msg.toolStatus === 'running' ? 'Fetching' : 'Fetched'}
                  </span>
                  <span className="tool-url" title={msg.content}>{msg.content}</span>
                </div>
              );
            }

            // ── Regular message bubble ──
            return (
              <div key={msg.id} className={`message ${msg.role}`}>
                <span className="message-label">
                  {msg.role === 'user' ? 'You' : 'Claude'}
                </span>
                <div className={`bubble${msg.error ? ' error' : ''}`}>
                  <span className="bubble-text">{msg.content}</span>
                  {msg.role === 'assistant' &&
                    isStreaming &&
                    idx === messages.length - 1 && (
                      <span className="cursor" aria-hidden="true" />
                    )}
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* ── Input bar ── */}
      <footer className="input-bar">
        <textarea
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message Claude… (Enter to send, Shift+Enter for newline)"
          rows={1}
          disabled={isStreaming}
          aria-label="Message input"
        />
        {isStreaming ? (
          <button className="btn stop-btn" onClick={stopStreaming}>◼ Stop</button>
        ) : (
          <button
            className="btn send-btn"
            onClick={() => void sendMessage()}
            disabled={!input.trim()}
          >Send ↵</button>
        )}
      </footer>
    </div>
  );
}
