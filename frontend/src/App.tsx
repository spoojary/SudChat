import { useState, useRef, useEffect } from 'react';

const MODELS = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' },
];

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  error?: boolean;
}

interface SseEvent {
  type: 'text' | 'done' | 'error';
  text?: string;
  message?: string;
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

  // Auto-scroll to bottom on new messages
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

    const assistantId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '' },
    ]);

    abortRef.current = new AbortController();

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: snapshot.map((m) => ({ role: m.role, content: m.content })),
          model,
          system: system.trim() || undefined,
        }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
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
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + event.text! }
                    : m,
                ),
              );
            } else if (event.type === 'error') {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: `Error: ${event.message ?? 'Unknown error'}`,
                        error: true,
                      }
                    : m,
                ),
              );
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
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `Error: ${msg}`, error: true }
              : m,
          ),
        );
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const stopStreaming = () => {
    abortRef.current?.abort();
  };

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
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>

          <button
            className={`icon-btn${showSettings ? ' active' : ''}`}
            onClick={() => setShowSettings((s) => !s)}
            title="System prompt"
            aria-label="Toggle system prompt settings"
          >
            ⚙
          </button>

          <button
            className="icon-btn"
            onClick={clearChat}
            disabled={isStreaming || messages.length === 0}
            title="Clear conversation"
            aria-label="Clear conversation"
          >
            ✕
          </button>
        </div>
      </header>

      {/* ── Settings panel ── */}
      {showSettings && (
        <div className="settings-panel">
          <label htmlFor="system-prompt" className="settings-label">
            System Prompt
          </label>
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
            <p className="empty-sub">Start chatting with {currentModelName}</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
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
          ))
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
          <button
            className="btn stop-btn"
            onClick={stopStreaming}
            aria-label="Stop generation"
          >
            ◼ Stop
          </button>
        ) : (
          <button
            className="btn send-btn"
            onClick={() => void sendMessage()}
            disabled={!input.trim()}
            aria-label="Send message"
          >
            Send ↵
          </button>
        )}
      </footer>
    </div>
  );
}
