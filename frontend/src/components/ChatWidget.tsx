import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatWidgetProps {
  apiBase?: string; // optional override
}

export const ChatWidget: React.FC<ChatWidgetProps> = ({ apiBase = 'http://localhost:5000' }) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hi! I can explain any voice feature (e.g., "AudSpec Rfilt 6") or your risk results. Ask me anything.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, open]);

  const parseJsonSafe = async (res: Response) => {
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      return res.json();
    }
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return { error: text || 'Non-JSON response' };
    }
  };

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    setError(null);
  const newMessages: ChatMessage[] = [...messages, { role: 'user' as const, content: trimmed }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, conversation: newMessages })
      });
  const data = await parseJsonSafe(res);
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Chat error');
      }
  setMessages([...newMessages, { role: 'assistant', content: String(data.reply || '') }]);
    } catch (e: any) {
      setError(e.message);
  setMessages([...newMessages, { role: 'assistant', content: 'Sorry, I could not respond. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  };

  const explainFeature = async (raw: string) => {
    const feature = raw.trim();
    if (!feature) return;
    setError(null);
  const newMessages: ChatMessage[] = [...messages, { role: 'user', content: `Explain feature: ${feature}` }];
    setMessages(newMessages);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feature_name: feature })
      });
  const data = await parseJsonSafe(res);
      if (!res.ok || data.error) throw new Error(data.error || 'Feature explanation error');
  setMessages([...newMessages, { role: 'assistant', content: String(data.reply || '') }]);
    } catch (e: any) {
      setError(e.message);
      setMessages([...newMessages, { role: 'assistant', content: 'Could not explain that feature right now.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 z-50 rounded-full shadow-lg bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] text-white p-4 hover:shadow-xl transition"
        aria-label={open ? 'Close chat' : 'Open chat'}
      >
        {open ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
      </button>

      {open && (
        <div className="fixed bottom-24 right-6 z-50 w-80 bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col overflow-hidden">
          <div className="px-4 py-3 bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] text-white flex items-center justify-between">
            <span className="font-semibold text-sm">VoiceCare Assistant</span>
            <button onClick={() => setOpen(false)} aria-label="Close" className="p-1 hover:bg-white/20 rounded">
              <X className="w-4 h-4" />
            </button>
          </div>
          <div ref={scrollRef} className="flex-1 px-3 py-3 space-y-3 overflow-y-auto bg-gray-50">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`text-sm whitespace-pre-wrap rounded-lg px-3 py-2 max-w-[85%] ${
                  m.role === 'user' ? 'ml-auto bg-[#2E7D32] text-white' : 'bg-white text-[#263238] shadow'
                }`}
              >
                {m.content}
              </div>
            ))}
            {loading && <div className="text-xs text-gray-500 animate-pulse">Assistant typing…</div>}
            {error && <div className="text-xs text-red-600">{error}</div>}
          </div>
          <div className="p-3 border-t bg-white space-y-2">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask a question…"
                className="flex-1 text-sm px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-[#2E7D32]"
                disabled={loading}
              />
              <button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="bg-[#2E7D32] disabled:opacity-50 text-white px-3 py-2 rounded-md flex items-center gap-1 text-sm hover:bg-[#256628]"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {['Jitter', 'Shimmer', 'HNR', 'MFCC 5', 'Pitch'].map(f => (
                <button
                  key={f}
                  onClick={() => explainFeature(f)}
                  className="text-xs px-2 py-1 rounded-full bg-gray-200 hover:bg-gray-300"
                  disabled={loading}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
};