// Rider chat: sends the message plus the last two turns to POST /chat and renders the answer.
// Auto replies are bubbles; escalations are a green card with the rule name and the ticket reference.
import { useState } from "react";
import { API } from "./App.jsx";

export default function Help() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function send() {
    const message = text.trim();
    if (!message || busy) return;
    const history = messages.slice(-2).map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setText("");
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history, conversation_id: conversationId }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const data = await res.json();
      setConversationId(data.conversation_id);
      setMessages((prev) => [...prev, { role: "agent", content: data.reply, meta: data }]);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-3 min-h-[50vh]">
        {messages.length === 0 && (
          <p className="text-neutral-500 text-sm">Tell us what happened with your trip, order, or account.</p>
        )}
        {messages.map((m, i) => (
          <Bubble key={i} m={m} />
        ))}
        {busy && <p className="text-neutral-400 text-sm">Thinking…</p>}
        {error && <p className="text-red-600 text-sm">Request failed: {error}</p>}
      </div>
      <div className="flex gap-2 items-end">
        <textarea
          className="flex-1 border border-neutral-300 rounded-lg p-3 text-sm bg-white resize-none"
          rows={2}
          placeholder="Type your message"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button
          className="bg-black text-white rounded-lg px-4 py-3 text-sm font-medium disabled:opacity-40"
          onClick={send}
          disabled={busy || !text.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

function Bubble({ m }) {
  if (m.role === "user") {
    return <div className="self-end bg-black text-white rounded-2xl px-4 py-2 max-w-[80%] text-sm">{m.content}</div>;
  }
  if (m.meta?.action === "escalate") {
    return (
      <div className="self-start bg-green-50 border border-green-300 rounded-2xl px-4 py-3 max-w-[85%] text-sm">
        <div className="text-green-800 font-medium mb-1">Passed to a specialist · reason: {m.meta.reason}</div>
        <div className="text-green-900">{m.content}</div>
        <div className="text-green-700 text-xs mt-1">intent: {m.meta.intent} · urgency: {m.meta.urgency}</div>
      </div>
    );
  }
  return (
    <div className="self-start bg-white border border-neutral-200 rounded-2xl px-4 py-2 max-w-[80%] text-sm">
      <div>{m.content}</div>
      {m.meta && (
        <div className="text-neutral-400 text-xs mt-1">
          intent: {m.meta.intent} ({m.meta.confidence.toFixed(2)}) · {m.meta.latency_ms} ms
        </div>
      )}
    </div>
  );
}
