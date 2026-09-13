// Trace tab: the ticket list from GET /traces on the left, the selected ticket's trace (GET /traces/{id}) as a
// vertical stepper on the right, an eval-metrics tile from GET /metrics, and a cache-only Replay button.
import { useEffect, useState } from "react";
import { API } from "./App.jsx";
import { Badge, Baselines, Step, StepBody, Totals, WhyNot } from "./TraceSteps.jsx";

export default function Trace() {
  const [rows, setRows] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const [replay, setReplay] = useState(null); // null | "busy" | message string

  async function loadList(selectNewest = false) {
    try {
      const res = await fetch(`${API}/traces?limit=50`);
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      setRows(data);
      setError(null);
      if (data.length && (selectNewest || selected === null)) setSelected(data[0].id);
    } catch (err) {
      setError(String(err));
    }
  }

  useEffect(() => {
    loadList();
  }, []);

  useEffect(() => {
    if (selected === null) return;
    setDetail(null);
    fetch(`${API}/traces/${selected}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setDetail)
      .catch((err) => setError(String(err)));
  }, [selected]);

  async function replayCached() {
    const req = detail.trace.request;
    setReplay("busy");
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Cache-Only": "1" },
        body: JSON.stringify({ message: req.message, history: req.history }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      setReplay(null);
      await loadList(true); // the replay is the newest ticket; show its trace
    } catch (err) {
      setReplay(`Replay failed: ${err.message}`);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-base font-semibold">Trace</h2>
          <p className="text-xs text-neutral-500">One row per ticket, newest first. Click a row to see every step.</p>
        </div>
        <MetricsTile />
      </div>
      {error && <p className="text-red-600 text-sm">Could not load traces: {error}</p>}
      <div className="grid gap-4 md:grid-cols-[18rem_1fr]">
        <ul className="bg-white border border-neutral-200 rounded-lg divide-y divide-neutral-100 max-h-[75vh] overflow-y-auto">
          {rows.map((r) => (
            <li key={r.id}>
              <button
                className={`w-full text-left p-3 hover:bg-neutral-50 ${r.id === selected ? "bg-neutral-100" : ""}`}
                onClick={() => setSelected(r.id)}
              >
                <div className="flex items-center gap-2 text-xs text-neutral-500">
                  <span>#{r.id}</span>
                  <span>{r.timestamp}</span>
                  {!r.has_trace && <span className="text-amber-600">no trace</span>}
                </div>
                <div className="text-sm truncate mt-0.5">{r.message}</div>
                <div className="flex gap-1 mt-1">
                  <Badge kind="intent">{r.intent}</Badge>
                  <Badge kind={r.action}>{r.action === "escalate" ? `escalate · ${r.reason}` : "auto"}</Badge>
                </div>
              </button>
            </li>
          ))}
          {rows.length === 0 && !error && <li className="p-4 text-neutral-400 text-sm">No tickets yet. Send a message on the Help tab.</li>}
        </ul>
        <div className="min-w-0">
          {selected !== null && detail === null && <p className="text-neutral-400 text-sm">Loading…</p>}
          {detail && !detail.trace && (
            <p className="text-sm text-neutral-500">Ticket #{detail.id} was written before traces existed. Nothing to show.</p>
          )}
          {detail?.trace && <TraceView detail={detail} onReplay={replayCached} replay={replay} />}
        </div>
      </div>
    </div>
  );
}

function TraceView({ detail, onReplay, replay }) {
  const t = detail.trace;
  const steps = [
    ["request", "Request", null],
    ["enrich", "Enrich", t.enrich],
    ["rules", "Rules", t.rules],
    ["retrieve", "Retrieve", t.retrieve],
    ["draft", "Draft", t.draft],
    ["check", "Safety check", t.check],
    ["action", "Action", t.action],
  ];
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="text-sm text-neutral-600">
          Ticket <span className="font-medium text-black">#{detail.id}</span> · {detail.timestamp} UTC
        </div>
        <div className="flex items-center gap-2">
          {typeof replay === "string" && replay !== "busy" && <span className="text-xs text-red-600">{replay}</span>}
          <button
            className="bg-black text-white rounded px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            onClick={onReplay}
            disabled={replay === "busy"}
            title="POST /chat with X-Cache-Only: 1. Same message, no network; a cache miss is refused."
          >
            {replay === "busy" ? "Replaying…" : "Replay (cached)"}
          </button>
        </div>
      </div>
      {steps.map(([key, title, data]) => (
        <Step key={key} name={key} title={title} data={data} trace={t}>
          <StepBody name={key} trace={t} />
          {key === "action" && <WhyNot trace={t} />}
        </Step>
      ))}
      <Totals totals={t.totals} />
      <Baselines trace={t} />
    </div>
  );
}

function MetricsTile() {
  const [m, setM] = useState(null);
  useEffect(() => {
    fetch(`${API}/metrics`).then((r) => r.json()).then(setM).catch(() => setM({ available: false }));
  }, []);
  const ours = m?.systems?.find((s) => s.system === "ours");
  const cell = (label, v, digits = 2) => (
    <div className="text-center px-2">
      <div className="text-base font-semibold tabular-nums">{v === null || v === undefined ? "–" : Number(v).toFixed(digits)}</div>
      <div className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</div>
    </div>
  );
  return (
    <div className="bg-white border border-neutral-200 rounded-lg px-3 py-2 text-sm min-w-[18rem]">
      {!m && <span className="text-neutral-400">Loading metrics…</span>}
      {m && !m.available && <span className="text-neutral-500">No eval results yet. Run <code>make eval</code> to fill this tile.</span>}
      {m?.available && (
        <>
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs font-medium">Golden-set eval · ours{ours ? ` (n=${ours.n})` : ""}</div>
            <div className="text-[10px] text-neutral-400">{m.written_at}</div>
          </div>
          <div className="flex justify-between mt-1">
            {cell("intent acc", ours?.intent?.accuracy)}
            {cell("macro-F1", ours?.intent?.macro_f1)}
            {cell("esc F1", ours?.escalation?.f1)}
            {cell("auto rate", ours?.auto_handle_rate)}
            {cell("judge /25", ours?.judge?.mean_total, 1)}
          </div>
          <p className="text-[11px] text-neutral-500 mt-1">
            Scored on the hand-labelled golden set (150 tweets from 2014–17), not on the tickets listed here.
            {ours && !ours.intent && " Labels are still empty, so accuracy and F1 are blank."}
          </p>
        </>
      )}
    </div>
  );
}
