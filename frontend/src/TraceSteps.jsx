// The cards of the Trace stepper: one collapsible <details> per pipeline step, plus the totals footer,
// the baselines strip and the "why not?" line. Pure rendering of the trace dict (docs/TRACE_SPEC.md).

export function Badge({ kind, children }) {
  const cls = { intent: "bg-neutral-100 text-neutral-700", auto: "bg-green-100 text-green-800",
                escalate: "bg-amber-100 text-amber-800", off: "bg-neutral-100 text-neutral-400" }[kind] || "bg-neutral-100";
  return <span className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-medium ${cls}`}>{children}</span>;
}

const ms = (n) => (n === undefined || n === null ? "" : `${n} ms`);

function status(name, t) {
  if (name === "request") return ["ran", "auto"];
  if (name === "rules") return [t.rules.reason ? `fired: ${t.rules.reason}` : "none fired", t.rules.reason ? "escalate" : "auto"];
  if (name === "action") return [t.action.action === "escalate" ? `escalate · ${t.action.reason}` : "auto", t.action.action];
  const s = t[name];
  if (!s.ran) return ["skipped", "off"];
  if (name === "check") return [s.safe ? "safe" : "unsafe", s.safe ? "auto" : "escalate"];
  if (name === "draft" && s.canned) return ["canned", "auto"];
  return [s.llm_calls !== undefined ? `${s.llm_calls} call${s.llm_calls === 1 ? "" : "s"} · ${s.cache_hits} cached` : "ran", "auto"];
}

export function Step({ name, title, trace, children }) {
  const [label, kind] = status(name, trace);
  const skipped = kind === "off";
  const step = trace[name];
  return (
    <details open={!skipped} className={`bg-white border border-neutral-200 rounded-lg ${skipped ? "opacity-60" : ""}`}>
      <summary className="cursor-pointer select-none px-3 py-2 flex items-center gap-2 text-sm">
        <span className="font-medium">{title}</span>
        <Badge kind={kind}>{label}</Badge>
        {step?.model && <span className="text-xs text-neutral-400">{step.model}</span>}
        <span className="ml-auto text-xs text-neutral-400">{ms(step?.latency_ms)}</span>
      </summary>
      <div className="px-3 pb-3 text-sm border-t border-neutral-100 pt-2">{children}</div>
    </details>
  );
}

export function StepBody({ name, trace }) {
  const t = trace;
  if (name === "request") return <Request r={t.request} />;
  if (name === "enrich") return <Enrich e={t.enrich} />;
  if (name === "rules") return <Rules r={t.rules} />;
  if (name === "retrieve") return t.retrieve.ran ? <Retrieve r={t.retrieve} /> : <Skipped why="a rule fired first, or intent = other" />;
  if (name === "draft") return t.draft.ran ? <Draft d={t.draft} /> : <Skipped why="escalated before drafting" />;
  if (name === "check") return t.check.ran ? <Check c={t.check} /> : <Skipped why="nothing to check" />;
  if (name === "action") return <Action a={t.action} />;
  return null;
}

const Skipped = ({ why }) => <p className="text-neutral-400">Skipped: {why}.</p>;
const Pre = ({ children }) => <pre className="bg-neutral-50 rounded p-2 text-xs overflow-x-auto whitespace-pre-wrap">{children}</pre>;

function Request({ r }) {
  return (
    <div>
      <div className="whitespace-pre-wrap">{r.message}</div>
      {r.history.length > 0 && (
        <div className="text-xs text-neutral-500 mt-1">history: {r.history.map((h) => `${h.role}: ${h.content}`).join(" · ")}</div>
      )}
      <div className="text-xs text-neutral-400 mt-1">conversation {r.conversation_id} · received {r.received_at}</div>
    </div>
  );
}

function Enrich({ e }) {
  const ents = Object.entries(e.parsed.entities).filter(([, v]) => v !== null && v !== false);
  return (
    <div className="grid gap-2 md:grid-cols-2">
      <Pre>{JSON.stringify(e.parsed, null, 2)}</Pre>
      <div>
        <div className="flex flex-wrap gap-1 mb-2">
          <Badge kind="intent">{e.parsed.intent} · {e.parsed.confidence.toFixed(2)}</Badge>
          <Badge kind="intent">{e.parsed.sentiment}</Badge>
          <Badge kind="intent">urgency {e.parsed.urgency}</Badge>
          {ents.map(([k, v]) => <Badge key={k} kind="escalate">{k}{v === true ? "" : `: ${v}`}</Badge>)}
          {ents.length === 0 && <span className="text-xs text-neutral-400">no entities</span>}
        </div>
        {e.retried && <p className="text-xs text-amber-700">first answer failed validation; retried once</p>}
        <details className="text-xs text-neutral-500"><summary className="cursor-pointer">raw model output</summary><Pre>{e.raw_output}</Pre></details>
      </div>
    </div>
  );
}

function Rules({ r }) {
  return (
    <table className="w-full text-xs">
      <thead className="text-neutral-500 text-left"><tr><th className="py-1 w-4"></th><th>rule</th><th>value checked</th><th>fires when</th><th>margin</th></tr></thead>
      <tbody>
        {r.rows.map((row) => (
          <tr key={row.rule} className={`border-t border-neutral-100 ${row.decisive ? "font-semibold" : ""}`}>
            <td className="py-1"><span className={`inline-block w-2.5 h-2.5 rounded-full ${row.fired ? "bg-green-500" : "bg-neutral-300"}`} /></td>
            <td>{row.rule}{row.decisive && <span className="ml-1 text-amber-700">← reason</span>}</td>
            <td className="font-mono">{String(row.value)}</td>
            <td className="text-neutral-500">{row.threshold}</td>
            <td className="tabular-nums text-neutral-500">{row.margin.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function DistanceBar({ d, cutoff }) {
  const far = cutoff !== undefined && d > cutoff;
  return (
    <div className="flex items-center gap-1 w-28 shrink-0" title={`cosine distance ${d}`}>
      <div className="flex-1 h-1.5 bg-neutral-200 rounded"><div className={`h-1.5 rounded ${far ? "bg-neutral-400" : "bg-black"}`} style={{ width: `${Math.max(2, (1 - Math.min(d, 1)) * 100)}%` }} /></div>
      <span className="text-[10px] tabular-nums text-neutral-500">{d.toFixed(2)}</span>
    </div>
  );
}

function Retrieve({ r }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div>
        <div className="text-xs text-neutral-500 mb-1">pairs · filter intent = {r.intent_filter} · {r.pairs.length} hits</div>
        {r.pairs.map((p) => (
          <div key={p.id} className={`flex gap-2 py-1 border-t border-neutral-100 ${p.distance > r.max_fact_distance ? "opacity-40" : ""}`}>
            <DistanceBar d={p.distance} cutoff={r.max_fact_distance} />
            <div className="min-w-0 text-xs"><div className="truncate">{p.customer_text}</div><div className="truncate text-neutral-500">↳ {p.reply_text}</div><div className="text-[10px] text-neutral-400">{p.id} · {p.source} · {p.intent}</div></div>
          </div>
        ))}
      </div>
      <div>
        <div className="text-xs text-neutral-500 mb-1">help chunks · usable when distance ≤ {r.max_fact_distance}</div>
        {r.chunks.map((c) => (
          <div key={c.id} className={`flex gap-2 py-1 border-t border-neutral-100 ${c.usable ? "" : "opacity-40"}`}>
            <DistanceBar d={c.distance} cutoff={r.max_fact_distance} />
            <div className="min-w-0 text-xs"><div className="font-medium">{c.title}{!c.usable && " (not shown to drafter)"}</div><div className="text-neutral-600">{c.text}</div><div className="text-[10px] text-neutral-400">{c.id}</div></div>
          </div>
        ))}
        {r.chunks.length === 0 && <p className="text-xs text-neutral-400">none</p>}
      </div>
    </div>
  );
}

function Draft({ d }) {
  return (
    <div>
      <div className="grid gap-2 md:grid-cols-2">
        <div><div className="text-xs text-neutral-500 mb-1">raw model output</div><Pre>{d.canned ? "(no LLM call: canned reply)" : d.raw_output}</Pre></div>
        <div><div className="text-xs text-neutral-500 mb-1">final reply · {d.final_reply?.length ?? 0} chars</div><Pre>{d.final_reply}</Pre></div>
      </div>
      <div className="text-xs text-neutral-500 mt-1">post-processing: {d.post_process_notes.length ? d.post_process_notes.join(", ") : "unchanged"}</div>
    </div>
  );
}

function Check({ c }) {
  return (
    <div className="flex items-center gap-2">
      <Badge kind={c.safe ? "auto" : "escalate"}>{c.safe ? "safe" : "unsafe"}</Badge>
      <span>{c.why}</span>
      {c.llm_calls === 0 && <span className="text-xs text-neutral-400">(decided without the model)</span>}
    </div>
  );
}

function Action({ a }) {
  return (
    <div>
      <div className="flex gap-1 mb-2">
        <Badge kind={a.action}>{a.action}</Badge><Badge kind="intent">reason: {a.reason}</Badge>
        <Badge kind="intent">ticket #{a.ticket_id}</Badge>{a.escalation_id && <Badge kind="intent">escalation #{a.escalation_id}</Badge>}
      </div>
      <div className="text-xs text-neutral-500 mb-1">what the customer saw</div>
      <div className="inline-block bg-neutral-100 rounded-2xl px-4 py-2 max-w-[85%]">{a.reply}</div>
    </div>
  );
}

export function WhyNot({ trace }) {
  const t = trace;
  let text;
  if (t.action.action === "auto") {
    const closest = [...t.rules.rows].sort((a, b) => a.margin - b.margin).slice(0, 2);
    text = `Why not escalate? Closest rules: ${closest.map((r) => `${r.rule} (${String(r.value)} vs ${r.threshold}, margin ${r.margin.toFixed(2)})`).join("; ")}.`;
  } else if (t.draft.final_reply) {
    text = `Why not auto? The draft "${t.draft.final_reply}" was written but ${t.action.reason === "unsafe_draft" ? "failed the safety check" : "not sent"}.`;
  } else {
    text = "Why not auto? Escalated before drafting: a rule fired on the enrichment alone.";
  }
  return <p className="text-xs text-neutral-600 mt-2 border-t border-neutral-100 pt-2">{text}</p>;
}

export function Totals({ totals }) {
  const cell = (label, v) => <div><span className="font-semibold tabular-nums">{v}</span> <span className="text-neutral-500">{label}</span></div>;
  return (
    <div className="bg-neutral-50 border border-neutral-200 rounded-lg px-3 py-2 text-xs flex flex-wrap gap-x-5 gap-y-1">
      {cell("LLM calls", totals.llm_calls)}{cell("cache hits", totals.cache_hits)}
      {cell("prompt tok", totals.prompt_tokens)}{cell("completion tok", totals.completion_tokens)}
      {cell("ms", totals.latency_ms)}{cell("USD at list price", `$${totals.estimated_paid_cost_usd.toFixed(6)}`)}
      <span className="text-neutral-400 ml-auto">tokens: {totals.token_counter}</span>
    </div>
  );
}

export function Baselines({ trace }) {
  const b = trace.baselines || {};
  const cols = [
    ["trivial", b.trivial], ["simple", b.simple],
    ["ours", { available: true, intent: trace.enrich.parsed.intent, escalate: trace.action.action === "escalate", reason: trace.action.reason, reply: trace.action.reply }],
  ];
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
      {cols.map(([name, p]) => (
        <div key={name} className={`bg-white border rounded-lg p-2 ${name === "ours" ? "border-black" : "border-neutral-200"}`}>
          <div className="font-medium mb-1">{name}</div>
          {!p?.available ? <div className="text-neutral-400">{p?.why || "not available"}</div> : (
            <>
              <div className="flex gap-1 mb-1"><Badge kind="intent">{p.intent}</Badge><Badge kind={p.escalate ? "escalate" : "auto"}>{p.escalate ? `escalate · ${p.reason}` : "auto"}</Badge></div>
              <div className="text-neutral-600 line-clamp-3">{p.reply}</div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}
