// Specialist queue: open escalations from GET /escalations, each with a Resolve button (POST /escalations/{id}/resolve).
import { useEffect, useState } from "react";
import { API } from "./App.jsx";

export default function Queue() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);

  async function load() {
    try {
      const res = await fetch(`${API}/escalations?status=open`);
      if (!res.ok) throw new Error(`${res.status}`);
      setRows(await res.json());
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }

  async function resolve(id) {
    await fetch(`${API}/escalations/${id}/resolve`, { method: "POST" });
    load();
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-base font-semibold">Open escalations ({rows.length})</h2>
        <button className="text-sm text-neutral-600 hover:text-black" onClick={load}>Refresh</button>
      </div>
      {error && <p className="text-red-600 text-sm">Could not load queue: {error}</p>}
      <div className="overflow-x-auto bg-white border border-neutral-200 rounded-lg">
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-neutral-600">
            <tr>
              <th className="p-2">#</th>
              <th className="p-2">Message</th>
              <th className="p-2">Intent</th>
              <th className="p-2">Urgency</th>
              <th className="p-2">Reason</th>
              <th className="p-2">Draft</th>
              <th className="p-2"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-neutral-100 align-top">
                <td className="p-2 text-neutral-500">{r.id}</td>
                <td className="p-2 max-w-xs">{r.message}</td>
                <td className="p-2 whitespace-nowrap">{r.intent}</td>
                <td className="p-2">{r.urgency}</td>
                <td className="p-2 whitespace-nowrap font-medium">{r.reason}</td>
                <td className="p-2 max-w-xs text-neutral-600">{r.draft || <span className="text-neutral-400">none (rule fired first)</span>}</td>
                <td className="p-2">
                  <button className="bg-black text-white rounded px-3 py-1 text-xs" onClick={() => resolve(r.id)}>Resolve</button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && !error && (
              <tr><td className="p-4 text-neutral-400" colSpan={7}>Nothing open.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
