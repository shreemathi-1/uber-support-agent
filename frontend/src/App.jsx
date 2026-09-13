// Three tabs, one piece of state. Help = the rider-facing chat; Queue = the specialist view of escalations;
// Trace = what the pipeline did for each ticket, step by step.
import { useState } from "react";
import Help from "./Help.jsx";
import Queue from "./Queue.jsx";
import Trace from "./Trace.jsx";

export const API = "http://localhost:8000";

export default function App() {
  // ?tab=trace opens a tab directly (deep links, screenshots); no router needed for three tabs
  const [tab, setTab] = useState(new URLSearchParams(window.location.search).get("tab") || "help");
  const tabClass = (name) =>
    `px-4 py-2 text-sm font-medium rounded-full ${tab === name ? "bg-white text-black" : "text-neutral-300 hover:text-white"}`;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-black text-white px-4 py-3 flex items-center justify-between">
        <div className="text-lg font-semibold tracking-tight">Uber <span className="font-normal">Help</span></div>
        <nav className="flex gap-1">
          <button className={tabClass("help")} onClick={() => setTab("help")}>Help</button>
          <button className={tabClass("queue")} onClick={() => setTab("queue")}>Queue</button>
          <button className={tabClass("trace")} onClick={() => setTab("trace")}>Trace</button>
        </nav>
      </header>
      <main className={`flex-1 w-full mx-auto p-4 ${tab === "trace" ? "max-w-7xl" : "max-w-3xl"}`}>
        {tab === "help" && <Help />}
        {tab === "queue" && <Queue />}
        {tab === "trace" && <Trace />}
      </main>
    </div>
  );
}
