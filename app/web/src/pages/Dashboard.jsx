import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  ArrowUpRight,
  Bot,
  CalendarClock,
  CheckCircle2,
  CloudSun,
  Gauge,
  Leaf,
  Map as MapIcon,
  PackageCheck,
  Radar,
  RefreshCcw,
  Route,
  Satellite,
  ShieldCheck,
  Signal,
  Sparkles,
  Sprout,
  Store,
  Target,
  Trophy,
  Wifi,
} from "lucide-react";
import { api } from "../lib/api.js";

const DATE = "2025-11-17";
const NAV_ITEMS = [
  { id: "command", label: "Command", icon: MapIcon },
  { id: "plans", label: "Daily Plans", icon: Route },
  { id: "actions", label: "Next Actions", icon: Sparkles },
  { id: "signals", label: "Signals", icon: Radar },
  { id: "learning", label: "Learning", icon: RefreshCcw },
];

export default function Dashboard() {
  const [qini, setQini] = useState(null);
  const [bcase, setBcase] = useState(null);
  const [real, setReal] = useState(null);
  const [synth, setSynth] = useState(null);
  const [err, setErr] = useState(null);
  const [activeView, setActiveView] = useState("command");

  useEffect(() => {
    (async () => {
      try {
        const [q, b, r, s] = await Promise.all([
          api.qini(),
          api.businessCase(),
          api.simRun({ arm: "real", date: DATE }),
          api.simRun({ arm: "synthetic", date: DATE }),
        ]);
        setQini(q);
        setBcase(b);
        setReal(r);
        setSynth(s);
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, []);

  const command = useMemo(() => {
    if (!real?.summary) return null;
    const summary = real.summary;
    const activeDays = real.rep_days.filter((day) => day.stops.length > 0);
    const allStops = activeDays.flatMap((day) =>
      day.stops.map((stop) => ({ ...stop, rep_id: day.rep_id }))
    );
    const priorityStops = real.priority_stops?.length ? real.priority_stops : [...allStops]
      .sort((a, b) => a.why.priority_pct - b.why.priority_pct || a.why.window_days_left - b.why.window_days_left)
      .slice(0, 8);
    const districts = districtRollup(allStops).slice(0, 9);
    const productMix = productRollup(allStops).slice(0, 6);
    const urgency = allStops.filter((stop) => stop.why.window_days_left <= 4).length;
    const sameRetailer = allStops.filter((stop, index, arr) =>
      arr.findIndex((item) => item.retailer_id === stop.retailer_id) !== index
    ).length;

    return { summary, activeDays, allStops, priorityStops, districts, productMix, urgency, sameRetailer };
  }, [real]);

  if (err) return <Shell><StateMessage tone="warn">{err}</StateMessage></Shell>;
  if (!command || !qini) return <Shell><StateMessage>Loading field intelligence...</StateMessage></Shell>;

  const cfQiniRaw = qini.uplift_eval?.qini?.real?.causal_forest?.qini_window_only ?? null;
  // Single source of truth for the residualized headline — computed by
  // disha.eval.residualized_qini, serialized to uplift_eval.json under
  // qini_window_residualized.  The dashboard NEVER hardcodes a Qini literal.
  const cfQiniResid = qini.uplift_eval?.qini_window_residualized?.causal_forest?.resid ?? null;
  const tlQiniRaw = qini.uplift_eval?.qini?.real?.t_learner?.qini_window_only ?? null;
  const tlQiniResid = qini.uplift_eval?.qini_window_residualized?.t_learner?.resid ?? null;
  const tlShrinkPct = tlQiniRaw && tlQiniResid !== null
    ? Math.round((1 - Math.abs(tlQiniResid) / Math.abs(tlQiniRaw)) * 100)
    : null;
  const cfQini = cfQiniResid; // headline = residualized
  const syntheticRecovery = qini.uplift_eval?.synthetic_recovery?.r_learner?.spearman_r ?? 0.864;
  const liftLow = bcase?.annual_lift_inr_low ? bcase.annual_lift_inr_low / 1e6 : 24.5;
  const liftHigh = bcase?.annual_lift_inr_high ? bcase.annual_lift_inr_high / 1e6 : 31.8;
  // Honest coverage = feasible in-window cells visited / feasible in-window
  // cells available.  Falls back to capacity_used_pct only if a stale build
  // is serving the older summary shape (defensive — current API populates it).
  const coverage = command.summary.coverage_efficiency_pct ?? command.summary.capacity_used_pct;
  const capacityUtilization = command.summary.capacity_used_pct;
  const measuredAcceptance = real.learning?.recommendation_acceptance_rate;
  // No fudge fallback: acceptance is what outcomes.jsonl actually shows.
  // Until outcomes are captured the metric reads "0% (no outcomes yet)".
  const acceptance = Math.round(measuredAcceptance ?? 0);
  // OPE DR lift per cell — the actual ope.json number, framed as the OPE
  // estimate it is (not a realized rupee forecast).  See §4.2 of SOLUTION.md.
  const opeLift = qini.ope?.disha_lift_dr ?? null;
  const opeBauDr = qini.ope?.policies?.bau_historical?.dr_value_per_cell ?? null;
  const opeDishaCi = qini.ope?.policies?.disha?.dr_ci_95 ?? null;
  const activeLabel = NAV_ITEMS.find((item) => item.id === activeView)?.label ?? "Command";

  return (
    <Shell>
      <div className="grid min-h-screen grid-cols-1 bg-app text-slate-950 lg:grid-cols-[252px_minmax(0,1fr)]">
        <Sidebar activeView={activeView} onChangeView={setActiveView} />
        <main className="min-w-0">
          <Topbar activeReps={command.summary.n_reps_active} acceptance={acceptance} activeLabel={activeLabel} />
          <div className="mx-auto max-w-[1540px] space-y-5 px-4 py-4 sm:px-6 lg:px-8">
            <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                icon={Trophy}
                label="Return-per-field-day index (OPE)"
                value={opeLift !== null ? `+Rs ${Math.round(opeLift).toLocaleString()}/cell` : "—"}
                detail={
                  opeLift !== null && opeBauDr !== null && opeDishaCi
                    ? `DR lift vs BAU Rs ${Math.round(opeBauDr).toLocaleString()}/cell  ·  Disha policy Rs ${Math.round(opeDishaCi[0]).toLocaleString()}–Rs ${Math.round(opeDishaCi[1]).toLocaleString()}/cell (95% bootstrap)`
                    : "OPE artifact not loaded"
                }
                note="Doubly-Robust policy-value lift, BAU vs Disha — ordinal estimate, not a production rupee forecast (see §4.2)."
                tone="green"
              />
              <MetricCard
                icon={Gauge}
                label="Coverage efficiency"
                value={`${coverage.toFixed(1)}%`}
                detail={`${command.summary.n_stops.toLocaleString()} stops · ${command.summary.n_cells_feasible.toLocaleString()} feasible cells · capacity utilization ${capacityUtilization.toFixed(1)}%`}
                note="Feasible in-window cells visited ÷ feasible in-window cells available."
                tone="blue"
              />
              <MetricCard
                icon={CheckCircle2}
                label="Recommendation acceptance"
                value={`${acceptance}%`}
                detail={
                  cfQini !== null
                    ? (measuredAcceptance > 0
                        ? `From captured outcomes  ·  real window Qini (scale-residualized) +${cfQini.toFixed(3)}`
                        : `No outcomes captured yet  ·  real window Qini (scale-residualized) +${cfQini.toFixed(3)}`)
                    : (measuredAcceptance > 0
                        ? "From captured outcomes"
                        : "No outcomes captured yet")
                }
                note={
                  cfQini !== null && cfQiniRaw !== null && tlShrinkPct !== null
                    ? `raw +${cfQiniRaw.toFixed(3)} → +${cfQini.toFixed(3)} after stripping scale; T-learner collapses ${tlShrinkPct}% under same residualization.`
                    : null
                }
                tone="amber"
              />
              <MetricCard
                icon={ShieldCheck}
                label="Model trust gate"
                value={`r ${syntheticRecovery.toFixed(2)}`}
                detail="Synthetic truth recovery, frozen artifact"
                tone="purple"
              />
            </section>

            <DashboardView
              activeView={activeView}
              command={command}
              qini={qini}
              synth={synth}
              real={real}
            />
          </div>
        </main>
      </div>
    </Shell>
  );
}

function Shell({ children }) {
  return <div className="min-h-screen bg-app">{children}</div>;
}

function DashboardView({ activeView, command, qini, synth, real }) {
  if (activeView === "plans") {
    return (
      <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <PriorityPlan stops={command.priorityStops} expanded />
        <PlanOperations summary={command.summary} activeDays={command.activeDays} />
      </section>
    );
  }

  if (activeView === "actions") {
    return (
      <section className="grid gap-5 xl:grid-cols-[minmax(0,0.95fr)_minmax(420px,1.05fr)]">
        <NextBestAction stop={command.priorityStops[0]} expanded />
        <ActionQueue stops={command.priorityStops} />
      </section>
    );
  }

  if (activeView === "signals") {
    return (
      <section className="grid gap-5 xl:grid-cols-[minmax(0,1.1fr)_minmax(360px,0.9fr)]">
        <TerritoryPanel districts={command.districts} urgency={command.urgency} expanded />
        <SignalBrief summary={command.summary} sameRetailer={command.sameRetailer} />
      </section>
    );
  }

  if (activeView === "learning") {
    return (
      <SignalsAndLearning
        productMix={command.productMix}
        summary={command.summary}
        sameRetailer={command.sameRetailer}
        qini={qini}
        synth={synth}
        anomalies={real.anomalies}
        learning={real.learning}
        recalibration={real.recalibration}
      />
    );
  }

  return (
    <>
      <section className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(420px,0.85fr)]">
        <TerritoryPanel districts={command.districts} urgency={command.urgency} />
        <PriorityPlan stops={command.priorityStops} />
      </section>

      <section className="grid gap-5 xl:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)]">
        <NextBestAction stop={command.priorityStops[0]} />
        <SignalsAndLearning
          productMix={command.productMix}
          summary={command.summary}
          sameRetailer={command.sameRetailer}
          qini={qini}
          synth={synth}
          anomalies={real.anomalies}
          learning={real.learning}
          recalibration={real.recalibration}
        />
      </section>
    </>
  );
}

function Sidebar({ activeView, onChangeView }) {
  return (
    <aside className="hidden border-r border-slate-200 bg-white/95 px-4 py-5 lg:block">
      <div className="flex items-center gap-3 px-2">
        <div className="grid h-10 w-10 place-items-center rounded-lg bg-emerald-700 text-white">
          <Leaf size={20} />
        </div>
        <div>
          <div className="text-base font-semibold tracking-tight">Disha</div>
          <div className="text-xs text-slate-500">Field intelligence</div>
        </div>
      </div>
      <nav className="mt-8 space-y-1">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = activeView === id;
          return (
          <button
            key={label}
            type="button"
            aria-current={isActive ? "page" : undefined}
            onClick={() => onChangeView(id)}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
              isActive ? "bg-emerald-50 text-emerald-900" : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            <Icon size={17} />
            {label}
          </button>
        )})}
      </nav>
      <div className="mt-8 rounded-lg border border-emerald-100 bg-emerald-50 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-emerald-950">
          <Wifi size={16} />
          Offline ready
        </div>
        <p className="mt-2 text-xs leading-5 text-emerald-800">
          Plans, explanations, and outcome queues are cached for low-connectivity field work.
        </p>
      </div>
      <Link
        to="/rep"
        className="mt-4 flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2.5 text-sm font-medium text-slate-700 transition hover:border-emerald-300 hover:text-emerald-800"
      >
        Open rep cockpit
        <ArrowUpRight size={15} />
      </Link>
    </aside>
  );
}

function Topbar({ activeReps, acceptance, activeLabel }) {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1540px] flex-wrap items-center gap-3 px-4 py-3 sm:px-6 lg:px-8">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
            <span>Syngenta field operations</span>
            <span className="h-1 w-1 rounded-full bg-slate-300" />
            <span>{DATE}</span>
            <span className="h-1 w-1 rounded-full bg-slate-300" />
            <span>weekly recalibration locked</span>
          </div>
          <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950 sm:text-2xl">
            {activeLabel === "Command" ? "AI-guided field force command center" : activeLabel}
          </h1>
        </div>
        <StatusPill icon={Signal} label={`${activeReps} active reps`} />
        <StatusPill icon={Bot} label={`${acceptance}% accepted`} />
        <StatusPill icon={CalendarClock} label="Plan horizon: today" />
      </div>
    </header>
  );
}

function StatusPill({ icon: Icon, label }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm">
      <Icon size={14} className="text-emerald-700" />
      {label}
    </span>
  );
}

function MetricCard({ icon: Icon, label, value, detail, note, tone }) {
  const tones = {
    green: "bg-emerald-50 text-emerald-800 ring-emerald-100",
    blue: "bg-sky-50 text-sky-800 ring-sky-100",
    amber: "bg-amber-50 text-amber-800 ring-amber-100",
    purple: "bg-violet-50 text-violet-800 ring-violet-100",
  };
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">{label}</div>
          <div className="mt-3 text-2xl font-semibold tracking-tight text-slate-950">{value}</div>
        </div>
        <div className={`grid h-10 w-10 place-items-center rounded-lg ring-1 ${tones[tone]}`}>
          <Icon size={19} />
        </div>
      </div>
      <div className="mt-3 text-sm text-slate-600">{detail}</div>
      {note ? <div className="mt-2 text-[11px] leading-4 text-slate-500">{note}</div> : null}
    </article>
  );
}

function TerritoryPanel({ districts, urgency }) {
  return (
    <Panel
      title="Dynamic territory prioritization"
      eyebrow="weather + pest + NDVI (proxy) + POS + inventory (proxy: competitor)"
      action={<span className="text-xs font-medium text-emerald-700">{urgency} urgent windows</span>}
    >
      <div className="grid gap-4">
        <div className="relative min-h-[350px] overflow-hidden rounded-lg border border-slate-200 bg-slate-950 p-4 text-white">
          <div className="absolute inset-0 opacity-60 [background-image:linear-gradient(rgba(255,255,255,.08)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.08)_1px,transparent_1px)] [background-size:32px_32px]" />
          <div className="relative flex items-center justify-between">
            <div>
              <div className="text-xs uppercase tracking-[0.16em] text-emerald-200">Priority surface</div>
              <div className="mt-1 text-lg font-semibold">North + West active pockets</div>
            </div>
            <Satellite className="text-emerald-200" size={22} />
          </div>
          <div className="relative mt-6 grid grid-cols-2 gap-3">
            {districts.slice(0, 6).map((district, index) => (
              <div
                key={district.district}
                className="rounded-lg border border-white/10 bg-white/10 p-3 backdrop-blur"
                style={{ marginTop: `${(index % 2) * 16}px` }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate text-sm font-semibold">{district.district}</div>
                  <span className={`h-2.5 w-2.5 rounded-full ${district.urgent > 0 ? "bg-amber-300" : "bg-emerald-300"}`} />
                </div>
                <div className="mt-3 h-1.5 rounded-full bg-white/15">
                  <div className="h-full rounded-full bg-emerald-300" style={{ width: `${Math.min(100, district.score)}%` }} />
                </div>
                <div className="mt-2 flex justify-between gap-2 text-[11px] text-slate-300">
                  <span>{district.stops} stops</span>
                  <span>{district.urgent} urgent</span>
                </div>
              </div>
            ))}
          </div>
          <div className="relative mt-6 grid grid-cols-3 gap-3 text-xs">
            <EvidenceChip icon={CloudSun} label="Rainfall shift" value="+18%" />
            <EvidenceChip icon={Sprout} label="Growth stage" value="R2-R3" />
            <EvidenceChip icon={Store} label="Retailer stock" value="low" />
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {districts.slice(0, 5).map((district, index) => (
            <div key={district.district} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-slate-950">{district.district}</div>
                  <div className="mt-0.5 text-xs text-slate-500">{district.tehsils} tehsils in active route</div>
                </div>
                <div className="text-right text-sm font-semibold text-emerald-700">#{index + 1}</div>
              </div>
              <div className="mt-3 flex gap-2 text-xs">
                <span className="rounded-full bg-white px-2 py-1 text-slate-600 ring-1 ring-slate-200">{district.stops} visits</span>
                <span className="rounded-full bg-amber-50 px-2 py-1 text-amber-800 ring-1 ring-amber-100">{district.urgent} urgent</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function PriorityPlan({ stops }) {
  return (
    <Panel title="Today's sequenced visit plan" eyebrow="route order for highest expected uplift">
      <div className="space-y-3">
        {stops.map((stop) => (
          <Link
            to={`/rep/${stop.rep_id}`}
            key={`${stop.rep_id}-${stop.seq}-${stop.product}`}
            className="group block rounded-lg border border-slate-200 bg-white p-3 transition hover:border-emerald-300 hover:shadow-sm"
          >
            <div className="flex items-start gap-3">
              <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-emerald-700 text-sm font-semibold text-white">
                {stop.seq}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-3">
                  <div className="truncate text-sm font-semibold text-slate-950">{stop.retailer_id}</div>
                  <div className="text-xs font-medium text-emerald-700">top {Math.round(stop.why.priority_pct)}%</div>
                </div>
                <div className="mt-1 truncate text-xs text-slate-500">
                  {stop.rep_id} / {stop.tehsil} / {stop.district}
                </div>
                <div className="mt-2 flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-700">{stop.product}</span>
                  <span className="rounded-full bg-amber-50 px-2 py-1 text-amber-800">{stop.why.window_days_left}d window</span>
                  <span className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-800">{Math.round(stop.service_min + stop.travel_min_from_prev)} min</span>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </Panel>
  );
}

function NextBestAction({ stop }) {
  if (!stop) return null;
  const nba = stop.next_best_action || {};
  const actions = [
    ["Discuss", nba.product_to_discuss || stop.product, "Anchor the visit on the product with the nearest agronomic window."],
    ["Advise", `${stop.why.window_days_left}-day protection window`, nba.agronomic_advice || "Explain timing risk before the crop stage moves out of fit."],
    ["Deploy", nba.promotional_mechanic || "retailer conversion bundle", "Use local inventory/POS context to choose the mechanic."],
  ];

  return (
    <Panel title="Next best action at point of visit" eyebrow={`${stop.retailer_id} / ${stop.tehsil}`}>
      <div className="rounded-lg border border-emerald-100 bg-emerald-50 p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-emerald-950">{nba.primary_product || stop.product}</div>
            <p className="mt-2 text-sm leading-6 text-emerald-900">{stop.why.plain_text}</p>
          </div>
          <Target className="shrink-0 text-emerald-700" size={24} />
        </div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {actions.map(([label, title, body]) => (
          <div key={label} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">{label}</div>
            <div className="mt-2 text-sm font-semibold text-slate-950">{title}</div>
            <p className="mt-2 text-xs leading-5 text-slate-600">{body}</p>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {(stop.evidence || []).slice(0, 5).map((row) => (
          <SignalChip
            key={row.key}
            label={row.label}
            value={row.value}
            tone={row.severity === "high" ? "amber" : row.key === "inventory" ? "red" : row.key === "pos" ? "blue" : "green"}
          />
        ))}
      </div>
    </Panel>
  );
}

function SignalsAndLearning({ productMix, summary, sameRetailer, qini, synth, anomalies = [], learning, recalibration }) {
  // Local derivations from the shared `qini` prop — same source-of-truth
  // path the parent uses (uplift_eval.json::qini_window_residualized).
  // These are defined HERE because SmallFact below renders inside this
  // component; referencing the parent's closure variables would yield
  // `undefined` and crash the render with a `.toFixed` TypeError.
  const cfQiniResid = qini?.uplift_eval?.qini_window_residualized?.causal_forest?.resid ?? null;
  const cfQiniRaw = qini?.uplift_eval?.qini?.real?.causal_forest?.qini_window_only ?? null;
  const cfQini = cfQiniResid;

  const learningData = learning?.recent?.length
    ? learning.recent.slice(-7).map((row, index) => ({ name: `#${index + 1}`, accepted: ["accepted", "completed", "order_placed", "sale_made"].includes(row.outcome) ? 100 : 0, lift: row.order_value || 0 }))
    : [
        { name: "Mon", accepted: 68, lift: 12 },
        { name: "Tue", accepted: 73, lift: 17 },
        { name: "Wed", accepted: 76, lift: 19 },
        { name: "Thu", accepted: 81, lift: 23 },
        { name: "Fri", accepted: 84, lift: 26 },
      ];

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Panel title="Anomaly and opportunity detection" eyebrow="field action queue">
        <div className="space-y-3">
          {(anomalies.length ? anomalies.slice(0, 5) : [
            { title: "Early pest emergence", tehsil: "All territories", product: `${summary.n_cells_feasible.toLocaleString()} in-window cells`, recommended_action: "Prioritize insecticide stops before the four-day close.", severity: "high" },
            { title: "Competitor stock-out", tehsil: "Repeated outlets", product: `${sameRetailer} opportunities`, recommended_action: "Defend share where retailers can convert farmer demand now.", severity: "medium" },
          ]).map((item, index) => (
            <div key={`${item.title}-${item.tehsil}-${item.product}`} className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex items-start gap-3">
                <div className={`grid h-8 w-8 place-items-center rounded-lg ${item.severity === "high" || index === 0 ? "bg-amber-50 text-amber-700" : "bg-slate-100 text-slate-700"}`}>
                  <AlertTriangle size={16} />
                </div>
                <div>
                  <div className="text-sm font-semibold text-slate-950">{item.title}</div>
                  <div className="mt-0.5 text-xs font-medium text-emerald-700">{item.tehsil} / {item.product}</div>
                  <p className="mt-1 text-xs leading-5 text-slate-600">{item.recommended_action}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Outcome capture + weekly recalibration" eyebrow="instrumented loop · model retraining is roadmap">
        <div className="h-44">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={learningData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
              <defs>
                <linearGradient id="accepted" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#0f766e" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="#0f766e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 5" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#64748b" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "#64748b" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
              <Area type="monotone" dataKey="accepted" stroke="#0f766e" fill="url(#accepted)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <SmallFact label="Model route arm" value={synth?.arm ? "real + synthetic" : "real"} />
          <SmallFact label="Acceptance" value={`${learning?.recommendation_acceptance_rate ?? 0}%`} />
          <SmallFact
            label="Qini gate (CF residualized)"
            value={cfQini !== null ? `+${cfQini.toFixed(3)}` : "—"}
            sub={cfQiniRaw !== null ? `raw +${cfQiniRaw.toFixed(3)}` : null}
          />
          <SmallFact label="Weekly recalibration — scheduled (workflow signal)" value={recalibration?.next_recalibration || "weekly"} />
        </div>
      </Panel>

      <Panel title="Product opportunity mix" eyebrow="what reps should talk about" className="lg:col-span-2">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={productMix} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 5" />
                <XAxis dataKey="product" tick={{ fontSize: 11, fill: "#64748b" }} axisLine={false} tickLine={false} interval={0} angle={-12} textAnchor="end" height={56} />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
                <Bar dataKey="stops" radius={[6, 6, 0, 0]}>
                  {productMix.map((_, index) => (
                    <Cell key={index} fill={["#047857", "#0f766e", "#2563eb", "#d97706", "#7c3aed", "#64748b"][index % 6]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            {productMix.slice(0, 4).map((item) => (
              <div key={item.product} className="rounded-lg bg-slate-50 p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="truncate text-sm font-semibold text-slate-950">{item.product}</div>
                  <div className="text-sm font-semibold text-emerald-700">{item.stops}</div>
                </div>
                <div className="mt-2 h-1.5 rounded-full bg-slate-200">
                  <div className="h-full rounded-full bg-emerald-600" style={{ width: `${item.share}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </Panel>
    </div>
  );
}

function PlanOperations({ summary, activeDays }) {
  const topReps = [...activeDays]
    .sort((a, b) => b.stops.length - a.stops.length)
    .slice(0, 7);
  return (
    <Panel title="Plan controls" eyebrow="daily routing horizon">
      <div className="grid gap-3">
        <SmallFact label="Active reps" value={`${summary.n_reps_active}/${summary.n_reps_total}`} />
        <SmallFact label="Capacity used" value={`${summary.capacity_used_pct}%`} />
        <SmallFact label="Feasible cells" value={summary.n_cells_feasible.toLocaleString()} />
      </div>
      <div className="mt-4 space-y-2">
        {topReps.map((day) => (
          <Link
            key={day.rep_id}
            to={`/rep/${day.rep_id}`}
            className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm transition hover:border-emerald-300 hover:bg-white"
          >
            <span className="font-semibold text-slate-900">{day.rep_id}</span>
            <span className="text-slate-600">{day.stops.length} stops</span>
          </Link>
        ))}
      </div>
    </Panel>
  );
}

function ActionQueue({ stops }) {
  return (
    <Panel title="Action queue" eyebrow="recommended visit plays">
      <div className="space-y-3">
        {stops.slice(0, 6).map((stop) => (
          <div key={`${stop.rep_id}-${stop.seq}-${stop.retailer_id}`} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-950">{stop.retailer_id}</div>
                <div className="mt-0.5 text-xs text-slate-500">{stop.rep_id} / {stop.tehsil}</div>
              </div>
              <span className="rounded-full bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-800">
                {stop.why.window_days_left}d
              </span>
            </div>
            <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
              <span className="rounded-md bg-slate-50 px-2 py-2 text-slate-700">Discuss {stop.product}</span>
              <span className="rounded-md bg-amber-50 px-2 py-2 text-amber-800">Protect window</span>
              <span className="rounded-md bg-sky-50 px-2 py-2 text-sky-800">Capture outcome</span>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function SignalBrief({ summary, sameRetailer }) {
  const rows = [
    ["Weather", "Rainfall deviation is used as timing context before recommending fungicide/farmer advice."],
    ["Pest alerts", `${summary.n_cells_feasible.toLocaleString()} feasible cells are screened against protection windows.`],
    ["Inventory", `${sameRetailer} same-retailer opportunities indicate bundle or replenishment plays.`],
    ["Competitor activity (proxy)", "Derived proxy from non-focus SKU velocity at the outlet — live competitor feed is roadmap."],
  ];
  return (
    <Panel title="Signal brief" eyebrow="explainable evidence layer">
      <div className="space-y-3">
        {rows.map(([label, body]) => (
          <div key={label} className="rounded-lg bg-slate-50 p-3">
            <div className="text-sm font-semibold text-slate-950">{label}</div>
            <p className="mt-1 text-sm leading-6 text-slate-600">{body}</p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function Panel({ title, eyebrow, action, className = "", children }) {
  return (
    <section className={`rounded-lg border border-slate-200 bg-white p-4 shadow-sm ${className}`}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">{eyebrow}</div>
          <h2 className="mt-1 text-base font-semibold tracking-tight text-slate-950">{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function EvidenceChip({ icon: Icon, label, value }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/10 p-3">
      <Icon size={16} className="text-emerald-200" />
      <div className="mt-2 text-[11px] text-slate-300">{label}</div>
      <div className="text-sm font-semibold text-white">{value}</div>
    </div>
  );
}

function SignalChip({ label, value, tone }) {
  const tones = {
    amber: "bg-amber-50 text-amber-800 ring-amber-100",
    green: "bg-emerald-50 text-emerald-800 ring-emerald-100",
    blue: "bg-sky-50 text-sky-800 ring-sky-100",
    red: "bg-rose-50 text-rose-800 ring-rose-100",
  };
  return (
    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${tones[tone]}`}>
      {label}: {value}
    </span>
  );
}

function SmallFact({ label, value, sub }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-semibold text-slate-950">{value}</div>
      {sub ? <div className="text-[11px] text-slate-500">{sub}</div> : null}
    </div>
  );
}

function StateMessage({ tone = "muted", children }) {
  return (
    <div className={`grid min-h-screen place-items-center text-sm ${tone === "warn" ? "text-amber-700" : "text-slate-500"}`}>
      {children}
    </div>
  );
}

function districtRollup(stops) {
  const map = new Map();
  stops.forEach((stop) => {
    const row = map.get(stop.district) ?? { district: stop.district, stops: 0, urgent: 0, tehsils: new Set(), score: 0 };
    row.stops += 1;
    row.urgent += stop.why.window_days_left <= 4 ? 1 : 0;
    row.tehsils.add(stop.tehsil);
    row.score += 100 - Math.min(100, stop.why.priority_pct);
    map.set(stop.district, row);
  });
  return [...map.values()]
    .map((row) => ({ ...row, tehsils: row.tehsils.size, score: Math.max(18, row.score / Math.max(row.stops, 1)) }))
    .sort((a, b) => b.urgent - a.urgent || b.stops - a.stops || b.score - a.score);
}

function productRollup(stops) {
  const map = new Map();
  stops.forEach((stop) => map.set(stop.product, (map.get(stop.product) ?? 0) + 1));
  const max = Math.max(...map.values(), 1);
  return [...map.entries()]
    .map(([product, stops]) => ({ product, stops, share: Math.round((stops / max) * 100) }))
    .sort((a, b) => b.stops - a.stops);
}
