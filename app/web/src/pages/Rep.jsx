import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  Bot,
  Check,
  ChevronRight,
  Clock,
  CloudUpload,
  Compass,
  MapPin,
  MessageCircle,
  PackageCheck,
  Phone,
  Play,
  RefreshCcw,
  Route,
  Send,
  ShieldCheck,
  Sparkles,
  Store,
  ThumbsDown,
  ThumbsUp,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import { api } from "../lib/api.js";
import { cachePlan, enqueueOutcome, flushOutcomes, pendingOutcomes, readCachedPlan } from "../lib/offline.js";

const DEMO_REP_DEFAULT = "REP_0100";
const DEMO_DATE = "2025-11-17";

export default function Rep() {
  const { repId: repIdParam } = useParams();
  const repId = repIdParam || DEMO_REP_DEFAULT;

  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [online, setOnline] = useState(navigator.onLine);
  const [pending, setPending] = useState(0);
  const [outcomes, setOutcomes] = useState({});
  const [activeStop, setActiveStop] = useState(null);
  const [demoStep, setDemoStep] = useState(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const data = await api.plan(repId, { date: DEMO_DATE, arm: "real" });
        setPlan(data);
        await cachePlan(repId, DEMO_DATE, "real", data);
        setError(null);
      } catch (e) {
        const cached = await readCachedPlan(repId, DEMO_DATE, "real");
        if (cached) {
          setPlan(cached);
          setError("Offline mode: using the last cached plan.");
        } else {
          setError(String(e));
        }
      }
      setPending((await pendingOutcomes()).length);
      setLoading(false);
    })();

    const onUp = () => setOnline(true);
    const onDown = () => setOnline(false);
    window.addEventListener("online", onUp);
    window.addEventListener("offline", onDown);
    return () => {
      window.removeEventListener("online", onUp);
      window.removeEventListener("offline", onDown);
    };
  }, [repId]);

  useEffect(() => {
    if (online && pending > 0) {
      (async () => {
        const queued = await flushOutcomes();
        if (queued.length) {
          try {
            await api.postOutcomes(queued);
            setPending(0);
          } catch {
            for (const outcome of queued) await enqueueOutcome(outcome);
            setPending((await pendingOutcomes()).length);
          }
        }
      })();
    }
  }, [online, pending]);

  const stops = plan?.rep_day?.stops ?? [];
  const metrics = useMemo(() => {
    const used = plan?.rep_day?.capacity_used_min ?? 0;
    const budget = plan?.rep_day?.capacity_budget_min ?? 480;
    const urgent = stops.filter((stop) => stop.why.window_days_left <= 4).length;
    const uniqueRetailers = new Set(stops.map((stop) => stop.retailer_id)).size;
    return { used, budget, urgent, uniqueRetailers };
  }, [plan, stops]);

  const handleOutcome = async (stop, kind) => {
    const outcome = {
      rep_id: repId,
      retailer_id: stop.retailer_id,
      tehsil: stop.tehsil,
      product: stop.product,
      seq: stop.seq,
      outcome: kind,
      captured_at: new Date().toISOString(),
    };
    setOutcomes((current) => ({ ...current, [stop.seq]: kind }));
    if (online) {
      try {
        await api.postOutcomes([outcome]);
      } catch {
        const count = await enqueueOutcome(outcome);
        setPending(count);
      }
    } else {
      const count = await enqueueOutcome(outcome);
      setPending(count);
    }
  };

  const startDemo = () => {
    setDemoStep(0);
    setActiveStop(null);
    setOutcomes({});
  };

  const stopDemo = () => {
    setDemoStep(null);
    setActiveStop(null);
  };

  useEffect(() => {
    if (demoStep === null) return;
    const timing = [700, 850, 950, 900, 1000, 950, 900];
    const timer = setTimeout(() => {
      if (demoStep === 1) setActiveStop(stops[0] || null);
      if (demoStep === 3) setOnline(false);
      if (demoStep === 4 && stops[0]) handleOutcome(stops[0], "accepted");
      if (demoStep === 5) setOnline(true);
      if (demoStep === 6) setActiveStop(null);
      setDemoStep((step) => (step !== null && step < 6 ? step + 1 : null));
    }, timing[demoStep] || 800);
    return () => clearTimeout(timer);
  }, [demoStep, stops]);

  return (
    <div className="min-h-screen bg-app text-slate-950">
      <RepHeader
        repId={repId}
        online={online}
        pending={pending}
        demoStep={demoStep}
        onStartDemo={startDemo}
        onStopDemo={stopDemo}
      />

      {loading && <CenterMessage>Loading today's plan...</CenterMessage>}
      {error && !plan && <CenterMessage tone="warn">{error}</CenterMessage>}

      {plan && (
        <main className="mx-auto grid max-w-6xl gap-4 px-4 pb-28 pt-4 lg:grid-cols-[minmax(0,0.92fr)_minmax(360px,0.55fr)]">
          <section className={activeStop ? "hidden lg:block" : ""}>
            {error && <OfflineBanner message={error} />}
            <DaySummary plan={plan} metrics={metrics} stops={stops} />
            <StopList stops={stops} outcomes={outcomes} activeSeq={activeStop?.seq} onOpenStop={setActiveStop} />
          </section>

          <aside className={activeStop ? "" : "hidden lg:block"}>
            {activeStop ? (
              <StopDetail
                stop={activeStop}
                outcome={outcomes[activeStop.seq]}
                onBack={() => setActiveStop(null)}
                onOutcome={(kind) => handleOutcome(activeStop, kind)}
              />
            ) : (
              <FieldCopilot stops={stops} metrics={metrics} />
            )}
          </aside>
        </main>
      )}

      {demoStep !== null && <DemoOverlay step={demoStep} />}
    </div>
  );
}

function RepHeader({ repId, online, pending, demoStep, onStartDemo, onStopDemo }) {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/92 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
        <Link to="/" className="grid h-10 w-10 place-items-center rounded-lg border border-slate-200 text-slate-600">
          <ArrowLeft size={18} />
        </Link>
        <div className="min-w-0 flex-1">
          <div className="text-xs uppercase tracking-[0.14em] text-slate-500">Rep cockpit</div>
          <div className="truncate text-lg font-semibold tracking-tight text-slate-950">{repId}</div>
        </div>
        <SyncPill online={online} pending={pending} />
      </div>
      <div className="mx-auto flex max-w-6xl gap-2 px-4 pb-3">
        {demoStep === null ? (
          <button onClick={onStartDemo} className="inline-flex items-center gap-2 rounded-lg bg-emerald-700 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-emerald-800">
            <Play size={14} />
            Demo morning
          </button>
        ) : (
          <button onClick={onStopDemo} className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700">
            <RefreshCcw size={14} />
            Reset demo
          </button>
        )}
        <div className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600">
          <Clock size={14} />
          {DEMO_DATE}
        </div>
      </div>
    </header>
  );
}

function SyncPill({ online, pending }) {
  if (!online && pending > 0) {
    return <span className="inline-flex items-center gap-2 rounded-full bg-amber-50 px-3 py-1.5 text-xs font-semibold text-amber-800 ring-1 ring-amber-100"><WifiOff size={14} />{pending} queued</span>;
  }
  if (!online) {
    return <span className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700 ring-1 ring-slate-200"><WifiOff size={14} />Offline</span>;
  }
  if (pending > 0) {
    return <span className="inline-flex items-center gap-2 rounded-full bg-sky-50 px-3 py-1.5 text-xs font-semibold text-sky-800 ring-1 ring-sky-100"><CloudUpload size={14} />Syncing</span>;
  }
  return <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-800 ring-1 ring-emerald-100"><Wifi size={14} />Synced</span>;
}

function OfflineBanner({ message }) {
  return (
    <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
      {message}
    </div>
  );
}

function DaySummary({ plan, metrics, stops }) {
  const capacityPct = Math.min(100, Math.round((metrics.used / Math.max(metrics.budget, 1)) * 100));
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">Today's optimized plan</div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
            {stops.length} stops across {metrics.uniqueRetailers} outlets
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            The route is sequenced by uplift rank, agronomic window urgency, and travel capacity. Each stop includes a plain-language reason and an outcome button — outcomes are captured and feed the weekly recalibration workflow.
          </p>
        </div>
        <div className="hidden rounded-lg bg-slate-950 px-4 py-3 text-white sm:block">
          <div className="text-xs text-slate-300">Route arm</div>
          <div className="mt-1 text-lg font-semibold capitalize">{plan.arm}</div>
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <MiniMetric icon={Route} label="Capacity" value={`${capacityPct}%`} sub={`${Math.round(metrics.used)}/${Math.round(metrics.budget)} min`} />
        <MiniMetric icon={AlertTriangle} label="Urgent windows" value={metrics.urgent} sub="4 days or less" />
        <MiniMetric icon={ShieldCheck} label="Explainability" value="100%" sub="every stop has WHY" />
      </div>
      <div className="mt-4 h-2 rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-emerald-600" style={{ width: `${capacityPct}%` }} />
      </div>
    </section>
  );
}

function MiniMetric({ icon: Icon, label, value, sub }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3">
      <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
        <Icon size={14} className="text-emerald-700" />
        {label}
      </div>
      <div className="mt-2 text-xl font-semibold text-slate-950">{value}</div>
      <div className="text-xs text-slate-500">{sub}</div>
    </div>
  );
}

function StopList({ stops, outcomes, activeSeq, onOpenStop }) {
  return (
    <ol className="mt-4 space-y-3">
      {stops.length === 0 && (
        <li className="rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-600">
          No high-priority in-window cells today. Use the day for relationship work and CRM hygiene.
        </li>
      )}
      {stops.map((stop) => (
        <li key={`${stop.seq}-${stop.retailer_id}-${stop.product}`}>
          <button
            onClick={() => onOpenStop(stop)}
            className={`w-full rounded-lg border bg-white p-4 text-left shadow-sm transition hover:border-emerald-300 hover:shadow-md ${
              activeSeq === stop.seq ? "border-emerald-400 ring-2 ring-emerald-100" : "border-slate-200"
            }`}
          >
            <div className="flex items-start gap-3">
              <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-emerald-700 text-sm font-semibold text-white">
                {stop.seq}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="truncate text-base font-semibold text-slate-950">{stop.retailer_id}</div>
                    <div className="mt-0.5 flex items-center gap-1 text-sm text-slate-500">
                      <MapPin size={13} />
                      {stop.tehsil}, {stop.district}
                    </div>
                  </div>
                  <OutcomeBadge outcome={outcomes[stop.seq]} />
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">{stop.product}</span>
                  <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-emerald-800">top {Math.round(stop.why.priority_pct)}%</span>
                  <span className={`rounded-full px-2.5 py-1 ${stop.why.window_days_left <= 4 ? "bg-amber-50 text-amber-800" : "bg-sky-50 text-sky-800"}`}>
                    {stop.why.window_days_left}d window
                  </span>
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">{Math.round(stop.service_min + stop.travel_min_from_prev)} min</span>
                </div>
              </div>
              <ChevronRight className="mt-2 shrink-0 text-slate-400" size={18} />
            </div>
          </button>
        </li>
      ))}
    </ol>
  );
}

function OutcomeBadge({ outcome }) {
  if (!outcome) return <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium text-slate-500">open</span>;
  if (["accepted", "completed", "order_placed", "sale_made"].includes(outcome)) return <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-800"><Check size={12} />{outcome === "order_placed" ? "order" : "accepted"}</span>;
  if (outcome === "no_purchase") return <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2 py-1 text-xs font-semibold text-rose-800"><X size={12} />no buy</span>;
  return <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700"><X size={12} />skipped</span>;
}

function FieldCopilot({ stops, metrics }) {
  const first = stops[0];
  return (
    <section className="sticky top-[92px] rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-emerald-50 text-emerald-700">
          <Bot size={18} />
        </div>
        <div>
          <div className="text-sm font-semibold text-slate-950">Field co-pilot</div>
          <div className="text-xs text-slate-500">Ready for point-of-visit guidance</div>
        </div>
      </div>
      {first && (
        <div className="mt-4 rounded-lg bg-slate-50 p-3">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">Start here</div>
          <div className="mt-2 text-sm font-semibold text-slate-950">{first.retailer_id}</div>
          <p className="mt-1 text-sm leading-6 text-slate-600">{first.why.plain_text}</p>
        </div>
      )}
      <div className="mt-4 grid grid-cols-2 gap-3">
        <MiniMetric icon={Compass} label="Route focus" value={stops.length} sub="priority stops" />
        <MiniMetric icon={AlertTriangle} label="Urgency" value={metrics.urgent} sub="act today" />
      </div>
    </section>
  );
}

function StopDetail({ stop, outcome, onBack, onOutcome }) {
  const nba = stop.next_best_action || {};
  const signalRows = stop.evidence || [];
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm lg:sticky lg:top-[92px]">
      <button onClick={onBack} className="mb-3 inline-flex items-center gap-2 text-sm font-medium text-slate-600 lg:hidden">
        <ArrowLeft size={16} />
        Back to plan
      </button>
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">Stop #{stop.seq}</div>
          <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">{stop.retailer_id}</h2>
          <div className="mt-1 text-sm text-slate-500">{stop.tehsil}, {stop.district}</div>
        </div>
        <div className="rounded-lg bg-amber-50 px-3 py-2 text-right text-amber-900">
          <div className="text-xs">Window</div>
          <div className="text-lg font-semibold">{stop.why.window_days_left}d</div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-emerald-100 bg-emerald-50 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-emerald-950">
          <Sparkles size={16} />
          Next best action
        </div>
        <div className="mt-2 text-lg font-semibold text-emerald-950">{nba.primary_product || stop.product}</div>
        <p className="mt-2 text-sm leading-6 text-emerald-900">
          {nba.agronomic_advice || "Recommend product discussion, agronomic timing advice, and a retailer conversion mechanic based on the current local window."}
        </p>
        {nba.promotional_mechanic && (
          <div className="mt-3 rounded-md bg-white/70 px-3 py-2 text-sm text-emerald-950">
            {nba.promotional_mechanic}
          </div>
        )}
      </div>

      <div className="mt-4 space-y-2">
        {signalRows.map((row) => (
          <div key={row.key || row.label} className="flex items-start gap-3 rounded-lg bg-slate-50 p-3">
            <PackageCheck className="mt-0.5 shrink-0 text-emerald-700" size={16} />
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{row.label}</div>
              <div className="mt-1 text-sm text-slate-700">{row.value}</div>
              <div className="mt-1 text-xs leading-5 text-slate-500">{row.reason}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-slate-200 p-3">
        <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">Why this recommendation</div>
        <p className="mt-2 text-sm leading-6 text-slate-700">{stop.why.plain_text}</p>
        <div className="mt-2 text-xs text-slate-500">Driver: {stop.why.driver_human}</div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <button className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700">
          <Phone size={15} />
          Call
        </button>
        <button className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700">
          <MessageCircle size={15} />
          WA
        </button>
        <button className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700">
          <Send size={15} />
          Send
        </button>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <button onClick={() => onOutcome("order_placed")} className={`rounded-lg px-3 py-3 text-sm font-semibold ${outcome === "order_placed" ? "bg-emerald-700 text-white" : "bg-emerald-50 text-emerald-800"}`}>
          <ThumbsUp size={16} className="mx-auto mb-1" />
          Order
        </button>
        <button onClick={() => onOutcome("no_purchase")} className={`rounded-lg px-3 py-3 text-sm font-semibold ${outcome === "no_purchase" ? "bg-rose-700 text-white" : "bg-rose-50 text-rose-800"}`}>
          <ThumbsDown size={16} className="mx-auto mb-1" />
          No buy
        </button>
        <button onClick={() => onOutcome("skipped")} className={`rounded-lg px-3 py-3 text-sm font-semibold ${outcome === "skipped" ? "bg-slate-700 text-white" : "bg-slate-100 text-slate-700"}`}>
          <Store size={16} className="mx-auto mb-1" />
          Skip
        </button>
      </div>
    </section>
  );
}

function CenterMessage({ children, tone = "muted" }) {
  return (
    <div className={`grid min-h-[70vh] place-items-center text-sm ${tone === "warn" ? "text-amber-700" : "text-slate-500"}`}>
      {children}
    </div>
  );
}

function DemoOverlay({ step }) {
  const scripts = [
    "Morning sync: route, WHY text, and outcome controls load for offline use.",
    "Open the first priority stop.",
    "Use the next-best-action card at the retailer counter.",
    "Connectivity drops; the plan remains usable.",
    "Outcome is captured locally for learning.",
    "Reconnect; queued outcomes sync.",
    "Demo complete: plan, action, explainability, offline, learning.",
  ];
  return (
    <div className="fixed bottom-4 left-4 right-4 z-30 mx-auto max-w-3xl">
      <div className="rounded-lg border border-emerald-200 bg-white p-4 shadow-xl">
        <div className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-700">demo step {step + 1}/7</div>
        <div className="mt-1 text-sm text-slate-800">{scripts[step]}</div>
      </div>
    </div>
  );
}
