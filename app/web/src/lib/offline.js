// IndexedDB-backed offline cache for the rep app.
import { get, set, del, keys } from "idb-keyval";

const PLAN_KEY = (repId, date, arm) => `plan::${repId}::${date}::${arm}`;
const OUTCOME_QUEUE = "outcomes::queue";

export async function cachePlan(repId, date, arm, plan) {
  await set(PLAN_KEY(repId, date, arm), { savedAt: Date.now(), plan });
}

export async function readCachedPlan(repId, date, arm) {
  const blob = await get(PLAN_KEY(repId, date, arm));
  return blob?.plan ?? null;
}

export async function enqueueOutcome(outcome) {
  const queue = (await get(OUTCOME_QUEUE)) || [];
  queue.push({ ...outcome, ts: Date.now() });
  await set(OUTCOME_QUEUE, queue);
  return queue.length;
}

export async function pendingOutcomes() {
  return (await get(OUTCOME_QUEUE)) || [];
}

export async function flushOutcomes() {
  const queue = (await get(OUTCOME_QUEUE)) || [];
  await del(OUTCOME_QUEUE);
  return queue;
}
