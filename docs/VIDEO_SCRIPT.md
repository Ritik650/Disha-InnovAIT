# Disha — Demo Video Script & Shot List

**Target length:** as specified by competition (default plan ≤ 3:00).
**Team members visible:** every team member appears on camera at least once
(intro round-robin + one technical-handoff cut).

---

## Act 1 — The Tuesday morning (≈60 s) — REP APP

**Setting:** mobile phone held by a team member (the "rep"), face visible.

| t | Voice-over | On screen |
|---|---|---|
| 00:00 | "It's 7:30 Tuesday morning in Patna district. A Syngenta rep opens Disha." | Hand opening rep app → `/rep/REP_0100` |
| 00:08 | "Today's ordered stops — already in his pocket, even if the cell tower's down." | List view: 6 stops, capacity bar |
| 00:18 | "He taps the first one. The model says where this ranks and why — top of his territory this week, window closing in 4 days, driven by offline-campaign reach." | Stop detail page; priority chip + WHY block highlighted |
| 00:30 | "Goes offline." | Tap the offline toggle in the demo overlay; sync pill turns "Offline" |
| 00:35 | "Logs the outcome. Queues locally." | "Completed" button tapped; pill becomes "1 pending" |
| 00:42 | "Reconnects. Syncs automatically." | Toggle online; pill briefly says "Syncing", then "Synced" |
| 00:55 | "That's a rep's morning. Plan, why, offline, sync — done in a minute." | Hold on header with green Synced pill |

Camera: phone-held POV throughout, but cut to the team member's face at
the 00:30 mark for the offline toggle (humanise the technology).

---

## Act 2 — Dual-arm simulator (≈75 s) — DASHBOARD

**Setting:** laptop screen, second team member narrating.

| t | Voice-over | On screen |
|---|---|---|
| 01:00 | "Now zoom out. This is a manager's view." | `/` dashboard loads |
| 01:05 | "Our method recovers known truth at r ≈ 0.86 on a controlled DGP." | Hero card: r=0.86 chip |
| 01:14 | "On real Syngenta data the honest economic signal is +0.26 window-constrained Qini after stripping scale — modest, but provably not the scale-routing artifact a naive T-learner would produce." | Hero card: +0.263 chip (sourced live from `uplift_eval.json::qini_window_residualized.causal_forest.resid`), then scroll down to QiniTable |
| 01:30 | "Side by side. Left, synthetic — clean recovery, the method works. Right, real — modest divergence, the honesty." | Both ArmPanels visible |
| 01:45 | "And the T-learner row tells you what 'just send reps to bigger tehsils' looks like — 76% of its apparent value disappears when you strip out scale." | Highlight T-learner row in QiniTable |
| 02:00 | "Three corrections we caught on ourselves during the build, pinned by adversarial tests. Almost no team has this." | HonestyArc section visible |

---

## Act 3 — Close (≈30 s) — BUSINESS CASE + HANDOFF

| t | Voice-over | On screen |
|---|---|---|
| 02:15 | "Across 500 reps and a six-month season, reallocating treatments from zero/negative-CATE cells to the positive tail is worth ₹24.5 to ₹31.8 million per season — same reps, same days, no added cost." | BusinessCase card; LOW + HIGH bands |
| 02:35 | "We're not showing you the biggest number. We're showing you the true one — and it still pays for itself." | Slow zoom on the lower bound ₹24.5M |
| 02:50 | All team members on camera; team name; "Thanks." | Team round-out |

---

## Shot list (what each team member is doing in the recording)

- **Member 1 (build lead):** narrates Acts 1–2, runs the laptop / phone.
- **Member 2 (data scientist):** narrates Act 2's "honesty arc" section.
- **Member 3 (product/design):** narrates Act 1's rep app close + Act 3's
  bounded business case.
- **Member 4 (anchor):** Act 3 closing line + team round-out shot.

(Names + roles filled in by team before recording.)

---

## What MUST NOT appear on screen

- The literal phrase "+0.263 is the headline metric." Always use the
  dual framing ("0.86 on synthetic → 0.26 honest on real, scale stripped").
- Any number without a unit or context.
- Any visualisation with a misleading axis (zoom-in to make small effects
  look large; truncated y-axis on small Qini differences).
- AI / synthetic-looking generic dashboard layouts. Disha's UI is
  field-credible, dark, agronomic, distinctive.

---

## Recording checklist (pre-record)

1. `make demo` running; both `http://localhost:5173/` and `http://localhost:5173/rep` open.
2. Demo Morning button works deterministically (test once before recording).
3. Phone in screen-recording mode for Act 1; laptop screen-record for Act 2-3.
4. All team members in same room or video-call frame.
5. One take for Act 1 (the offline toggle is the moment).

---

## Single-archive packaging

```bash
# from repo root, after a clean clone:
make test       # 130+ tests pass
make build      # rebuild all artifacts (~5 min)
make demo       # verify dual-arm dashboard + rep app render
zip -r disha-<team>.zip . -x 'app/web/node_modules/*' -x '.git/*' -x 'data/cache/*'
```
