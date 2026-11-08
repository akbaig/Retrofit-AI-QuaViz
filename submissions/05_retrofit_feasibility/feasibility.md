# 5. Retrofit feasibility and business case

Low-cost retrofit instrumentation, Capex/Opex, read-only OT integration, and a worked EUR cash-flow model for one duty pump. **Separate from** the edge software stack in [Chapter 4](../04_edge/edge_design.md).

Canonical kit source: [`../../retrofit_starter/RETROFIT_FEASIBILITY.md`](../../retrofit_starter/RETROFIT_FEASIBILITY.md).  
Reproduce numbers: [`../../retrofit_starter/examples/business_case.py`](../../retrofit_starter/examples/business_case.py).

All EUR figures below are **illustrative** (list prices + stated assumptions on synthetic data). Not a quote for a real site.

## 5.1 Instrumentation pathways

| Level | What it adds | What it can see | What it cannot see |
|---|---|---|---|
| **0 — existing signals** | Motor current (MCC/drive), discharge pressure, VFD speed/power if present | Hydraulic restriction, severe impeller wear, clogging on VFDs, stuck/offset transmitters, communication loss | Early bearing damage, imbalance, misalignment (no vibration) |
| **1 — low-cost wireless** | Suction pressure + wireless vib/temp on bearing housing | Clogging, cavitation, stage-2+ bearing damage, imbalance/misalignment | Short events between sparse (~5 min) updates; flow-based efficiency |
| **2 — full wired (QuaViz target)** | Wired vib/temp (1 s), power meter, optional flow, winding PT100, valve feedback | Operating point / efficiency, early mechanical faults, better sensor-vs-machine naming | Direct seal leak (not in this benchmark) |

Transfer rule of thumb: start from drive/MCC, add one vib+temp point per critical bearing, add process sensors only where they change a maintenance decision.

## 5.2 Bill of materials (one pump)

List prices as of 30 Oct 2026 (ex VAT unless noted); see kit doc for references.

| Item | Example | ~EUR |
|---|---|---:|
| Wired vib + temp (IO-Link) | ifm VVB001 | ~530–632 |
| Wired vib + temp (Modbus) | Banner QM30VT2 | ~320 |
| Pressure transmitter | WIKA A-10 / Danfoss MBS 3100 | ~145–288 |
| PT100 winding | RS PRO 4-wire | ~108 |
| Energy meter | Carlo Gavazzi EM24 / Schneider PM5110 | ~430–630 |
| Edge computer | RevPi Connect 4 / SIMATIC IOT2050 | ~497–602 |
| IO-Link master / gateway | ifm AL1350 / Moxa MGate | ~373–769 |

| Package | Contents | Hardware total |
|---|---|---|
| **A** wired IO-Link | 2× VVB001, AL1350, 2× pressure, PT100, EM24, RevPi | ~€2.8–3.1 k |
| **B** Modbus minimum | 2× QM30VT2, same instruments, IOT2050 | ~€2.1–2.4 k |

**Installed Capex (assumption):** package B ~€2.25 k + cabling / enclosure / labour ~€2.25 k → **~€4,500 / pump**.  
**Annual Opex (assumption):** connectivity, software care, sensor replacements → **~€700 / pump·yr**.

## 5.3 Cash-flow model (formulas)

For one critical duty pump (no automatic standby):

\[
\begin{aligned}
\text{caught/yr} &= \lambda \cdot R \\
\text{gross saving per caught event}
  &= (T_u - T_p)\,C_{\text{dt}} + (R_u - R_p) \\
\text{benefit/yr} &= \text{caught/yr} \times \text{gross saving} \\
\text{nuisance/yr} &= f_{\text{FA}} \cdot 365 \cdot C_{\text{FA}} \\
\text{net/yr} &= \text{benefit} - \text{nuisance} - \text{Opex} \\
\text{payback (yr)} &= \frac{\text{Capex}}{\text{net/yr}} \quad (\text{if net}>0)
\end{aligned}
\]

| Symbol | Meaning | Base value |
|---|---|---|
| \(\lambda\) | Developing faults per year | 0.6 /yr |
| \(R\) | Event recall (share caught early) | see scenarios |
| \(f_{\text{FA}}\) | False + excess alerts per pump-day | see scenarios |
| \(T_u, T_p\) | Unplanned / planned stop hours | 10 h / 3 h |
| \(C_{\text{dt}}\) | Downtime cost | €1,500 /h |
| \(R_u, R_p\) | Unplanned / planned repair cost | €9,000 / €4,000 |
| \(C_{\text{FA}}\) | Cost to check one false alert | €120 (~1.5 h labour) |
| Capex / Opex | Installed / annual | €4,500 / €700 |

**Gross saving per caught event**

\[
(10-3)\times 1500 + (9000-4000) = \mathbf{€15{,}500}
\]

**Why false alarms dominate.** One false alert per pump-day costs

\[
1 \times 365 \times 120 = \mathbf{€43{,}800}/\text{yr}
\]

— enough to erase the benefit of catching real faults. That is why [Chapter 2 calibration](../02_calibration/calibration.md) prioritizes persistence and high thresholds, and why the official scorer weights alert burden.

## 5.4 Computed scenarios

Same Capex/Opex and cost assumptions; only \(R\) and \(f_{\text{FA}}\) change.

### A — Kit documentation case (B1 validation metrics)

\(R = 0.889\), \(f_{\text{FA}} = 0.071\) (as in `RETROFIT_FEASIBILITY.md` §5).

| Step | Computation | Result |
|---|---|---:|
| Caught / yr | \(0.6 \times 0.889\) | 0.533 |
| Benefit / yr | \(0.533 \times 15{,}500\) | €8,268 |
| Nuisance / yr | \(0.071 \times 365 \times 120\) | €3,110 |
| Net / yr | \(8268 - 3110 - 700\) | **€4,458** |
| Payback | \(4500 / 4458\) | **~1.0 yr** |

Reproduce:

```sh
cd retrofit_starter
python examples/business_case.py --recall 0.889 --fa-per-day 0.071
```

### B — Frozen **best** on official validation (optimistic)

From [`original_validation_report.json`](../../retrofit_starter/solution/final_model/results/original_validation_report.json): \(R = 1.0\), \(f_{\text{FA}} = 0.0714\). Validation entered selection — treat as optimistic.

| Step | Result |
|---|---:|
| Caught / yr | 0.600 |
| Benefit / yr | €9,300 |
| Nuisance / yr | €3,127 |
| Net / yr | **€5,473** |
| Payback | **~0.8 yr** |

### C — Frozen **best** on post-freeze holdout (transfer stress)

From holdout `Quaviz_frozen`: \(R = 0.705\), \(f_{\text{FA}} = 0.413\).

| Step | Result |
|---|---:|
| Caught / yr | 0.423 |
| Benefit / yr | €6,556 |
| Nuisance / yr | €18,089 |
| Net / yr | **−€12,233** |
| Payback | **never** (net &lt; 0) |

Under these plant-cost assumptions, **alert burden on the holdout overwhelms downtime savings**. That does not void detection value; it shows transfer sites need the same discipline as Chapter 2 (or site-specific re-tuning of policy) before Capex is justified. Prefer holdout-like metrics for transfer claims; prefer validation only for “best case under selection.”

### Sensitivity (kit base case A)

From `business_case.py` (one input varied at a time):

| Input | Low → High | Net EUR/yr | Payback |
|---|---|---|---|
| Faults / yr | 0.2 → 1.2 | −1,054 → 12,726 | never → 0.4 yr |
| Recall | 0.4 → 0.95 | −90 → 5,025 | never → 0.9 yr |
| FA / pump-day | 0 → 0.5 | 7,568 → −14,332 | 0.6 yr → never |
| Downtime €/h | 0 → 5,000 | −1,143 → 17,526 | never → 0.3 yr |
| Capex | 2,500 → 12,000 | 4,458 (unchanged) | 0.6 → 2.7 yr |

Break-even for a 5-year payback (kit assumptions): downtime cost ≈ **€547 /h**.  
Redundant duty/standby (\(C_{\text{dt}} \approx 0\), \(\lambda = 0.3\)): net ≈ **−€2,476 /yr** — case rests on repair avoidance / energy / crew time, not downtime.

## 5.5 Read-only OT integration

```
 [ SIS ]  untouched, authoritative trips
 [ PLC / drive / MCC ] ── read-only (OPC UA / Modbus read)
         │
 [ Retrofit sensors ] ──> [ Edge box ]  buffer + CPU infer (offline OK)
                                   │  one-way / DMZ
                                   ▼
                         [ LAN console / historian ]  advisory only
```

- No write path to PLC, drive, valves, or SIS (IEC 62443, NIST SP 800-82r3, IEC 61511).
- Ships: IACS UR E26/E27 / IMO MSC.428(98) — treat cloud links as untrusted; keep monitoring outside critical zones or one-way.
- Class CBM credit needs approved schemes; this prototype is advisory only.

Edge *software* mapping of this box: [Chapter 4](../04_edge/edge_design.md).

## 5.6 Pilot sequence

1. Read-only data 4–8 weeks; baselines; no operator alerts yet.  
2. Shadow mode vs work orders / SCADA.  
3. Add one retrofit sensor where shadow shows a blind spot.  
4. Advisory pilot with named owner and false-alert close-out rules.  
5. Scale on measured net value (avoided stops − nuisance), not accuracy alone.

## 5.7 When retrofit is / is not cost-effective

**Favours retrofit:** critical duty with no standby; downtime ≳ €500/h; maintenance org that can act on ~hours of lead time.  
**Weak case:** automatic standby (downtime ≈ 0); shelf-spare / hot-swap pumps; FA rates near 1/day; hazardous-area / class costs that exceed avoided loss; no owner for alerts.

## 5.8 Link to model choices

| Design choice | Feasibility reason |
|---|---|
| Strict alert policy (Ch. 2) | FA labour €120/check; 1 FA/day ≈ €43.8 k/yr |
| Sensor specialists | Avoid scheduling mechanical overhauls for cheap transmitter faults |
| CPU tree ensemble on edge (Ch. 4) | Fits RevPi / IOT2050 class BOM without GPU |
| What-if repair costs (Ch. 3) | Align with planned intervention (~€1.8 k parts/labour examples), not unplanned €9 k overhauls |
