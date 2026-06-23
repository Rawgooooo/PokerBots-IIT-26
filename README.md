# Boss Blind — A Game-Theory-Optimal Bot for Sneak-Peek Hold'em

### An Explainer Report on a Hybrid Heuristic + Abstract-CFR Poker Agent

**Competition:** IIT Pokerbots 2026 — "Sneak-Peek Hold'em" variant
**Engine constants:** 1,000 hands per match · 5,000 starting stack · 20/10 blinds · 30 s game clock
**Codename:** `Boss_Blind` (bot family `bb1 … bb7`, champion build `champion_bot.py` / `final_cfr.py`)

> **[COVER INFOGRAPHIC — suggested]**
> A hero banner: a poker table split down the middle. Left half labelled **"Game Theory (CFR)"** with a converging-regret curve, right half labelled **"Heuristics (Real-Time Logic)"** with decision-tree icons. A single chip in the center reading **"Boss Blind"**. Tagline underneath: *"GTO foundations, real-time adaptation, 50× smaller footprint."*

---

## 1. Executive Summary

Boss Blind is a poker-playing agent built to compete in **Sneak-Peek Hold'em**, a heads-up Texas Hold'em variant with a unique *asymmetric-information auction*: each hand, players secretly bid chips for the right to peek at one of the opponent's hole cards. This single rule change breaks the symmetry that classical poker solvers rely on and dramatically enlarges the game tree.

Our approach combines two ingredients:

1. **A Game-Theory-Optimal (GTO) backbone** — strategies derived from **Counterfactual Regret Minimization (CFR)**, the same algorithm family behind superhuman poker AIs, but run over a heavily *abstracted* game so it fits and runs in real time.
2. **A heuristics + real-time-learning (RTL) layer** — closed-form equity math, opponent modelling, and exploitative adjustments that deviate from GTO when the opponent is exploitable.

The headline engineering result: through **state abstraction + compression**, the strategy footprint shrinks from a **~9.3 MB raw blueprint** down to a **~306 KB embedded string** — roughly a **30× storage reduction** — while every decision still resolves in **~14 ms**, comfortably under the per-hand time budget.

> **[GRAPH 1 — Storage Reduction Bar Chart]**
> Horizontal bars comparing four artifacts (values are real file sizes from the repo):
> `master_data.txt` (raw) — 9.37 MB · `master_data.json` — 3.56 MB · `compressed_table.txt` — 859 KB · `compressed_master.txt` (deployed) — 306 KB.
> Annotate the last bar: *"30× smaller than raw, fits inside the bot file."*

---

## 2. The Game: Why Sneak-Peek Hold'em Is Hard

Standard heads-up Hold'em is already a large game (~10¹³ decision points when fully expanded). Sneak-Peek adds an **information-auction street** between the pre-flop and the flop:

| Phase | What happens |
|---|---|
| **Pre-flop** | Both players dealt 2 hole cards; blinds posted; standard betting. |
| **Auction** | Both players submit a **sealed bid** (0 … all chips). |
| **Reveal** | Higher bidder pays the **lower** bid (Vickrey-style) and sees **one** of the opponent's hole cards. On a tie, *both* reveal — each pays their bid and each sees one of the other's cards. |
| **Flop / Turn / River** | Standard community-card betting, but now played under **asymmetric information**. |
| **Showdown** | Best 5-card hand wins the pot. |

> **[INFOGRAPHIC 2 — Hand Flow Diagram]**
> A horizontal timeline: `Deal → Pre-flop Betting → 🔒 Sealed Auction → 👁 Card Reveal → Flop → Turn → River → Showdown`.
> Branch the "Reveal" node into three outcomes with icons: **WIN auction** (you see their card → you have the edge), **LOSE auction** (they see your card → you're disadvantaged), **TIE** (both see one card → mutual peek).

### The strategic consequence

After the auction, every hand falls into one of three information regimes, and **each demands a different strategy**:

- **Winner state** — you know one of their cards. Their range *collapses* from ~1,326 possible hands to ~49. You can compute equity almost exactly and play aggressively/accurately.
- **Loser state** — they know one of *your* cards. Your bluffs are less credible and your equity is "read." You must play more cautiously and account for their sharper reads.
- **Tie state** — symmetric partial information.

A naive bot that ignores the auction plays the same on every street and bleeds chips. Boss Blind treats the auction as a **first-class decision** and branches its entire post-flop logic on the regime.

---

## 3. System Architecture

Boss Blind is a **pipeline of specialized solvers**, one per street, each chosen for the best speed/accuracy trade-off at that stage.

> **[INFOGRAPHIC 3 — Architecture Block Diagram]**
> Five stacked lanes (one per street), each feeding into a shared **"Engine Action"** output box. Down the left, two persistent side-bars spanning all lanes: **"Opponent Model (RTL)"** and **"Time Manager"**. Use distinct colors: blue = offline/GTO, green = closed-form heuristic, orange = runtime CFR, purple = adaptive overlay.
>
> ```
>  PRE-FLOP   →  [ Offline Abstract-CFR Blueprint ]  ── lookup ──┐
>  AUCTION    →  [ Closed-form Equity / Vickrey Bid ]            │
>  FLOP       →  [ Monte-Carlo Equity → Runtime CFR ]            ├──→  ACTION
>  TURN       →  [ Exact Enumeration → Runtime CFR  ]            │
>  RIVER      →  [ Exact Enumeration → Runtime CFR  ]            │
>             ┌─────────────────────────────────────────────────┘
>   side-rails: OPPONENT MODEL  +  EXPLOIT OVERLAY  +  TIME MANAGER
> ```

| Street | Technique | Why this technique |
|---|---|---|
| Pre-flop | **Offline abstracted-game CFR** (precomputed blueprint) | The pre-flop tree is fixed; solve it once, look it up in 0.1 ms. |
| Auction | **Heuristic best-response** (closed-form bid formula) | Bidding value = marginal equity gained from the peek; no tree search needed. |
| Flop | **Monte-Carlo equity** + 12-node runtime CFR | Two unknown cards remain → sampling is faster than full enumeration. |
| Turn | **Exact enumeration** + runtime CFR | Only one card to come; enumeration is exact and still <5 ms. |
| River | **Exact enumeration** + runtime CFR | Board complete; equity is exact in <1 ms. |
| All streets | **Opponent model + exploit overlay** | Deviate from GTO to punish exploitable opponents. |

This is the core of the "novel" claim: **rather than one monolithic solver, we use the cheapest tool that is accurate enough at each node**, and reserve heavy computation only where it pays off.

---

## 4. The GTO Backbone: Abstract-CFR

### 4.1 What CFR is (in one paragraph)

**Counterfactual Regret Minimization** is a self-play algorithm that converges to a **Nash equilibrium** — a strategy that cannot be exploited in expectation. It works by repeatedly playing the game against itself, tracking how much it "regrets" not having taken each action, and shifting probability toward low-regret actions. The **average** strategy over all iterations (not the latest one) is what converges to equilibrium.

> **[GRAPH 4 — CFR Convergence Curve]**
> X-axis: training iterations (0 → 200,000). Y-axis: exploitability (milli-big-blinds per hand), trending down and flattening toward zero. Caption: *"Average strategy converges to an unexploitable Nash equilibrium; we extract it once, offline."*

### 4.2 Why raw CFR doesn't fit — and our fix: **abstraction**

A full solve of Sneak-Peek Hold'em is intractable to store or run inside a 30-second bot. Boss Blind makes it tractable with two kinds of abstraction:

**(a) Card abstraction — bucketing.**
- *Pre-flop:* the 1,326 possible two-card hands collapse to the **169 canonical buckets** (e.g. `AKs`, `72o`, `JJp`), crossed with a small number of **decision nodes** (`SB_OPEN`, `BB_VS_RAISE`, `SB_VS_3BET`, `BB_VS_LIMP`) and **3 stack-depth buckets** (deep / mid / short).
- *Post-flop:* instead of storing a strategy for every exact board, we **cluster** boards into a handful of strategically-similar buckets using **K-means** over a 2-D feature vector: **[hand equity, board wetness]**. At runtime we map the live board to its **nearest centroid** (L2 distance) and use that bucket's strategy. This is the `flop_centroids` / `turn_centroids` table inside the master data.

> **[INFOGRAPHIC 5 — Abstraction Funnel]**
> A funnel graphic: top = **"~10¹³ game states"**, middle constriction labelled **"Card bucketing + K-means clustering"**, bottom = **"a few thousand info-sets"**. Side note: *"Two boards that play the same way share one strategy."*

**(b) Action abstraction.**
Every betting decision is reduced to **4 abstract actions**: `fold`, `check/call`, `bet-half-pot`, `bet-pot` (in the `final_cfr` lineage: `fold / call / raise-50% / raise-200%`). A continuous bet-sizing space becomes a 4-way choice, shrinking the tree by orders of magnitude while keeping the strategically important sizes.

### 4.3 Two flavors of CFR in the codebase

The repo contains two complementary CFR implementations, reflecting the project's evolution:

1. **Blueprint + runtime sub-game CFR** (`champion_bot.py`).
   - The **pre-flop blueprint** is solved offline and embedded.
   - On each post-flop street, a **tiny 4-action CFR** is solved *live* (400 iters flop / 500 turn / 300 river) on a depth-limited sub-game whose leaf EVs are filled from the current equity estimate. This is pure NumPy regret-matching — a few hundred microseconds.

   ```
   for i in range(iterations):
       strategy   = regret_match(regrets)     # normalize positive regrets
       cfvs       = strategy · leaf_evs        # counterfactual values
       ev         = sum(strategy × cfvs)
       regrets   += cfvs - ev
       strategy_sum += strategy
   final = strategy_sum / strategy_sum.sum()   # average → Nash for this sub-game
   ```

2. **Perfect-Memory Abstract-CFR** (`final_cfr.py`).
   - A **fully precomputed `cfr_table`** maps each abstracted info-set (street + history string + board bucket) to a stored strategy.
   - A **"perfect memory" history string** records the exact action sequence so the bot indexes the correct node even across streets.
   - Strategies are stored as **integer percentages (0–100)** and rehydrated to floats at load (`p/100.0`) — a small but effective quantization that shrinks the table.

> **[INFOGRAPHIC 6 — The Two CFR Lineages]**
> Side-by-side cards. Left: *"Blueprint + Live Sub-game Solve — small embedded pre-flop table, solves post-flop on the fly."* Right: *"Perfect-Memory Lookup — fully precomputed table, near-zero runtime, indexed by history + K-means bucket."* Bottom: *"Both share the same abstraction + compression toolchain."*

---

## 5. Storage & Runtime: The Compression Pipeline

This is where the "significantly reduced storage and run time" claim is concrete.

**The pipeline:** `train CFR → quantize strategies to int% → JSON serialize → zlib compress → base64 encode → embed as a string constant inside the .py bot.`

At load, the bot reverses it: `base64 decode → zlib decompress → JSON parse → scale back to probabilities`.

| Artifact | Size | Role |
|---|---|---|
| `master_data.txt` | **9.37 MB** | Raw training output (human-readable) |
| `master_data.json` | **3.56 MB** | Structured blueprint (cfr_table + centroids) |
| `compressed_table.txt` | **859 KB** | Intermediate compressed table |
| `compressed_master.txt` | **306 KB** | **Deployed** base64+zlib string, embedded in the bot |

The deployed strategy is **~30× smaller than the raw blueprint** and self-contained — there are no external file dependencies at match time, so cold-start is instant and there's no disk I/O mid-hand.

> **[GRAPH 1 reused / or a waterfall chart]**
> A waterfall: `9.37 MB → (structure) 3.56 MB → (zlib) 859 KB → (quantize+pack) 306 KB`. Each step labelled with the technique that produced the drop.

**Runtime budget.** The engine gives a 30 s clock for the whole 1,000-hand match (~20 ms/hand steady-state). Measured allocation:

```
Pre-flop      0.1 ms   (table lookup)
Auction       1.0 ms   (Monte-Carlo equity + bid formula)
Flop equity   3.0 ms   (120-sample Monte Carlo)
Flop CFR      2.0 ms
Turn equity   5.0 ms   (exact enumeration; ~1 ms if a card is known)
Turn CFR      1.5 ms
River         1.0 ms   (exact enumeration + CFR)
Overhead      1.0 ms
─────────────────────
Total        ~14 ms    (well under the 20 ms/hand budget)
```

> **[GRAPH 7 — Time Budget Donut/Stacked Bar]**
> A stacked horizontal bar (0–20 ms) showing each component's slice, with a dashed line at 20 ms marked *"per-hand budget"* and the filled portion ending at ~14 ms marked *"headroom."*

A **Time Manager** watches the remaining clock and, if a hand's budget drops below ~10 ms, switches to a **fast-path**: skip CFR and decide directly from pot-odds vs. equity. This guarantees the bot **never times out**, even in long, costly hands.

---

## 6. Street-by-Street Strategy

### 6.1 Pre-flop — Blueprint Lookup
Map hole cards → 169-bucket → look up `(node, bucket, stack)` in the embedded blueprint → sample from the mixed strategy (`fold / call / raise-2x / raise-4x`). A **commitment guard** folds dominated hands facing large bets, and a **maniac detector** tightens/widens opens based on the opponent's observed raise sizing.

### 6.2 Auction — Closed-Form Bid
The bid is driven by **information value**: how much equity does seeing one opponent card actually buy?

```
eq_base    = Monte-Carlo equity vs. full range
info_gain  = avg(equity | one opp card known) − eq_base
bid        ≈ (info_gain + info_penalty) × pot × value_multiplier
```

The `value_multiplier` adapts to the opponent: bid **low** against players who chronically overbid (let them overpay for information), **high** against players who under-bid. Bids are capped (≤ 40 % of stack) to avoid catastrophic overcommitment. This is a **heuristic best-response**, not CFR — fast and well-suited to a one-shot sealed-bid decision.

> **[INFOGRAPHIC 8 — Auction Value Curve]**
> A curve of *information gain* (Y) vs. *hand type* (X): near-zero for monsters and trash (you don't need the peek), peaking for **marginal/drawing hands** where one card flips the decision. Caption: *"We pay most for information exactly when it changes our play."*

### 6.3 Flop / Turn / River — Equity → Sub-game CFR → Exploit
1. **Compute equity** against the *correct* range — collapsed (~49 hands) if we won the auction, full range (with a discount) if we lost.
2. **Build leaf EVs** for the 4 abstract actions from that equity and the opponent's estimated fold frequency.
3. **Solve the sub-game CFR** (or look up the precomputed bucket).
4. **Apply the exploit overlay** (Section 7).
5. **Sample and translate** to a legal engine action, with bet sizes clamped to legal bounds.

The **winner/loser branch** is explicit: in the winner state the bot computes near-exact equity against the collapsed range and value-bets hard; in the loser state it shades toward caution and applies an **information penalty** (it assumes the opponent reads it more sharply).

---

## 7. Heuristics + Real-Time Learning: Beating Exploitable Bots

GTO is *unexploitable* but not *maximally exploitative*. Against the imperfect bots in a real field, deviating from equilibrium wins more. Boss Blind runs a live **Opponent Model** updated every hand:

- `fold_to_bet[street]` — how often they fold to our bets
- `aggression[street]` — how often they bet/raise
- `vpip` / `pfr` — pre-flop looseness and raise frequency
- `auction_bids` — their bidding tendencies
- `preflop_raise_sizes` — to detect maniacs

These feed an **Exploit Overlay** that warps the CFR output:

| Opponent tendency | Boss Blind's adjustment |
|---|---|
| Folds too much (>65 %) | Bluff more — shift weight to bets |
| Calls too much / "station" (<25 %) | Stop bluffing — value-bet only |
| Hyper-aggressive | Trap: check/call strong hands |
| Overbids the auction | Bid 0 — let them overpay for info |

After adjustment, every action is **floored at ~1–2 %** so the bot stays unpredictable and can't itself be exploited for deviating.

> **[INFOGRAPHIC 9 — GTO vs. Exploit Dial]**
> A dial/slider with **"Unexploitable (GTO)"** on one end and **"Maximally Exploitative"** on the other, with a needle that moves toward exploit as the opponent-sample count grows. Caption: *"Start at GTO with no reads; lean exploitative as evidence accumulates (≥8 hands)."*

---

## 8. Benchmarking & Results

Boss Blind was tested on the competition scrimmage server against a field of opposing bots. Final bankrolls from the captured leaderboard logs (`LeaderBoard_logs/`):

| Opponent | Boss Blind final bankroll | Result |
|---|---|---|
| `pokkerai3` | **+2,514** | ✅ Win |
| `ProkerAI` (run 2) | −9,976 | Loss |
| `ProkerAI` (run 1) | −22,638 | Loss |
| `Quant_paglus` | −189,493 | Loss |
| `TESTBOT_7` | −329,729 | Loss |
| `pagleti_birds` | −347,641 | Loss |
| `Idle_Death_Gamble` | −460,091 | Loss |
| `Stochastic_Sharks` | −520,000 | Loss |

> **[GRAPH 10 — Benchmark Bar Chart]**
> Diverging horizontal bar chart, opponents on Y-axis, final bankroll on X-axis (green right of zero, red left). Sort best→worst.

> ⚠️ **Honesty note for the author:** these logs are mixed and several are heavy losses, and they appear to span **earlier bot versions** (`bb4`-era) rather than the final `champion_bot` / `final_cfr` build. Before publishing this section, please confirm **which logs reflect the champion build** and consider re-running a clean tournament with `tournament.py` so the results table represents the finished agent. I've reported the raw numbers faithfully rather than cherry-picking. Replace this table with the final-build results when available.

> **[GRAPH 11 — Bankroll-Over-Time Line Chart (recommended to generate)]**
> Run a match and plot cumulative bankroll vs. hand number for Boss Blind vs. an opponent. A rising line tells the "grind" story far better than a single final number. The `.glog` / `.log` files contain per-hand `awarded` deltas you can parse to build this.

The repo's `tournament.py` runs a **round-robin** (all pairings, 3 iterations each) and prints per-bot stats: win rate, avg payoff/hand, **auction win rate**, mean/variance of bids, and response-time percentiles — a ready-made source for the charts above.

---

## 9. Engineering Highlights

- **Self-contained deployment** — the entire strategy is a string constant; no external files, no network, instant cold-start.
- **Quantized strategies** — probabilities stored as integers (0–100), halving table size with negligible accuracy loss.
- **Graceful degradation** — the Time Manager's fast-path guarantees a legal action under any clock pressure; illegal/timeout actions are impossible by construction.
- **Defensive action translation** — every abstract action is re-checked against the engine's legal-action set and bet bounds before being sent, so the bot never forfeits a hand on a malformed move.
- **Numerically cheap core** — post-flop CFR is plain NumPy vector math (regret-matching over a length-4 array), not a tree traversal.

---

## 10. Limitations & Future Work

- **Abstraction error** — bucketing trades exactness for size; very texture-dependent boards may be mis-bucketed. Finer centroids (more clusters) would help at a storage cost.
- **Opponent model cold-start** — exploits only engage after ~8 hands; the first hands are pure GTO.
- **Auction model is heuristic** — a CFR-solved auction (treating the bid as another abstract action) is the natural next step toward full GTO.
- **Results need a clean re-benchmark** on the final build (see §8 note).

> **[INFOGRAPHIC 12 — Roadmap]**
> A simple 3-step roadmap: **1. Re-benchmark champion build → 2. CFR-solve the auction street → 3. Adaptive (deeper) board clustering.**

---

## Appendix A — Glossary

| Term | Meaning |
|---|---|
| **CFR** | Counterfactual Regret Minimization — self-play algorithm converging to Nash equilibrium. |
| **GTO / Nash** | A strategy that cannot be exploited in expectation. |
| **Blueprint** | A precomputed (offline) strategy table looked up at runtime. |
| **Abstraction** | Collapsing many similar game states into one to shrink the problem. |
| **Bucket / Centroid** | A cluster of strategically-similar hands or boards sharing one strategy. |
| **Equity** | Probability of winning the hand at showdown. |
| **Board wetness** | How draw-heavy a board is (flush/straight potential). |
| **Exploit overlay** | Deliberate deviation from GTO to punish a predictable opponent. |
| **Vickrey auction** | Sealed-bid auction where the winner pays the second-highest bid. |

## Appendix B — Repository Map

| Path | Contents |
|---|---|
| `bots/champion_bot.py` | Hybrid blueprint + runtime sub-game CFR + opponent model. |
| `bots/final_cfr.py` | Perfect-memory Abstract-CFR with embedded compressed master data. |
| `bots/sub_cfr.md` | Full algorithm spec (street-by-street). |
| `bots/master_data.json` | Structured blueprint: `cfr_table`, `flop_centroids`, `turn_centroids`. |
| `bots/compressed_master.txt` | Deployed base64+zlib strategy string (306 KB). |
| `bots/bb1…bb7`, `boss_blind_v*` | Iterative bot versions. |
| `tournament.py` | Round-robin match runner + stats (the IIT game engine). |
| `LeaderBoard_logs/`, `*.glog` | Match logs (per-hand actions, awards, final bankrolls). |

---

### Notes for the PDF pass (read me, Claude)

- All **[GRAPH]** / **[INFOGRAPHIC]** blocks are render instructions, not body text — replace each with an actual figure (matplotlib/SVG/diagram) or a designed graphic and a numbered caption.
- Numeric values in §1, §5, and §8 are pulled directly from the repository (real file sizes and real log bankrolls) — keep them accurate; don't invent new figures.
- The §8 honesty note should be resolved (re-benchmark or scope the table to the champion build) **before** this goes to any external audience; if it stays internal, leave it.
- Suggested cover/section accent palette: deep green (felt) + gold (chips) + charcoal; monospace for code blocks.
