# Bot Algorithm — Complete Concise Summary

---

## Global State (persists across all streets each hand)

```
opponent_range:    set of possible opponent hands (updates after auction)
revealed_card:     the opponent card seen if auction won, else None
opp_has_info:      True if we lost auction (they know one of our cards)
opponent_model:    running statistics on opponent behavior across all rounds
time_manager:      tracks remaining budget, decides fast-path vs full compute
```

---

## Street 1 — Preflop

**Method**: Offline CFR (abstracted game, run once before competition)

**Offline (training)**:
- Abstract game tree: 4 nodes × 169 hand buckets × 3 stack depths
- Run vanilla CFR for 200,000 iterations
- Extract **average strategy** (not current — average is what converges to Nash)
- Compress → base64 → embed in bot.py as constant string

**Runtime**:
- Map hole cards to canonical bucket (e.g. AKs, 72o, JJp)
- Determine node type (SB_OPEN / BB_VS_RAISE / SB_VS_3BET etc.)
- Determine stack depth bucket (deep / mid / short)
- Lookup action probabilities from embedded table
- Sample action from that distribution (mixed strategy — not deterministic)
- Apply exploit overlay nudge if opponent model has enough data (≥10 hands)

**Output**: ActionFold / ActionCall / ActionRaise(amount)

---

## Street 2 — Auction

**Method**: Closed-form formula (not CFR)

**Compute**:
```
eq_win  = MC equity vs full range (120 sims) — value if I win
penalty = 0.06  — equity loss from opponent knowing one of my cards
eq_lose = eq_win - penalty

delta = eq_win - eq_lose
bid   = clamp(delta × pot × 1.4, 0, my_chips)
```
The 1.4× overrealization multiplier accounts for the fact that knowing
an opponent card lets you play more accurately → your equity overrealizes.

**After auction resolves — update global state**:
```
if opp_revealed_hands is non-empty:
    revealed_card = opp_revealed_hands[0]
    opponent_range = all hands containing revealed_card (excludes hole + board)
    # ~49 possible opponent hands instead of ~1326
else:
    opp_has_info = True
    opponent_range = full range (all hands not in hole + board)
    # apply 0.94 equity discount throughout rest of hand
```

**Output**: ActionBid(amount)

---

## Streets 3, 4, 5 — Flop / Turn / River

**Same pipeline for all three streets, different equity method per street.**

### Step 1: Compute Equity (using opponent_range)

```
Flop  (3 board cards): Monte Carlo, 120 sims, sample from opponent_range
Turn  (4 board cards): Exact enumeration via eval7
                       known card  → ~1,900 combos  (~1ms)
                       unknown     → ~45,000 combos (~5ms)
River (5 board cards): Exact enumeration via eval7
                       known card  → ~44 combos     (<1ms)
                       unknown     → ~1,035 combos  (<1ms)
```

If `opp_has_info`: multiply final equity by 0.94 (they play sharper against us).

### Step 2: Build Leaf EVs for Subgame Tree

Tree has 4 hero actions: fold / check-call / bet-half / bet-pot.
Each maps to terminal EV using the equity computed above:

```
EV(fold)        = 0
EV(check-call)  = equity × pot
EV(bet-half)    = opp_fold_freq × pot  +  (1 - opp_fold_freq) × equity × (pot + 0.5×pot)
EV(bet-pot)     = opp_fold_freq × pot  +  (1 - opp_fold_freq) × equity × (pot + pot)
```

`opp_fold_freq` comes from opponent_model. Default 0.5 until 8+ hands observed.

### Step 3: Runtime CFR on 12-node Tree

```
iterations: 400 (flop) / 500 (turn) / 300 (river)
regrets:    numpy array shape (4,)
strategy:   regret-matching over 4 actions

for i in range(iterations):
    strategy = regret_match(regrets)          # pure numpy
    cfvs     = strategy · leaf_evs            # dot product
    ev       = sum(strategy × cfvs)
    regrets += cfvs - ev
    strategy_sum += strategy

final = strategy_sum / strategy_sum.sum()     # average strategy → Nash for this subgame
```

### Step 4: Exploit Overlay

Adjust the CFR output based on observed opponent patterns:

```
opp folds too much  (>65%): shift weight toward bet-half and bet-pot (bluff more)
opp calls too much  (<30%): shift weight toward check-call (never bluff)
opp is very aggro:          shift toward check-call with strong hands (trap)
opp is passive:             shift toward betting (deny free cards)

After adjustment: renormalize, floor each action at 2% (stay unpredictable)
```

### Step 5: Convert to Engine Action

Sample from final adjusted probabilities → map to ActionFold / ActionCheck /
ActionCall / ActionRaise(amount clamped to raise_bounds).

---

## Time Management

```
Budget per round  = time_bank / rounds_remaining

Allocation:
  Preflop:  0.1ms   (lookup only)
  Auction:  1ms     (MC + formula)
  Flop:     40%     of per_round_budget
  Turn:     45%     of per_round_budget
  River:    10%     of per_round_budget

Fast-path trigger: if per_round_budget < 8ms
  → skip CFR, use pure equity threshold decision
  → fold if equity < pot_odds - 0.05, else call, raise if equity > 0.65
```

---

## Opponent Model (updates in on_hand_end every round)

```
Tracks per street:
  fold_to_bet[street]    → how often they fold when we bet
  aggression[street]     → how often they bet/raise
  vpip                   → how often they voluntarily enter pot preflop
  pfr                    → how often they raise preflop
  auction_bid_fracs      → their bids as fraction of chips (are they always overbidding?)

Used in:
  Preflop blueprint:  nudge toward tighter/looser opens based on their VPIP
  Auction formula:    if they always overbid, bid 0 (let them overpay for info)
  CFR leaf EVs:       opp_fold_freq drives EV(bet-half) and EV(bet-pot)
  Exploit overlay:    bluff/value frequency adjustments post-flop
```

---

## Complete Time Budget (per round, steady state)

```
Preflop:            0.1ms
Auction:            1.0ms
Flop equity (MC):   3.0ms
Flop CFR:           2.0ms
Turn equity:        5.0ms  (1ms if card known)
Turn CFR:           1.5ms
River equity:       0.5ms
River CFR:          0.5ms
Overhead:           1.0ms
─────────────────────────
Total:             ~14ms   (well within 20ms/round budget)
```

---

## What Each Component Is (Precisely)

| Street | Component | Technical Name |
|---|---|---|
| Preflop | Embedded table | Offline abstracted-game CFR |
| Auction | Bid formula | Heuristic best response |
| Flop equity | 120 random samples | Monte Carlo simulation (not MCCFR) |
| Turn/River equity | Full enumeration | Exact exhaustive enumeration |
| Post-flop decisions | 12-node runtime solve | Depth-limited subgame CFR |
| Adaptation | Fold/aggression stats | Exploitative best response |
