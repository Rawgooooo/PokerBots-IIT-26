'''
Monte Carlo Equity Bot v2 — Sneak Peek Hold'em
Fixes: fast MC, aggressive heads-up play, adaptive auction, opponent modeling

CL1 Bot Logic: "Monte Carlo Equity Bot v2"
The 
cl1.py
 bot is a highly adaptive, math-based poker bot explicitly built for the Sneak Peek Hold'em variant. It relies heavily on probability math (equity and pot odds) and adapts its decision-making parameters based on the specific opponent it's playing against.

Here's a step-by-step walkthrough of how its brain works:

1. Instant Preflop Equity Evaluator (
_preflop_equity_fast
)
Simulating thousands of hands pre-flop takes a massive amount of time, which risks timing out and auto-folding.

To solve this, CL1 uses an instant preflop lookup table and formula:

Pocket Pairs: Pre-calculated win percentages exactly hardcoded (e.g., Aces = 85.3%, Twos = 57.5%).
Unpaired Cards: Uses a normalized math formula based on high cards, kickers, suit bonuses (if suited), and connectedness (if they are close in rank like 8-9).
This allows CL1 to know its exact win probability instantly, preserving the 20-second time bank strictly for post-flop play.

2. Dynamic Post-Flop Monte Carlo Simulations (
monte_carlo_equity
)
Once the community cards (flop, turn, river) arrive, the math gets complicated. CL1 runs a Monte Carlo Simulation, meaning it deals thousands of random possible opponent cards and remaining board cards to see how often it wins.

To respect the strict 20-second match limit:

CL1 uses a 
_sim_count
 budget. If it has plenty of time (>14 seconds left), it runs 100-150 simulations. If time is running out (<6 seconds), it drops to 40-50 simulations to play faster and avoid forced folding.
It caches the equity per betting round ("street"). If the opponent raises on the flop, CL1 doesn't recount its equity; it re-uses the simulation from earlier in the same street to save time.
3. Opponent Profiling & Adaptive Aggression (
_aggression_multiplier
)
CL1 tracks how its opponent plays over the course of the match:

opp_fold_count: How often the opponent folds before showdown.
opp_raise_count: How often the opponent raises.
Based on this, it determines an Aggression Multiplier (
aggr
):

Against Passive/Foldy Players (Fold > 50%): It multiplies its bets by 1.45x, bullying them and stealing pots heavily because it knows they'll fold.
Against Maniacs (Raise > 75%): It recognizes aggressive bullies (like Boss Blind) and shrinks its aggression multiplier to 0.80x, letting the opponent bloat the pot while CL1 only value-bets when its hand is strong.
4. The "Sneak Peek" Auction Strategy (
_compute_bid
)
The auction is the unique phase of this game where players bid to see one opponent hole card. Information is valuable, but overpaying for it is fatal. CL1 approaches this logically:

The "Always Overbid" Checker: It records the auction track record. If the opponent wins >88% of auctions, CL1 realizes it's fighting someone who bids their entire stack (like Boss Blind). It refuses to engage in an expensive bidding war and simply bids 1 chip (or 0 if tied), letting the opponent overpay for the info.
True Information Value: If the opponent is normal, CL1 calculates how much its equity shifts if it knows an opponent card. It runs a fast Monte Carlo simulation with a hypothetical known card, compares it to the unknown equity, and bids exactly that strict percentage difference (capped at a max of 6% of its stack).
5. The Core Decision Engine (
decide
 & 
_preflop_action
)
Every time CL1 has to fold, call, or raise, it strictly compares its Equity to its Pot Odds:

Pot Odds Formua: Cost To Call / (Pot Size + Cost To Call).
The Rule: If Equity > Pot Odds, it calls/raises. If Equity < Pot Odds, it folds.
Handling Massive Raises: If an opponent shoves thousands of chips (like Boss Blind does early on), pot odds naturally approach 50%. This is dangerous. CL1 enforces a strict +0.05 safety margin on top of pot odds against massive raises >800 chips. It doesn't flip coins; it only calls giant raises if it genuinely has a winning hand.
Bet Sizing: When it decides to bet, it scales its bet relative to the pot size (pot * 0.9 or pot * 0.5) multiplied by its adaptive aggression multiplier.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import math
import eval7

# ─────────────────────────────────────────────────────────────────────────────
#  PREFLOP EQUITY TABLE  (heads-up, instant lookup — no MC cost preflop)
# ─────────────────────────────────────────────────────────────────────────────

_RANK_ORDER = '23456789TJQKA'
_R = {r: i for i, r in enumerate(_RANK_ORDER)}

# Pocket pair HU equities vs random hand
_PAIR_EQ = {12:0.853, 11:0.823, 10:0.795, 9:0.770, 8:0.748,
             7:0.726,  6:0.704,  5:0.683, 4:0.661, 3:0.641,
             2:0.619,  1:0.597,  0:0.575}

def _preflop_equity_fast(c1: str, c2: str) -> float:
    """
    Instant preflop HU equity estimate.
    Pairs use known values; unpaired hands use a formula tuned to real equities.
    (AA=0.85, AKs=0.67, AKo=0.65, 72o=0.35)
    """
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    suited  = (s1 == s2)
    i1, i2  = _R[r1], _R[r2]
    if i1 < i2:
        i1, i2 = i2, i1

    if i1 == i2:
        return _PAIR_EQ.get(i1, 0.62)

    base     = 0.35 + (i1 / 12) * 0.25        # 0.35 (2x) → 0.60 (Ax)
    kicker   = (i2 / 12) * 0.08
    gap      = i1 - i2
    gap_pen  = min(gap - 1, 5) * 0.015 if gap > 1 else 0.0
    suit_bon = 0.025 if suited else 0.0
    conn_bon = 0.012 if gap == 1 else (0.007 if gap == 2 else 0.0)

    return max(0.33, min(0.70, base + kicker - gap_pen + suit_bon + conn_bon))


# ─────────────────────────────────────────────────────────────────────────────
#  MONTE CARLO EQUITY  (post-flop only)
# ─────────────────────────────────────────────────────────────────────────────

_FULL_DECK = [r + s for r in '23456789TJQKA' for s in 'dcsh']

def _deck_minus(excluded_set: set) -> list:
    return [eval7.Card(c) for c in _FULL_DECK if c not in excluded_set]

def monte_carlo_equity(
    hole_cards:  list,
    board_cards: list,
    opp_known:   list | None = None,
    n_sims: int = 150
) -> float:
    my_e7    = [eval7.Card(c) for c in hole_cards]
    board_e7 = [eval7.Card(c) for c in board_cards]
    opp_e7   = [eval7.Card(c) for c in opp_known] if opp_known else []

    excluded     = set(hole_cards + board_cards + (opp_known or []))
    deck         = _deck_minus(excluded)
    board_needed = 5 - len(board_e7)
    opp_needed   = 2 - len(opp_e7)

    wins = ties = 0.0
    for _ in range(n_sims):
        random.shuffle(deck)
        idx      = 0
        opp_hand = opp_e7 + deck[idx: idx + opp_needed]; idx += opp_needed
        run_board = board_e7 + deck[idx: idx + board_needed]

        my_val  = eval7.evaluate(my_e7    + run_board)
        opp_val = eval7.evaluate(opp_hand + run_board)

        if   my_val < opp_val:  wins += 1.0
        elif my_val == opp_val: ties += 0.5

    return (wins + ties) / n_sims


# ─────────────────────────────────────────────────────────────────────────────
#  BET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_raise(amount: int, cs: PokerState):
    lo, hi = cs.raise_bounds
    return ActionRaise(max(lo, min(hi, amount)))

def _bet_or_check(amount: int, cs: PokerState):
    if cs.can_act(ActionRaise):
        return _clamp_raise(amount, cs)
    return ActionCheck() if cs.can_act(ActionCheck) else ActionCall()


# ─────────────────────────────────────────────────────────────────────────────
#  CORE DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def decide(equity: float, cs: PokerState, aggr: float = 1.0):
    """
    Maps equity → action with pot-odds math + aggression scaling.
    aggr > 1.0 against passive/foldy opponents, < 1.0 against maniacs.
    """
    pot  = cs.pot
    ctc  = cs.cost_to_call          # cost to call

    # ── No bet to face → we open ─────────────────────────────────────────────
    if ctc == 0:
        if equity >= 0.63:
            return _bet_or_check(int(pot * 0.9 * aggr), cs)
        if equity >= 0.54:
            return _bet_or_check(int(pot * 0.5 * aggr), cs)
        if equity >= 0.49 and aggr > 1.1 and cs.can_act(ActionRaise):
            lo, _ = cs.raise_bounds
            return ActionRaise(lo)
        return ActionCheck() if cs.can_act(ActionCheck) else ActionFold()

    # ── Facing a bet ─────────────────────────────────────────────────────────
    pot_odds = ctc / (pot + ctc)

    if equity >= pot_odds + 0.20 and cs.can_act(ActionRaise):
        return _clamp_raise(int(pot * 0.85 * aggr), cs)

    if equity >= pot_odds + 0.10 and cs.can_act(ActionRaise):
        return _clamp_raise(int(pot * 0.5 * aggr), cs)

    if equity >= pot_odds - 0.04:
        # Avoid ActionCall if it is not legally available (though engine usually falls back, better safe)
        return ActionCall() if cs.can_act(ActionCall) else ActionFold()

    return ActionFold()


# ─────────────────────────────────────────────────────────────────────────────
#  PLAYER
# ─────────────────────────────────────────────────────────────────────────────

class Player(BaseBot):
    """
    Adaptive MC Equity Bot v2.

    Key improvements over v1:
    - Preflop: instant equity table (zero MC cost)
    - Post-flop MC: capped 40-150 sims, cached per street (no timeout risk)
    - Aggression adapts to opponent fold/raise frequencies
    - Auction: bids correctly scaled to chip count (fixes bug of illegal 1 bid when 0 chips)
    - Sensible pot-odds preflop call/fold against big raises
    """

    def __init__(self) -> None:
        # Opponent model counters
        self.opp_total_hands    = 0
        self.opp_fold_count     = 0     # we won without showdown = they folded
        self.opp_raise_count    = 0     # they raised at least once this hand
        self.auction_wins       = 0     # times WE won the auction (saw their card)
        self.auction_attempts   = 0

        # Per-round state
        self._opp_raised_this_hand = False
        self._we_won_auction       = False
        self._cached_street        = None
        self._cached_equity        = 0.5

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self._opp_raised_this_hand = False
        self._we_won_auction       = False
        self._cached_street        = None
        self._cached_equity        = 0.5

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff    = current_state.payoff
        opp_cards = current_state.opp_revealed_cards

        self.opp_total_hands += 1

        # Did they fold? (we won but no showdown)
        if payoff > 0 and len(opp_cards) < 2:
            self.opp_fold_count += 1

        if self._opp_raised_this_hand:
            self.opp_raise_count += 1

        self.auction_attempts += 1
        if self._we_won_auction:
            self.auction_wins += 1


    @property
    def opp_always_overbids(self) -> bool:
        if self.auction_attempts < 30:
            return False
        return (self.auction_wins / self.auction_attempts) < 0.12

    def _aggression_multiplier(self) -> float:
        if self.opp_total_hands < 15:
            return 1.0
        fold_rate  = self.opp_fold_count  / self.opp_total_hands
        raise_rate = self.opp_raise_count / self.opp_total_hands
        # Passive/foldy → be more aggressive and steal more
        if fold_rate > 0.50:
            return 1.45
        if fold_rate > 0.35:
            return 1.25
        # Maniac raising every hand → play tighter, only value bet
        if raise_rate > 0.75:
            return 0.80
        return 1.05

    @staticmethod
    def _sim_count(time_bank: float, street: str) -> int:
        """Budget sims so we stay well under the 20-second total limit."""
        base = {'flop': 100, 'turn': 120, 'river': 150}.get(street, 100)
        if time_bank > 14: return base
        if time_bank > 10: return max(70,  int(base * 0.70))
        if time_bank > 6:  return max(50,  int(base * 0.50))
        return 40

    # ── auction ───────────────────────────────────────────────────────────────

    def _compute_bid(self, hole_cards, board_cards, opp_known, pot, my_chips) -> int:
        # NEVER bid more than we have
        if my_chips <= 0:
            return 0
            
        # If opponent always outbids us anyway: bid 1 (or 0 if we have none)
        # But if we have chips, bid 1 so they pay 1 chip
        if self.opp_always_overbids:
            return min(1, my_chips)

        excluded = set(hole_cards + board_cards + (opp_known or []))
        possible = [c for c in _FULL_DECK if c not in excluded]
        if not possible:
            return min(1, my_chips)

        # Quick delta: how much does info shift our equity?
        eq_base = monte_carlo_equity(
            hole_cards, board_cards,
            opp_known if opp_known else None,
            n_sims=50
        )
        sample   = random.sample(possible, min(8, len(possible)))
        eq_known = sum(
            monte_carlo_equity(hole_cards, board_cards, (opp_known or []) + [c], n_sims=30)
            for c in sample
        ) / len(sample)

        info_delta = abs(eq_known - eq_base)
        # True-value bid capped at 6% of stack (info rarely worth more)
        bid = int(info_delta * pot)
        
        # Ensure bid is strictly between 0 and my_chips
        bid = min(bid, int(my_chips * 0.06))
        return min(max(0, bid), my_chips)

    # ── preflop ───────────────────────────────────────────────────────────────

    def _preflop_action(self, equity: float, cs: PokerState, aggr: float):
        pot  = cs.pot
        ctc  = cs.cost_to_call

        # Facing a raise
        if ctc > 0:
            pot_odds = ctc / (pot + ctc)
            
            # Against a massive raise (e.g. > 1500), pot odds naturally approach ~50%
            # If our equity is better than pot_odds, we should call.
            # But we require a STRICT margin of safety to avoid busting on marginal flips against a bully.
            if ctc > 800:
                if equity >= pot_odds + 0.05: # strict call against massive raises
                    if equity >= 0.65 and cs.can_act(ActionRaise):
                        return _clamp_raise(cs.my_chips, cs) # Shove right back with premiums
                    # Avoid ActionCall if it is not legally available (better safe)
                    return ActionCall() if cs.can_act(ActionCall) else ActionFold()
                return ActionFold()

            # Normal raises
            if equity < pot_odds - 0.02:
                return ActionFold()
            if equity >= pot_odds + 0.15 and cs.can_act(ActionRaise):
                return _clamp_raise(int(pot * 0.75 * aggr), cs)
            return ActionCall() if cs.can_act(ActionCall) else ActionFold()

        # We open (ctc == 0)
        if equity >= 0.60:
            if cs.can_act(ActionRaise):
                return _clamp_raise(max(60, int(pot * 3 * aggr)), cs)
            return ActionCall()

        if equity >= 0.50:
            if cs.can_act(ActionRaise):
                lo, _ = cs.raise_bounds
                return ActionRaise(lo)
            return ActionCall() if cs.can_act(ActionCall) else ActionFold()

        return ActionCheck() if cs.can_act(ActionCheck) else ActionFold()

    # ── main ──────────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street     = current_state.street
        hole       = current_state.my_hand
        board      = current_state.board
        opp_known  = current_state.opp_revealed_cards
        pot        = current_state.pot
        my_chips   = current_state.my_chips
        time_bank  = game_info.time_bank

        if current_state.cost_to_call > 0:
            self._opp_raised_this_hand = True

        # Detect auction win (we now have opp card info)
        if opp_known and not self._we_won_auction:
            self._we_won_auction = True

        aggr = self._aggression_multiplier()

        # ── AUCTION ──────────────────────────────────────────────────────────
        if street == 'auction':
            bid = self._compute_bid(hole, board, opp_known, pot, my_chips)
            return ActionBid(bid)

        # ── PREFLOP ──────────────────────────────────────────────────────────
        if street == 'pre-flop':
            equity = _preflop_equity_fast(hole[0], hole[1])
            return self._preflop_action(equity, current_state, aggr)

        # ── FLOP / TURN / RIVER ──────────────────────────────────────────────
        if self._cached_street != street:
            n = self._sim_count(time_bank, street)
            self._cached_equity = monte_carlo_equity(
                hole, board,
                opp_known if opp_known else None,
                n_sims=n
            )
            self._cached_street = street

        return decide(self._cached_equity, current_state, aggr)


if __name__ == '__main__':
    run_bot(Player(), parse_args())

