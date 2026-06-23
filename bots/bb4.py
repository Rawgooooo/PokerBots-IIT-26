'''
Boss Blind V4 (BB4) - "The Apex Predator"
Ultimate Sneak Peek Hold'em Bot.
Features:
- Anti-Vampire Auction Warfare: Traps trackers, steals ties.
- Polarized Sizing: Bets standard buckets (e.g. 75% pot) with both nuts and bluffs 
  to prevent equity reverse-engineering.
- 169-Hand Pre-computed Matrix for zero-latency baseline equity.
- Opponent Fold-to-CBet and Bid tracking.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import math
import eval7

# --- Constants ---
RANKS = '23456789TJQKA'
FAST_FOLD_TIME = 2.0

# --- Pre-computation for Zero Latency Equity Estimates ---
def chen_score(card1: str, card2: str) -> float:
    val_map = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    idx1, idx2 = RANKS.index(r1), RANKS.index(r2)
    if idx1 < idx2:
        idx1, idx2 = idx2, idx1
        r1, r2 = r2, r1
    score = val_map[r1]
    is_pair = r1 == r2
    if is_pair:
        score = max(5, score * 2)
    if s1 == s2:
        score += 2
    if not is_pair:
        gap = idx1 - idx2 - 1
        if gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        if gap in (0, 1) and idx1 < RANKS.index('Q') and idx2 < RANKS.index('Q'):
            score += 1
    return math.ceil(score)

def hand_to_index(card1: str, card2: str) -> int:
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    if i1 < i2:
        i1, i2 = i2, i1
        s1, s2 = s2, s1
    high, low = i1, i2
    suited = s1 == s2
    if high == low:
        return 12 - high
    n_above = (11 - high) * (12 - high) // 2
    suited_idx = 13 + n_above + (high - low - 1)
    offsuit_idx = 91 + n_above + (high - low - 1)
    return suited_idx if suited else offsuit_idx

def build_169_strengths():
    strengths = [0.0] * 169
    for i in range(13):
        rank = RANKS[12 - i]
        strengths[i] = float(max(5, 2 * {'A':10,'K':8,'Q':7,'J':6,'T':5,'9':4.5,'8':4,'7':3.5,'6':3,'5':2.5,'4':2,'3':1.5,'2':1}[rank]))
    for high in range(12, 0, -1):
        for low in range(high - 1, -1, -1):
            idx = 13 + (11 - high) * (12 - high) // 2 + (high - low - 1)
            strengths[idx] = float(chen_score(RANKS[high] + 'c', RANKS[low] + 'c'))
    for high in range(12, 0, -1):
        for low in range(high - 1, -1, -1):
            idx = 91 + (11 - high) * (12 - high) // 2 + (high - low - 1)
            strengths[idx] = float(chen_score(RANKS[high] + 'c', RANKS[low] + 'd'))
    return strengths

_HAND_STRENGTHS = build_169_strengths()

def equity_vs_range(my_idx: int, opp_indices: list[int]) -> float:
    if not opp_indices: return 0.5
    s_me = _HAND_STRENGTHS[my_idx]
    total = 0.0
    for j in opp_indices:
        s_opp = _HAND_STRENGTHS[j]
        total += s_me / (s_me + s_opp) if (s_me + s_opp) > 0 else 0.5
    return total / len(opp_indices)

def indices_containing_card(card: str) -> list[int]:
    out = []
    seen = set()
    for r in range(13):
        for s in range(4):
            c2 = RANKS[r] + 'cdhs'[s]
            if c2 == card: continue
            idx = hand_to_index(card, c2)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out if out else list(range(169))

def board_wetness(board: list[str]) -> float:
    if len(board) < 3: return 0.15
    ranks = [RANKS.index(c[0]) for c in board]
    suits = [c[1] for c in board]
    from collections import Counter
    sc = Counter(suits)
    max_suit = max(sc.values()) if sc else 0
    r_sorted = sorted(ranks)
    gaps = [r_sorted[i+1] - r_sorted[i] for i in range(len(r_sorted)-1)]
    connected = sum(1 for g in gaps if g <= 2)
    flush_draw = 1.0 if max_suit >= 2 else 0.0
    straight_draw = 0.5 * (connected >= 2)
    return 0.10 + 0.08 * flush_draw + 0.07 * straight_draw

# --- Fast Post-Flop Monte Carlo for Critical Spots ---
def estimate_hand_equity_mc(my_cards, board, opp_known, iterations=30):
    if iterations <= 0: return 0.0
    known_cards = set(eval7.Card(c) for c in (my_cards + board + opp_known))
    deck = [card for card in (eval7.Card(r+s) for r in RANKS for s in 'cdhs') if card not in known_cards]
    hero = [eval7.Card(c) for c in my_cards]
    brd = [eval7.Card(c) for c in board]
    opp_k = [eval7.Card(c) for c in opp_known]
    rem_b = 5 - len(brd)
    rem_o = 2 - len(opp_k)
    wins = ties = 0
    for _ in range(iterations):
        samp = random.sample(deck, rem_b + rem_o)
        opp_c = opp_k + samp[:rem_o]
        full_b = brd + samp[rem_o:]
        hs = eval7.evaluate(hero + full_b)
        os = eval7.evaluate(opp_c + full_b)
        if hs > os: wins += 1
        elif hs == os: ties += 1
    return (wins + 0.5 * ties) / iterations


class Player(BaseBot):
    def __init__(self) -> None:
        self.opp_vpip = 0
        self.opp_folds = 0
        self.opp_auction_bids = []
        self.hands_played = 0
        self.trap_mode = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        # Auction Trap Mode: 10% of the time, we bid 0 to break the opponent's tracker
        self.trap_mode = random.random() < 0.10

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff
        opp_cards = current_state.opp_revealed_cards
        if payoff > 0 and len(opp_cards) < 2:
            self.opp_folds += 1

    def _pot_odds(self, state: PokerState) -> float:
        cost = state.cost_to_call
        if cost <= 0: return 0.0
        tot = state.pot + cost
        return cost / float(tot) if tot > 0 else 1.0

    def _make_bet(self, amount: int, state: PokerState):
        if not state.can_act(ActionRaise):
            if state.can_act(ActionCall): return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()
        min_r, max_r = state.raise_bounds
        bet = max(min_r, min(max_r, int(amount)))
        return ActionRaise(bet)

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street = current_state.street
        time_bank = game_info.time_bank
        fast_fold = time_bank < FAST_FOLD_TIME

        # --- PRE-FLOP ---
        if street == 'pre-flop':
            chen = chen_score(current_state.my_hand[0], current_state.my_hand[1])
            if fast_fold:
                return ActionCheck() if current_state.can_act(ActionCheck) else (ActionCall() if current_state.can_act(ActionCall) else ActionFold())
            
            # Polarized Pre-Flop 3-Betting:
            # We raise with top hands (Chen > 11) AND sometimes with decent suited connectors/gappers (Chen 7-9) to disguise range
            if chen > 11 or (chen >= 7 and random.random() < 0.15):
                if current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    # Default to a pot-sized raise, or 3x the cost to call
                    raise_amt = min_r
                    if current_state.cost_to_call > 0:
                        raise_amt = current_state.cost_to_call * 3 + current_state.pot
                    return self._make_bet(raise_amt, current_state)
                
            if chen > 7.5:
                if current_state.cost_to_call > 1500 and chen < 10: 
                    return ActionFold() # Avoid getting trapped by Maniacs if hand is mid
                return ActionCall() if current_state.can_act(ActionCall) else ActionCheck()

            return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()

        # --- AUCTION WARFARE ---
        if street == 'auction':
            pot = current_state.pot
            
            # Subvert BB3's tie logic. If bot tries to bid 10% of pot to tie, we bid 12% to steal it!
            # If we are in Trap mode, we bid 0.
            if self.trap_mode:
                return ActionBid(0)
                
            delta_eq = board_wetness(current_state.board)
            raw_bid = pot * delta_eq
            
            # BB3 uses exactly 10% ties. We bid 11.5% to just beat their tie threshold safely.
            anti_tie_bid = int(pot * 0.115) + 1
            
            bid = max(int(raw_bid), anti_tie_bid)
            
            # If the opponent is BB2 (always 1) or BB3 (tracked), bid firmly but below the danger zone
            bid = min(current_state.my_chips, bid)
            
            # Absolute max cap is 25% of our stack so we never mutually self-destruct against BB1
            bid = min(bid, int(current_state.my_chips * 0.25))
            
            # Rule: bid must be at least 10 if we have the chips
            return ActionBid(min(max(10, bid), current_state.my_chips))

        # --- POST-FLOP ---
        # Get baseline equity instantly
        my_idx = hand_to_index(current_state.my_hand[0], current_state.my_hand[1])
        opp_indices = list(range(169))
        if len(current_state.opp_revealed_cards) == 1:
            opp_indices = indices_containing_card(current_state.opp_revealed_cards[0])
        elif len(current_state.opp_revealed_cards) == 2:
            opp_indices = [hand_to_index(current_state.opp_revealed_cards[0], current_state.opp_revealed_cards[1])]
            
        equity = equity_vs_range(my_idx, opp_indices)
        
        # If the hand is deep and we have time, calculate real MC equity since 169-grid is an approximation
        if not fast_fold and current_state.pot > 1000 and street in ['turn', 'river']:
            equity = estimate_hand_equity_mc(current_state.my_hand, current_state.board, current_state.opp_revealed_cards, iterations=40)

        pot_odds = self._pot_odds(current_state)

        # Defense Mechanism
        if current_state.cost_to_call > 0 and equity < max(0.25, pot_odds - 0.05):
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

        # POLARIZED BET SIZING
        # Instead of `int(equity * base)` which leaks our exact hand strength,
        # we always be 33% pot, 75% pot, or 120% pot.
        if equity > 0.65 or (equity < 0.35 and random.random() < 0.10):
            # Nuts or strict Bluff -> Bet 75% or 120% pot
            target_pot = current_state.pot + current_state.cost_to_call
            fraction = random.choice([0.75, 1.20])
            amt = int(target_pot * fraction)
            return self._make_bet(amt, current_state)
            
        if equity > 0.50:
            # Value bet -> Bet 33% pot
            target_pot = current_state.pot + current_state.cost_to_call
            amt = int(target_pot * 0.33)
            return self._make_bet(amt, current_state)

        # Passive path
        if current_state.cost_to_call == 0:
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()
        
        if current_state.cost_to_call > 0 and equity >= pot_odds - 0.05:
            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

        return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
