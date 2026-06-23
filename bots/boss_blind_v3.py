'''
Sneak Peek Hold'em bot: Chen pre-flop, EV-of-info auction, tabular post-flop,
opponent profiling, vectorized 169 equity, tie exploitation, Fast-Fold under 2s.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import math

# --- Constants ---
RANKS = '23456789TJQKA'
NUM_ROUNDS = 1000
FAST_FOLD_TIME = 2.0
TIE_BID_FRACTION = 0.10  # round 10% of pot for tie exploitation


def chen_score(card1: str, card2: str) -> float:
    """Chen Formula: pairs, suitedness, gaps. Zero-latency pre-flop strength."""
    val_map = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5, '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5, '4': 2, '3': 1.5, '2': 1}
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    idx1, idx2 = RANKS.index(r1), RANKS.index(r2)
    if idx1 < idx2:
        idx1, idx2 = idx2, idx1
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    score = val_map[r1]
    is_pair = r1 == r2
    if is_pair:
        score = max(5, score * 2)
    if s1 == s2:
        score += 2
    if not is_pair:
        gap = idx1 - idx2 - 1
        if gap == 1:
            score -= 1
        elif gap == 2:
            score -= 2
        elif gap == 3:
            score -= 4
        elif gap >= 4:
            score -= 5
        if gap in (0, 1) and idx1 < RANKS.index('Q') and idx2 < RANKS.index('Q'):
            score += 1
    return math.ceil(score)


def hand_to_index(card1: str, card2: str) -> int:
    """Map two cards to canonical 0..168 hand index (pairs, suited, offsuit)."""
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
    # 13 pairs done; then suited (13..90), then offsuit (91..168)
    # suited: (high, low) with high>low: (12,11),(12,10),...,(12,0),(11,10),...,(1,0) -> 12+11+...+1 = 78
    # index for (h,l) suited: sum over k from 0 to (12-h) of (h-1-k) for h>l -> simpler: row h has h choices for l
    # 0-indexed: high 12->0, low 11->0. Pair (12,11) = 0,0 -> suited 0 = 13. (12,10)=13+1, (11,10)=13+12+0
    # Suited: idx = 13 + (12-high)*(12-high+1)//2 + (high - low - 1)  for high>low
    # Actually standard: 13 pairs, then AKs,AQs,...,A2s,KQs,...,32s = 78 suited, then 78 offsuit.
    # Suited (h,l): number of pairs (H,L) with H>L in order: (12,11),(12,10),...,(1,0). (12,11)=0 -> 13. (12,10)=1->14.
    # So (high,low) in 0..12, high>low: offset = (11-high)*(12-high)//2 + (high-low-1) for suited
    # 11-high = number of high ranks above us; (11-high)*(12-high)/2 = count of (H,L) with H>high. Then (high-low-1) for same high.
    n_above = (11 - high) * (12 - high) // 2
    suited_idx = 13 + n_above + (high - low - 1)
    offsuit_idx = 91 + n_above + (high - low - 1)
    return suited_idx if suited else offsuit_idx


def build_169_strengths():
    """Pre-computed strength per 169 hand type (from Chen). Used for vectorized equity."""
    strengths = [0.0] * 169
    # Pairs: indices 0..12 (AA=0, 22=12). Chen pairs: max(5, 2*val). AA=20, KK=16, ..., 22=5.
    for i in range(13):
        rank = RANKS[12 - i]
        chen = max(5, 2 * {'A':10,'K':8,'Q':7,'J':6,'T':5,'9':4.5,'8':4,'7':3.5,'6':3,'5':2.5,'4':2,'3':1.5,'2':1}[rank])
        strengths[i] = float(chen)
    # Suited 13..90: (high, low) high from 12 down, low from high-1 down to 0.
    for high in range(12, 0, -1):
        for low in range(high - 1, -1, -1):
            idx = 13 + (11 - high) * (12 - high) // 2 + (high - low - 1)
            r1, r2 = RANKS[high], RANKS[low]
            chen = chen_score(r1 + 'c', r2 + 'c')
            strengths[idx] = float(chen)
    # Offsuit 91..168
    for high in range(12, 0, -1):
        for low in range(high - 1, -1, -1):
            idx = 91 + (11 - high) * (12 - high) // 2 + (high - low - 1)
            r1, r2 = RANKS[high], RANKS[low]
            chen = chen_score(r1 + 'c', r2 + 'd')
            strengths[idx] = float(chen)
    return strengths


# One-time pre-computed hand strengths for 169 hand types (vectorized equity).
_HAND_STRENGTHS = build_169_strengths()


def equity_vs_range(my_idx: int, opp_indices: list[int]) -> float:
    """Instant lookup: win probability vs opponent range. E[strength_me/(strength_me+strength_opp)]."""
    if not opp_indices:
        return 0.5
    s_me = _HAND_STRENGTHS[my_idx]
    total = 0.0
    for j in opp_indices:
        s_opp = _HAND_STRENGTHS[j]
        total += s_me / (s_me + s_opp) if (s_me + s_opp) > 0 else 0.5
    return total / len(opp_indices)


def indices_containing_card(card: str) -> list[int]:
    """All 169 hand indices that contain the given card (for pruning after seeing one opp card)."""
    out = []
    seen = set()
    for r in range(13):
        for s in range(4):
            c2 = RANKS[r] + 'cdhs'[s]
            if c2 == card:
                continue
            idx = hand_to_index(card, c2)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out if out else list(range(169))


def board_wetness(board: list[str]) -> float:
    """Wet board (flush/straight draws) -> higher info value. Return multiplier for delta_equity ~0.1--0.25."""
    if len(board) < 3:
        return 0.15
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


class Player(BaseBot):
    def __init__(self) -> None:
        self.log_data = []
        self.opp_vpip_this_round = False
        self.opp_raised_this_round = False
        self.window = 50
        self.opp_class = "Medium"
        self.auction_bids_ours = []
        self.auction_wins = 0
        self.auction_rounds = 0

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.opp_raised_this_round = False
        self.opp_vpip_this_round = False
        if len(self.log_data) >= 20 and game_info.round_num % 25 == 1:
            self._update_opponent_profile()

    def _update_opponent_profile(self) -> None:
        window = min(self.window, len(self.log_data))
        if window == 0:
            return
        recent = self.log_data[-window:]
        vpip_rate = sum(1 for r in recent if r.get('vpip')) / float(window)
        if vpip_rate > 0.7:
            self.opp_class = "High Risk"
        elif vpip_rate < 0.2:
            self.opp_class = "Low Risk"
        else:
            self.opp_class = "Medium"

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff
        opp_cards = current_state.opp_revealed_cards
        they_won = payoff < 0
        they_folded = payoff > 0 and len(opp_cards) < 2
        self.auction_rounds += 1
        if len(opp_cards) > 0:
            self.auction_wins += 1
        self.log_data.append({
            'vpip': self.opp_vpip_this_round,
            'they_raised': self.opp_raised_this_round,
            'they_won': they_won,
            'they_folded': they_folded,
        })

    def _pot_odds(self, state: PokerState) -> float:
        cost = state.cost_to_call
        if cost <= 0:
            return 0.0
        total = state.pot + cost
        return cost / float(total) if total > 0 else 1.0

    def _make_bet(self, amount: float, state: PokerState):
        """Valid bet: non-negative integer, within raise_bounds, not exceeding opponent stack."""
        if not state.can_act(ActionRaise):
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()
        min_raise, max_raise = state.raise_bounds
        bet = max(0, int(amount))
        bet = max(min_raise, min(max_raise, bet))
        return ActionRaise(bet)

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street = current_state.street
        time_bank = game_info.time_bank
        # Fast-Fold mode when time bank < 2 seconds
        fast_fold = time_bank < FAST_FOLD_TIME

        if current_state.cost_to_call > 0:
            self.opp_raised_this_round = True
        if street == 'pre-flop':
            opp_forced = BIG_BLIND if current_state.is_bb else SMALL_BLIND
            opp_in = STARTING_STACK - current_state.opp_chips
            if opp_in > opp_forced:
                self.opp_vpip_this_round = True

        # --- Pre-flop: Chen. Raise >12, Call >8, else Fold/Check ---
        if street == 'pre-flop':
            chen = chen_score(current_state.my_hand[0], current_state.my_hand[1])
            if fast_fold:
                if current_state.can_act(ActionCheck):
                    return ActionCheck()
                if current_state.can_act(ActionCall):
                    return ActionCall()
                return ActionFold()
            if chen > 12:
                if current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    return ActionRaise(min_r)
                if current_state.can_act(ActionCall):
                    return ActionCall()
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()
            if chen > 8:
                if current_state.can_act(ActionCall):
                    return ActionCall()
                if current_state.can_act(ActionCheck):
                    return ActionCheck()
                return ActionFold()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        # --- Auction: Bid = Pot * delta_equity; tie hack; exploit low bidders ---
        if street == 'auction':
            pot = current_state.pot
            delta_eq = board_wetness(current_state.board)
            raw_bid = pot * delta_eq
            # Tie exploitation: round to psychological 10% of pot to sometimes tie (both pay, both see card)
            tie_bid = max(0, int(pot * TIE_BID_FRACTION))
            if abs(raw_bid - tie_bid) <= pot * 0.05:
                bid = tie_bid
            else:
                bid = int(round(raw_bid))
            # Opponent profiling: if we often win at low bids, they bid low -> bid slightly higher to secure info
            if self.auction_rounds >= 10 and self.auction_wins / max(1, self.auction_rounds) > 0.6:
                bid = min(current_state.my_chips, int(bid * 1.15) + 1)
            bid = max(0, min(current_state.my_chips, bid))
            self.auction_bids_ours.append(bid)
            return ActionBid(bid)

        # Track auction result (we don't see opp bid; we see if we won by having revealed cards next street)
        # So we update auction_rounds/auction_wins in get_move when we see opp_revealed_cards on flop/turn/river.

        # --- Post-flop: tabular decision matrix; prune range if we have opp card(s) ---
        my_idx = hand_to_index(current_state.my_hand[0], current_state.my_hand[1])
        opp_indices = list(range(169))
        if len(current_state.opp_revealed_cards) == 1:
            opp_indices = indices_containing_card(current_state.opp_revealed_cards[0])
        elif len(current_state.opp_revealed_cards) == 2:
            opp_indices = [hand_to_index(current_state.opp_revealed_cards[0], current_state.opp_revealed_cards[1])]
        if not opp_indices:
            opp_indices = list(range(169))
        equity = equity_vs_range(my_idx, opp_indices)
        pot_odds = self._pot_odds(current_state)

        if fast_fold:
            if equity < pot_odds and current_state.cost_to_call > 0 and current_state.can_act(ActionFold):
                return ActionFold()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()

        # Decision matrix by street (flop/turn/river) and strength
        strong = equity > 0.55
        medium = 0.45 <= equity <= 0.55
        weak = equity < 0.45

        if current_state.cost_to_call > 0 and equity < max(0.25, pot_odds - 0.05):
            if current_state.can_act(ActionFold):
                return ActionFold()
        if strong and (equity > pot_odds + 0.10 or current_state.cost_to_call == 0):
            base = current_state.pot + current_state.cost_to_call + BIG_BLIND
            amt = int(equity * base)
            return self._make_bet(amt, current_state)
        if medium and current_state.cost_to_call == 0 and current_state.can_act(ActionRaise):
            min_r, _ = current_state.raise_bounds
            return ActionRaise(min_r)
        if weak and current_state.cost_to_call > 0:
            if current_state.can_act(ActionFold):
                return ActionFold()
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        if current_state.can_act(ActionCall):
            return ActionCall()
        return ActionFold()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
