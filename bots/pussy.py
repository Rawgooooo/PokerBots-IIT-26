'''
CFR Poker Bot V3 — Geometric Aggression & Info Exploitation
Layer 1: Precomputed Preflop Blueprint (Offline CFR)
Layer 2: Shark Auction Logic (Bid for Info dominance)
Layer 3: Geometric Post-Flop Engine (Exponential pot growth, overbets)
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import eval7
import numpy as np
import zlib
import base64
import json
from itertools import combinations

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

NUM_ROUNDS = 1000
STARTING_STACK = 5000
BIG_BLIND = 20

# Increased simulation counts slightly for thinner value margins
MC_SIMS_FLOP = 150
MC_SIMS_AUCTION = 80
MC_SIMS_PREFLOP = 40
MC_SIMS_TURN_FALLBACK = 150

RANKS = '23456789TJQKA'
RANK_VAL = {r: i+2 for i, r in enumerate(RANKS)}

# Preflop tree node types
NODE_SB_OPEN     = 'SB_OPEN'
NODE_BB_VS_LIMP  = 'BB_VS_LIMP'
NODE_BB_VS_RAISE = 'BB_VS_RAISE'
NODE_SB_VS_3BET  = 'SB_VS_3BET'

# ══════════════════════════════════════════════════════════════════════════════
# PREFLOP BLUEPRINT (Keep existing data)
# ══════════════════════════════════════════════════════════════════════════════

try:
    _PREFLOP_BLUEPRINT_B64 = ""
    _PREFLOP_DATA = json.loads(zlib.decompress(base64.b64decode(_PREFLOP_BLUEPRINT_B64)))
    PREFLOP_BLUEPRINT = _PREFLOP_DATA['strategy']
    PREFLOP_EQUITY = _PREFLOP_DATA['equity']
except Exception:
    PREFLOP_BLUEPRINT = {}
    PREFLOP_EQUITY = {}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def hand_to_bucket(card1_str, card2_str):
    r1, s1 = card1_str[0], card1_str[1]
    r2, s2 = card2_str[0], card2_str[1]
    v1, v2 = RANK_VAL[r1], RANK_VAL[r2]
    if v1 < v2:
        r1, r2, s1, s2, v1, v2 = r2, r1, s2, s1, v2, v1
    if v1 == v2:
        return f"{r1}{r2}p"
    elif s1 == s2:
        return f"{r1}{r2}s"
    else:
        return f"{r1}{r2}o"

def make_state_key(node, bucket, stack_bucket):
    return f"{node}|{bucket}|{stack_bucket}"

def _build_deck():
    return [eval7.Card(r+s) for r in RANKS for s in 'shdc']

def mc_equity(hole, board, opp_known_card=None, n=100):
    deck = _build_deck()
    excluded = set(str(c) for c in hole + board + ([opp_known_card] if opp_known_card else []))
    remaining = [c for c in deck if str(c) not in excluded]

    wins = 0.0
    for _ in range(n):
        needed = 2 + (5 - len(board))
        if len(remaining) < needed: break
        sample = random.sample(remaining, needed)
        
        if opp_known_card:
            opp_hand = [opp_known_card, sample[0]]
            runout = sample[1:]
        else:
            opp_hand = sample[:2]
            runout = sample[2:]

        full_board = board + runout
        my_score = eval7.evaluate(hole + full_board)
        opp_score = eval7.evaluate(opp_hand + full_board)

        if my_score > opp_score: wins += 1.0
        elif my_score == opp_score: wins += 0.5

    return wins / max(n, 1)

def exact_equity_turn(hole, board, opp_known_card=None):
    if not opp_known_card:
        return mc_equity(hole, board, opp_known_card=None, n=MC_SIMS_TURN_FALLBACK)
    
    deck = _build_deck()
    excluded = set(str(c) for c in hole + board + [opp_known_card])
    remaining = [c for c in deck if str(c) not in excluded]
    
    wins = 0.0
    total = 0
    for opp2 in remaining:
        river_pool = [c for c in remaining if str(c) != str(opp2)]
        for river in river_pool:
            opp_hand = [opp_known_card, opp2]
            full_board = board + [river]
            my_score = eval7.evaluate(hole + full_board)
            opp_score = eval7.evaluate(opp_hand + full_board)
            wins += 1 if my_score > opp_score else (0.5 if my_score == opp_score else 0)
            total += 1
    return wins / total if total > 0 else 0.5

def exact_equity_river(hole, board, opp_known_card=None):
    deck = _build_deck()
    excluded = set(str(c) for c in hole + board + ([opp_known_card] if opp_known_card else []))
    remaining = [c for c in deck if str(c) not in excluded]

    wins = 0.0
    total = 0
    if opp_known_card:
        for opp2 in remaining:
            my_score = eval7.evaluate(hole + board)
            opp_score = eval7.evaluate([opp_known_card, opp2] + board)
            wins += 1 if my_score > opp_score else (0.5 if my_score == opp_score else 0)
            total += 1
    else:
        for (c1, c2) in combinations(remaining, 2):
            my_score = eval7.evaluate(hole + board)
            opp_score = eval7.evaluate([c1, c2] + board)
            wins += 1 if my_score > opp_score else (0.5 if my_score == opp_score else 0)
            total += 1
    return wins / total if total > 0 else 0.5

# ══════════════════════════════════════════════════════════════════════════════
# OPPONENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class OpponentModel:
    def __init__(self):
        self.auction_bids = []
        self.fold_count = 0
        self.call_count = 0
        self.raise_count = 0
        self.total_actions = 0
        self.hands_played = 0

    def record_auction_bid(self, bid):
        self.auction_bids.append(bid)

    def avg_auction_bid(self, default=200):
        if len(self.auction_bids) < 3: return default
        return np.mean(self.auction_bids)

    def fold_frequency(self, default=0.45):
        total = self.fold_count + self.call_count + self.raise_count
        if total < 5: return default
        return self.fold_count / total

    def is_aggro(self, threshold=0.35):
        total = self.fold_count + self.call_count + self.raise_count
        if total < 8: return False
        return self.raise_count / total > threshold

# ══════════════════════════════════════════════════════════════════════════════
# PLAYER BOT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Player(BaseBot):
    '''V3 Bot: Geometric Sizing & Asymmetric Info Warfare'''

    def __init__(self) -> None:
        self.pussy_out = False
        self.opp_model = OpponentModel()
        self.revealed_opp_card = None
        self.opp_has_info = False
        self.auction_won = False
        self.auction_done = False
        self.this_hand_we_bet = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        if game_info.bankroll > (999 - game_info.round_number)*15 + 5000:
            self.pussy_out = True
        self.revealed_opp_card = None
        self.opp_has_info = False
        self.auction_won = False
        self.auction_done = False
        self.this_hand_we_bet = False
        self.opp_model.hands_played += 1

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        my_delta = current_state.payoff
        street = current_state.street
        if self.this_hand_we_bet and my_delta > 0 and street not in ('river',):
            self.opp_model.fold_count += 1
        elif self.this_hand_we_bet and my_delta <= 0:
            self.opp_model.call_count += 1

    def _update_auction_info(self, current_state):
        if self.auction_done: return
        street = current_state.street
        if street in ('turn', 'river'):
            self.auction_done = True
            if current_state.opp_revealed_cards and len(current_state.opp_revealed_cards) > 0:
                self.revealed_opp_card = eval7.Card(current_state.opp_revealed_cards[0])
                self.auction_won = True
            else:
                self.opp_has_info = True

    def _get_equity(self, hole, board, opp_known_card=None):
        n_board = len(board)
        if n_board == 3: return mc_equity(hole, board, opp_known_card, n=MC_SIMS_FLOP)
        elif n_board == 4: return exact_equity_turn(hole, board, opp_known_card)
        elif n_board == 5: return exact_equity_river(hole, board, opp_known_card)
        else: return mc_equity(hole, board, opp_known_card, n=MC_SIMS_PREFLOP)

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 1: PREFLOP (UNCHANGED)
    # ──────────────────────────────────────────────────────────────────────────

    def _preflop_action(self, state, game_info):
        hand = state.my_hand
        bucket = hand_to_bucket(str(hand[0]), str(hand[1]))
        ctc = state.cost_to_call
        
        if state.is_bb:
            node = NODE_BB_VS_LIMP if ctc == 0 else NODE_BB_VS_RAISE
        else:
            node = NODE_SB_OPEN if ctc <= BIG_BLIND else NODE_SB_VS_3BET

        eff_stack = min(state.my_chips, state.opp_chips)
        bb_depth = eff_stack / BIG_BLIND
        stack_bucket = 'deep' if bb_depth > 150 else ('mid' if bb_depth > 50 else 'short')

        key = make_state_key(node, bucket, stack_bucket)
        strat = PREFLOP_BLUEPRINT.get(key)
        eq = PREFLOP_EQUITY.get(bucket, 0.5)

        if strat is None: return self._equity_preflop_decision(eq, state)

        actions = ['fold', 'call', 'raise_2x', 'raise_4x']
        probs = np.array([strat.get(a, 0.0) for a in actions])
        if probs.sum() <= 0: return self._equity_preflop_decision(eq, state)
        probs /= probs.sum()
        chosen = np.random.choice(actions, p=probs)

        if state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
        else:
            lo, hi = 0, 0

        if chosen == 'fold':
            return ActionFold() if state.can_act(ActionFold) else ActionCheck()
        elif chosen == 'call':
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionCall() if state.can_act(ActionCall) else ActionFold()
        elif chosen == 'raise_2x':
            if state.can_act(ActionRaise):
                return ActionRaise(max(lo, min(hi, 2 * BIG_BLIND + state.my_wager)))
            return ActionCall()
        elif chosen == 'raise_4x':
            if state.can_act(ActionRaise):
                return ActionRaise(max(lo, min(hi, 4 * BIG_BLIND + state.my_wager)))
            return ActionCall()
        return ActionCheck()

    def _equity_preflop_decision(self, eq, state):
        ctc = state.cost_to_call
        if eq > 0.60 and state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
            return ActionRaise(max(lo, min(hi, int(state.my_wager + 3.5 * BIG_BLIND))))
        elif eq > 0.45: # Slightly looser
            if state.can_act(ActionCall): return ActionCall()
            return ActionCheck()
        else:
            return ActionCheck() if state.can_act(ActionCheck) else ActionFold()

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 2: SHARK AUCTION (BID AGGRESSIVELY FOR INFO)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_auction_bid(self, hole, board, pot, my_chips):
        eq = mc_equity(hole, board, opp_known_card=None, n=MC_SIMS_AUCTION)
        
        # Base info value: Knowing a card is worth more than just equity shift.
        # It's the ability to perfect-bluff and perfect-value-bet.
        # We increase the base valuation multiplier.
        delta = 0.12 + 0.08 * (1.0 - abs(eq - 0.5) * 2) 
        true_value = int(2.5 * delta * pot) # Increased multiplier from 2.0 to 2.5
        
        opp_avg = self.opp_model.avg_auction_bid(default=200)

        # Shark logic:
        # If we have a very strong draw (eq ~0.5) or strong hand (eq > 0.6),
        # we WANT the info to stack them.
        aggression_factor = 1.0
        if 0.4 < eq < 0.7:
            aggression_factor = 1.3 # Pay premium for info in swingy spots

        if opp_avg < 50:
            bid = max(true_value, int(opp_avg * 2.0) + 10)
        elif opp_avg > 800:
            # Let them bleed chips for info if they overpay drastically
            bid = 0 
        else:
            bid = int(true_value * aggression_factor)

        # Loose cap: Allow up to 35% of stack if pot is already big relative to stack
        stack_cap = 0.35 if (pot/my_chips > 0.2) else 0.25
        bid = min(bid, int(my_chips * stack_cap))
        
        return max(1, min(bid, my_chips))

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 3: GEOMETRIC POST-FLOP ENGINE
    # ──────────────────────────────────────────────────────────────────────────

    def _get_geometric_size(self, pot, stack, streets_left):
        """
        Calculate bet size such that if we bet this % of pot on every remaining street,
        we get all-in exactly by the river.
        Formula: B = Pot * ( (SPR + 1)^(1/N) - 1 )
        """
        if streets_left <= 0: return stack
        spr = stack / max(1, pot)
        growth_factor = (spr + 1) ** (1 / streets_left) - 1
        bet_size = int(pot * growth_factor)
        # Cap min bet at 33% pot to avoid weak sizing, max at 150% (unless river)
        bet_size = max(int(pot*0.33), bet_size)
        if streets_left > 1:
            bet_size = min(int(pot*1.5), bet_size)
        return bet_size

    def _postflop_action(self, equity, state):
        pot = state.pot
        ctc = state.cost_to_call
        my_chips = state.my_chips
        streets_left = {'flop': 3, 'turn': 2, 'river': 1}[state.street]
        
        # ─── CASE 1: ASYMMETRIC WARFARE (WE KNOW THEIR CARD) ───
        if self.auction_won:
            # We have perfect info. DO NOT PLAY GTO. Play specific exploit.
            
            # Subcase A: We have them crushed.
            if equity >= 0.85:
                if state.can_act(ActionRaise):
                    lo, hi = state.raise_bounds
                    # Geometric growth to get stacks in
                    geo_bet = self._get_geometric_size(pot + ctc, my_chips, streets_left)
                    # If on river, just JAM
                    if streets_left == 1: geo_bet = my_chips
                    
                    amt = max(lo, min(hi, int(state.my_wager + geo_bet + ctc)))
                    self.this_hand_we_bet = True
                    return ActionRaise(amt)
                return ActionCall()

            # Subcase B: We are crushed.
            if equity <= 0.20:
                # Fold immediately. Don't pay.
                if state.can_act(ActionCheck): return ActionCheck()
                return ActionFold()

            # Subcase C: Middleware / Draws (0.20 < eq < 0.85)
            # Play aggressive semi-bluffs if we have decent equity
            if equity > 0.50:
                if ctc == 0 and state.can_act(ActionRaise):
                    # Value bet
                    lo, hi = state.raise_bounds
                    amt = max(lo, min(hi, int(state.my_wager + pot * 0.6)))
                    self.this_hand_we_bet = True
                    return ActionRaise(amt)
                return ActionCall()
            
            # Marginal call
            pot_odds = ctc / (pot + ctc)
            if equity > pot_odds + 0.05: return ActionCall()
            return ActionFold()

        # ─── CASE 2: STANDARD PLAY (UNKNOWN CARD) ───
        
        # 1. THINNED VALUE THRESHOLDS
        # We lower thresholds to extract from weaker hands.
        val_thresh_strong = 0.75
        val_thresh_thin   = 0.55 # Bet thin!
        
        # 2. FACING BET
        if ctc > 0:
            pot_odds = ctc / (pot + ctc)
            # Safety margin: Be tighter calling bets than making them
            required_equity = pot_odds + (0.05 if not self.opp_has_info else 0.10)
            
            if equity >= required_equity:
                # Raise for value?
                if equity > 0.80 and state.can_act(ActionRaise):
                     # Trapping logic: if opp is aggro, just call strong hands occasionally
                    if self.opp_model.is_aggro() and random.random() < 0.3:
                        return ActionCall()
                    
                    lo, hi = state.raise_bounds
                    # Raise big
                    amt = max(lo, min(hi, int(state.my_wager + pot * 1.0)))
                    self.this_hand_we_bet = True
                    return ActionRaise(amt)
                return ActionCall()
            
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        # 3. WE ARE AGGRESSOR (Check/Bet)
        if state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
            
            # A. Monster Hand (Overbet / Geometric)
            if equity > 0.85:
                # River Overbet: Polarize range
                if streets_left == 1: 
                    bet = int(pot * 1.5) # 150% pot overbet
                else:
                    bet = self._get_geometric_size(pot, my_chips, streets_left)
                
                amt = max(lo, min(hi, int(state.my_wager + bet)))
                self.this_hand_we_bet = True
                return ActionRaise(amt)

            # B. Strong Value
            if equity > val_thresh_strong:
                bet = int(pot * 0.75)
                amt = max(lo, min(hi, int(state.my_wager + bet)))
                self.this_hand_we_bet = True
                return ActionRaise(amt)

            # C. Thin Value / Protection
            if equity > val_thresh_thin:
                bet = int(pot * 0.40) # Smaller sizing for thin value
                amt = max(lo, min(hi, int(state.my_wager + bet)))
                self.this_hand_we_bet = True
                return ActionRaise(amt)

            # D. Polarized Bluffing
            # Bluff if low equity (draws) but not zero, and opp folds > 40%
            if (equity < 0.35 and equity > 0.15 
                and self.opp_model.fold_frequency() > 0.40
                and not self.opp_has_info
                and random.random() < 0.20):
                
                # Bluff Big (Polarized) - 80% pot
                bet = int(pot * 0.80)
                amt = max(lo, min(hi, int(state.my_wager + bet)))
                self.this_hand_we_bet = True
                return ActionRaise(amt)

        return ActionCheck()

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        
        if current_state.can_act(ActionFold) and self.pussy_out:
            return ActionFold()
            
        street = current_state.street
        hand = current_state.my_hand
        board = current_state.board
        pot = current_state.pot
        my_chips = current_state.my_chips

        hole = [eval7.Card(str(c)) for c in hand]
        board_cards = [eval7.Card(str(c)) for c in board]

        self._update_auction_info(current_state)

        if street == 'pre-flop':
            return self._preflop_action(current_state, game_info) if current_state.can_act(self._preflop_action(current_state, game_info)) else ActionFold()

        if street == 'auction':
            bid = self._compute_auction_bid(hole, board_cards, pot, my_chips)
            return ActionBid(bid) if current_state.can_act(ActionBid(bid)) else ActionFold()

        equity = self._get_equity(hole, board_cards, self.revealed_opp_card)
        return self._postflop_action(equity, current_state) if current_state.can_act(self._postflop_action(equity, current_state)) else ActionFold()

if __name__ == '__main__':
    run_bot(Player(), parse_args())