"""
champion_bot.py — IIT Pokerbots 2026 (SneakPeek Hold'em)
Hybrid CFR + Real-Time Learning (RTL) Expert System Bot
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import numpy as np
import random
import zlib
import base64
import json
import time
import os
from itertools import combinations

# Load embedded blueprint dynamically
DIR = os.path.dirname(os.path.abspath(__file__))
BP_PATH = os.path.join(DIR, "preflop_blueprint.b64")
if os.path.exists(BP_PATH):
    with open(BP_PATH, "rb") as f:
        _PREFLOP_BLUEPRINT_B64 = f.read()
else:
    _PREFLOP_BLUEPRINT_B64 = b""

try:
    _bp = json.loads(zlib.decompress(base64.b64decode(_PREFLOP_BLUEPRINT_B64)))
    PREFLOP_STRATEGY = _bp['strategy']
    PREFLOP_EQUITY   = _bp['equity']
except Exception:
    PREFLOP_STRATEGY = {}
    PREFLOP_EQUITY   = {}

NUM_ROUNDS     = 1000
STARTING_STACK = 5000
BIG_BLIND      = 20
SMALL_BLIND    = 10
S_PREFLOP = 'pre-flop'
S_AUCTION = 'auction'
S_FLOP    = 'flop'
S_TURN    = 'turn'
S_RIVER   = 'river'

A_FOLD       = 0
A_CHECK_CALL = 1
A_BET_HALF   = 2
A_BET_POT    = 3
N_ACTIONS    = 4

CFR_ITERS = {S_FLOP: 400, S_TURN: 500, S_RIVER: 300}
OPP_MIN_SAMPLES = 8
AUCTION_INFO_PENALTY = 0.15

_ALL_CARDS = [f"{r}{s}" for r in '23456789TJQKA' for s in 'shdc']
_RANK_VAL = {r: i + 2 for i, r in enumerate('23456789TJQKA')}

def hand_to_bucket(c1: str, c2: str) -> str:
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    v1, v2 = _RANK_VAL[r1], _RANK_VAL[r2]
    if v1 < v2:
        r1, r2, s1, s2 = r2, r1, s2, s1
        v1, v2 = v2, v1
    if v1 == v2:
        return f"{r1}{r2}p"
    return f"{r1}{r2}s" if s1 == s2 else f"{r1}{r2}o"

def _stack_bucket(eff_chips: int) -> str:
    bb = eff_chips / BIG_BLIND
    return 'deep' if bb > 150 else ('mid' if bb > 50 else 'short')

def _preflop_key(node: str, bucket: str, stack: str) -> str:
    return f"{node}|{bucket}|{stack}"

def _remaining(hole, board, opp_known=None):
    ex = set(hole) | set(board)
    if opp_known:
        ex.add(opp_known)
    return [c for c in _ALL_CARDS if c not in ex]

def mc_equity(hole, board, opp_known=None, n=120) -> float:
    rem = _remaining(hole, board, opp_known)
    board_needed = 5 - len(board)
    opp_need = 1 if opp_known else 2
    sample_size = opp_need + board_needed
    hole_e7 = [eval7.Card(c) for c in hole]
    wins = 0.0
    for _ in range(n):
        s = random.sample(rem, sample_size)
        if opp_known:
            opp_str, run_str = [opp_known, s[0]], s[1:]
        else:
            opp_str, run_str = s[:2], s[2:]
        board_e7 = [eval7.Card(c) for c in board + run_str]
        opp_e7 = [eval7.Card(c) for c in opp_str]
        my_sc = eval7.evaluate(hole_e7 + board_e7)
        op_sc = eval7.evaluate(opp_e7 + board_e7)
        if my_sc > op_sc: wins += 1.0
        elif my_sc == op_sc: wins += 0.5
    return wins / n

def exact_equity_turn(hole, board, opp_known=None) -> float:
    if opp_known:
        rem = _remaining(hole, board, opp_known)
        hole_e7, bd_e7, ok_e7 = [eval7.Card(c) for c in hole], [eval7.Card(c) for c in board], eval7.Card(opp_known)
        wins, total = 0.0, 0
        for opp2 in rem:
            o2e7 = eval7.Card(opp2)
            rpool = [c for c in rem if c != opp2]
            for rv in rpool:
                fb = bd_e7 + [eval7.Card(rv)]
                my_sc, op_sc = eval7.evaluate(hole_e7 + fb), eval7.evaluate([ok_e7, o2e7] + fb)
                wins += 1.0 if my_sc > op_sc else (0.5 if my_sc == op_sc else 0.0)
                total += 1
        return wins / total if total > 0 else 0.5
    else: return mc_equity(hole, board, opp_known=None, n=300)

def exact_equity_river(hole, board, opp_known=None) -> float:
    rem = _remaining(hole, board, opp_known)
    hole_e7, bd_e7 = [eval7.Card(c) for c in hole], [eval7.Card(c) for c in board]
    wins, total = 0.0, 0
    if opp_known:
        ok_e7 = eval7.Card(opp_known)
        for opp2 in rem:
            o2e7 = eval7.Card(opp2)
            my_sc, op_sc = eval7.evaluate(hole_e7 + bd_e7), eval7.evaluate([ok_e7, o2e7] + bd_e7)
            wins += 1.0 if my_sc > op_sc else (0.5 if my_sc == op_sc else 0.0)
            total += 1
    else:
        for (c1, c2) in combinations(rem, 2):
            c1e7, c2e7 = eval7.Card(c1), eval7.Card(c2)
            my_sc, op_sc = eval7.evaluate(hole_e7 + bd_e7), eval7.evaluate([c1e7, c2e7] + bd_e7)
            wins += 1.0 if my_sc > op_sc else (0.5 if my_sc == op_sc else 0.0)
            total += 1
    return wins / total if total > 0 else 0.5

def get_equity(hole, board, opp_known=None) -> float:
    n = len(board)
    if n == 3: return mc_equity(hole, board, opp_known, n=120)
    elif n == 4: return exact_equity_turn(hole, board, opp_known)
    elif n == 5: return exact_equity_river(hole, board, opp_known)
    else:
        if len(hole) == 2: return PREFLOP_EQUITY.get(hand_to_bucket(hole[0], hole[1]), 0.5)
        return 0.5

def build_leaf_evs(equity: float, pot: float, opp_fold_freq: float) -> np.ndarray:
    ff = float(np.clip(opp_fold_freq, 0.05, 0.95))
    evs = np.zeros(N_ACTIONS)
    evs[A_FOLD] = 0.0
    evs[A_CHECK_CALL] = equity * pot
    evs[A_BET_HALF] = ff * pot + (1.0 - ff) * (equity * 1.5 * pot - 0.5 * pot)
    evs[A_BET_POT] = ff * pot + (1.0 - ff) * (equity * 2.0 * pot - 1.0 * pot)
    return evs

def run_cfr(leaf_evs: np.ndarray, iterations: int, legal: list) -> np.ndarray:
    mask = np.zeros(N_ACTIONS)
    for a in legal: mask[a] = 1.0
    n_legal = len(legal)
    regrets, s_sum = np.zeros(N_ACTIONS), np.zeros(N_ACTIONS)
    for _ in range(iterations):
        pos = np.maximum(regrets * mask, 0.0)
        total = pos.sum()
        strat = pos / total if total > 1e-10 else mask / n_legal
        ev = float(np.dot(strat, leaf_evs))
        regrets += (leaf_evs - ev) * mask
        s_sum += strat
    tot = s_sum.sum()
    return s_sum / tot if tot > 1e-10 else mask / n_legal

class OpponentModel:
    def __init__(self):
        self.fold_to_bet = {S_FLOP: [], S_TURN: [], S_RIVER: []}
        self.aggression = {S_FLOP: [], S_TURN: [], S_RIVER: []}
        self.vpip, self.pfr, self.auction_bids, self.preflop_raise_sizes = [], [], [], []

    def record_fold(self, street: str):
        if street in self.fold_to_bet: self.fold_to_bet[street].append(True)

    def record_call(self, street: str):
        if street in self.fold_to_bet: self.fold_to_bet[street].append(False)

    def record_aggression(self, street: str, is_aggro: bool):
        if street in self.aggression: self.aggression[street].append(is_aggro)

    def fold_freq(self, street: str) -> float:
        d = self.fold_to_bet.get(street, [])
        return float(np.mean(d)) if len(d) >= OPP_MIN_SAMPLES else 0.40

    def is_aggro(self, street: str) -> bool:
        d = self.aggression.get(street, [])
        return float(np.mean(d)) > 0.55 if len(d) >= OPP_MIN_SAMPLES else False

    def is_station(self, street: str) -> bool:
        d = self.fold_to_bet.get(street, [])
        return float(np.mean(d)) < 0.25 if len(d) >= OPP_MIN_SAMPLES else False

    def is_maniac_preflop(self) -> bool:
        if len(self.preflop_raise_sizes) < 3: return False
        return float(np.mean(self.preflop_raise_sizes)) > 20.0

    def average_auction_bid(self) -> float:
        if len(self.auction_bids) < 3: return -1.0
        return float(np.mean(self.auction_bids))

    def apply_exploit(self, probs: np.ndarray, street: str, equity: float) -> np.ndarray:
        ff, p = self.fold_freq(street), probs.copy()
        if ff > 0.65:
            boost = min(0.30, (ff - 0.50) * 0.8)
            p[A_BET_HALF] += boost * 0.5; p[A_BET_POT] += boost * 0.5
            p[A_FOLD] -= boost * 0.7; p[A_CHECK_CALL] -= boost * 0.3
        elif self.is_station(street):
            cut = min(0.40, (0.40 - ff) * 0.9)
            p[A_BET_HALF] -= cut * 0.7; p[A_BET_POT] -= cut * 0.9
            if equity < 0.70: p[A_CHECK_CALL] += cut * 0.8; p[A_BET_HALF] -= cut * 0.1
            else: p[A_CHECK_CALL] += cut * 0.2; p[A_BET_POT] += cut * 0.3
        
        if self.is_aggro(street) and equity > 0.70:
            p[A_CHECK_CALL] += 0.20; p[A_BET_HALF] -= 0.10; p[A_BET_POT] -= 0.10

        p = np.maximum(p, 0.01)
        return p / p.sum()

class TimeManager:
    def __init__(self): self.rounds_left = NUM_ROUNDS
    def start_round(self, round_num: int): self.rounds_left = max(1, NUM_ROUNDS - round_num + 1)
    def fast_path(self, time_bank: float) -> bool: return (time_bank / self.rounds_left) < 0.010
    def cfr_iters(self, street: str, time_bank: float) -> int:
        base = CFR_ITERS.get(street, 400)
        return base // 2 if (time_bank / self.rounds_left) < 0.015 else base

def _can(state, cls) -> bool: return cls in state.legal_actions
def _safe_raise(state, target: int):
    min_raise, max_raise = state.raise_bounds if hasattr(state, 'raise_bounds') else (state.cost_to_call, state.my_chips)
    if not _can(state, ActionRaise): return ActionCall() if _can(state, ActionCall) else ActionCheck()
    return ActionRaise(max(min_raise, min(max_raise, int(target))))

def _fast_decide(equity: float, state):
    ctc = state.cost_to_call
    pot_odds = ctc / (state.pot + ctc) if ctc > 0 else 0.0
    commitment = ctc / state.my_chips if state.my_chips > 0 else 0.0
    
    if ctc > 0:
        if equity < pot_odds + 0.05: return ActionFold() if _can(state, ActionFold) else ActionCheck()
        if commitment > 0.30 and equity < 0.65: return ActionFold() if _can(state, ActionFold) else ActionCheck()

    if equity > 0.75 and _can(state, ActionRaise): return _safe_raise(state, state.my_wager + int(state.pot * 0.6))
    if _can(state, ActionCall): return ActionCall()
    if _can(state, ActionCheck): return ActionCheck()
    return ActionFold()

def _cfr_to_action(probs: np.ndarray, state):
    chosen, pot = int(np.random.choice(N_ACTIONS, p=probs)), state.pot
    if chosen == A_FOLD: return ActionFold() if _can(state, ActionFold) else ActionCheck()
    if chosen == A_CHECK_CALL:
        if _can(state, ActionCheck): return ActionCheck()
        if _can(state, ActionCall): return ActionCall()
        return ActionFold()
    if chosen == A_BET_HALF: return _safe_raise(state, state.my_wager + int(0.5 * pot))
    if chosen == A_BET_POT: return _safe_raise(state, state.my_wager + int(1.0 * pot))
    return ActionCheck()

def _legal_abstract(state) -> list:
    legal = [A_CHECK_CALL]
    if _can(state, ActionFold): legal.append(A_FOLD)
    if _can(state, ActionRaise): legal += [A_BET_HALF, A_BET_POT]
    return legal

class Player(BaseBot):
    def __init__(self):
        self.opp_model, self.time_mgr = OpponentModel(), TimeManager()
        self._reset_hand_state()

    def _reset_hand_state(self):
        self.hole, self.revealed_opp_card, self.opp_has_info, self.auction_done = [], None, False, False
        self._opp_vpip, self._opp_pfr, self._preflop_raises, self._we_bet_street, self._last_opp_wager = False, False, 0, {}, 0

    def on_hand_start(self, game_info, state):
        self._reset_hand_state()
        self.hole = list(state.my_hand)
        self.time_mgr.start_round(game_info.round_num)

    def on_hand_end(self, game_info, state):
        self.opp_model.vpip.append(self._opp_vpip)
        self.opp_model.pfr.append(self._opp_pfr)

    def get_move(self, game_info, state):
        street, time_bank = state.street, game_info.time_bank
        self._check_auction_info(state)

        if state.cost_to_call > 0 and not self._opp_vpip and street == S_PREFLOP:
            if state.opp_wager > BIG_BLIND: self._opp_pfr = True
            self._opp_vpip = True

        if street == S_PREFLOP and state.opp_wager > BIG_BLIND:
            self.opp_model.preflop_raise_sizes.append(state.opp_wager / BIG_BLIND)

        if street in self._we_bet_street:
            prev_opp_w, curr_opp_w = self._we_bet_street[street], state.opp_wager
            if state.cost_to_call == 0 and curr_opp_w != prev_opp_w:
                self.opp_model.record_call(street); del self._we_bet_street[street]
            elif state.cost_to_call == 0 and curr_opp_w == prev_opp_w: del self._we_bet_street[street]
            elif curr_opp_w > prev_opp_w: self.opp_model.record_aggression(street, True)

        if self.time_mgr.fast_path(time_bank):
            eq = get_equity(self.hole, list(state.board), self.revealed_opp_card)
            if self.opp_has_info: eq *= (1.0 - AUCTION_INFO_PENALTY)
            return _fast_decide(eq, state)

        if street == S_PREFLOP: return self._preflop(state)
        if street == S_AUCTION: return self._auction(state)
        if street in (S_FLOP, S_TURN, S_RIVER): return self._postflop(state, game_info)
        return ActionCheck() if _can(state, ActionCheck) else ActionCall()

    def _check_auction_info(self, state):
        if self.auction_done: return
        if state.street in (S_FLOP, S_TURN, S_RIVER):
            self.auction_done = True
            if state.opp_revealed_cards: self.revealed_opp_card = state.opp_revealed_cards[0]
            else: self.opp_has_info = True

    def _preflop(self, state):
        if len(self.hole) < 2: return ActionCheck() if _can(state, ActionCheck) else ActionFold()
        bucket, ctc, is_bb = hand_to_bucket(self.hole[0], self.hole[1]), state.cost_to_call, state.is_bb
        eq = PREFLOP_EQUITY.get(bucket, 0.5)

        if self.opp_model.is_maniac_preflop():
            if eq < 0.70: return ActionFold() if _can(state, ActionFold) else ActionCheck()
            target = state.my_wager + ctc + int(state.pot * 0.8)
            if _can(state, ActionRaise): return _safe_raise(state, target)
            if _can(state, ActionCall): return ActionCall()
        
        if ctc > 0 and state.my_chips > 0:
            commitment = ctc / state.my_chips
            if ctc > state.pot * 1.5 and eq < (ctc / (state.pot + ctc)) + 0.05: return ActionFold() if _can(state, ActionFold) else ActionCheck()
            if commitment > 0.40 and eq < 0.65: return ActionFold() if _can(state, ActionFold) else ActionCheck()
            if commitment > 0.20 and eq < 0.55: return ActionFold() if _can(state, ActionFold) else ActionCheck()
            if commitment > 0.10 and eq < 0.48: return ActionFold() if _can(state, ActionFold) else ActionCheck()

        if ctc == 0 or (not is_bb and ctc == SMALL_BLIND): node = 'BB_VS_LIMP' if is_bb else 'SB_OPEN'
        elif not is_bb and ctc > BIG_BLIND: node = 'SB_VS_3BET'
        else: node = 'BB_VS_RAISE'

        strat = PREFLOP_STRATEGY.get(_preflop_key(node, bucket, _stack_bucket(min(state.my_chips, state.opp_chips))))
        if not strat: return _fast_decide(eq, state)

        names = ['fold', 'call', 'raise_2x', 'raise_4x']
        probs = np.array([max(0.0, strat.get(a, 0.0)) for a in names])
        if probs.sum() < 1e-6: return _fast_decide(eq, state)
        chosen = np.random.choice(names, p=probs / probs.sum())

        if chosen == 'fold': return ActionFold() if _can(state, ActionFold) else ActionCheck()
        if chosen == 'call' or self._preflop_raises >= 2:
            if _can(state, ActionCheck): return ActionCheck()
            if _can(state, ActionCall): return ActionCall()
            return ActionFold()

        target = state.my_wager + ctc + int(ctc * 1.5) if ctc > BIG_BLIND * 2 else state.my_wager + (2 if chosen == 'raise_2x' else 4) * BIG_BLIND
        max_raise = state.my_chips if eq >= 0.74 else int(state.my_chips * 0.75) if eq >= 0.60 else int(state.my_chips * 0.35)
            
        if target - state.my_wager > max_raise:
            if _can(state, ActionCall): return ActionCall()
            if _can(state, ActionCheck): return ActionCheck()
            return ActionFold()

        self._preflop_raises += 1
        if _can(state, ActionRaise): return _safe_raise(state, target)
        if _can(state, ActionCall): return ActionCall()
        if _can(state, ActionCheck): return ActionCheck()
        return ActionFold()

    def _auction(self, state):
        board, pot, my_chips = list(state.board), float(state.pot), state.my_chips
        eq_base, rem = mc_equity(self.hole, board, opp_known=None, n=80), _remaining(self.hole, board)
        info_gain = max(0.0, float(np.mean([mc_equity(self.hole, board, opp_known=c, n=40) for c in random.sample(rem, min(6, len(rem)))])) - eq_base) if len(rem) >= 4 else 0.05
        
        opp_avg_bid = self.opp_model.average_auction_bid()
        value_mul = 0.8 if opp_avg_bid > state.my_chips * 0.30 else 1.6 if 0 <= opp_avg_bid < pot * 0.5 else 1.3
        
        bid = max(int(opp_avg_bid + 2) if 0 <= opp_avg_bid < pot * 1.5 else int(pot * 0.5), int((info_gain + AUCTION_INFO_PENALTY) * pot * value_mul))
        
        self.opp_model.auction_bids.append(pot - state.pot)
        
        return ActionBid(max(0, min(bid, int(my_chips * 0.40), my_chips)))

    def _postflop(self, state, game_info):
        street, board, pot, ctc = state.street, list(state.board), float(state.pot), state.cost_to_call
        equity = float(np.clip((get_equity(self.hole, board, self.revealed_opp_card) * (1.0 - AUCTION_INFO_PENALTY)) if self.opp_has_info else get_equity(self.hole, board, self.revealed_opp_card), 0.0, 1.0))
        
        if ctc > pot * 1.5 and equity < (ctc / (state.pot + ctc)) + 0.05: return ActionFold() if _can(state, ActionFold) else ActionCheck()

        final = self.opp_model.apply_exploit(run_cfr(build_leaf_evs(equity, pot, self.opp_model.fold_freq(street)), self.time_mgr.cfr_iters(street, game_info.time_bank), _legal_abstract(state)), street, equity)
        action = _cfr_to_action(final, state)

        if isinstance(action, ActionRaise): self._we_bet_street[street] = state.opp_wager
        return action

if __name__ == '__main__':
    run_bot(Player(), parse_args())
