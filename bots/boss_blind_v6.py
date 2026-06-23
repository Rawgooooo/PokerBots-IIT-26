'''
Simple example pokerbot, written in Python.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random

# Card utilities for Chen formula and Monte Carlo-style equity estimation
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
FULL_DECK = [r + s for r in RANKS for s in SUITS]
RANK_TO_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}


class Player(BaseBot):
    '''
    A pokerbot.
    '''

    def __init__(self) -> None:
        '''
        Called when a new game starts. Called exactly once.
        '''
        # Chen thresholds
        self.preflop_fold_threshold = 6          # below this Chen score we fold pre-flop when facing a bet
        self.high_chen_threshold = 10           # above this Chen score we can run high-bet experiments

        # Equity decision threshold (for P(us) vs P(opp))
        self.equity_diff_threshold = 0.08

        # Auction control
        self.base_auction_min = 2000
        self.base_auction_max = 2500
        self.adjusted_auction_bounds = False
        self.auction_rounds = 0
        self.auction_wins = 0

        # Per-round bookkeeping
        self.round_logs: dict[int, dict] = {}

        # High-bet experiment tracking (first 25 high-Chen rounds)
        self.high_bet_experiments = 0
        self.high_bet_max_rounds = 25
        self.high_bet_hero_wins = 0
        self.high_bet_opp_wins = 0

        # For High Risk Metric: P(raising | them winning/losing, where we raised it to a high bet)
        self.highbet_oppwin_total = 0
        self.highbet_oppwin_raised = 0
        self.highbet_opplose_total = 0
        self.highbet_opplose_raised = 0

        # For Low Risk Metric: P(folding | high) - P(folding | low)
        self.preflop_high_total = 0
        self.preflop_high_fold = 0
        self.preflop_low_total = 0
        self.preflop_low_fold = 0

        # Opponent risk label and metrics (decided after first 50 rounds)
        self.opponent_risk_label = "Medium"
        self.classified_after_round: int | None = None
        self.last_high_risk_metric = 0.0
        self.last_low_risk_metric = 0.0

    # -----------------------------
    # Helper methods
    # -----------------------------

    def _chen_score(self, hand: list[str]) -> float:
        '''Chen formula score for a two-card starting hand.'''
        if len(hand) != 2:
            return 0.0
        c1, c2 = hand[0], hand[1]
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]

        val_map = {
            'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5,
            '9': 4.5, '8': 4, '7': 3.5, '6': 3,
            '5': 2.5, '4': 2, '3': 1.5, '2': 1,
        }
        idx1 = RANKS.index(r1)
        idx2 = RANKS.index(r2)

        # Ensure first card is highest
        if idx1 < idx2:
            idx1, idx2 = idx2, idx1
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        score = val_map.get(r1, 0)
        is_pair = r1 == r2

        if is_pair:
            score = max(5, score * 2)

        # Suited bonus
        if s1 == s2:
            score += 2

        # Gap penalty
        gap = 0
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

        # Small-card straight bonus
        if (not is_pair and gap in (0, 1)
                and idx1 < RANKS.index('Q') and idx2 < RANKS.index('Q')):
            score += 1

        # Chen score is typically rounded up
        return float(int(score) if score == int(score) else int(score) + 1)

    def _evaluate_7cards(self, cards: list[str]) -> tuple:
        '''
        Lightweight hand evaluator for 7 cards.
        Returns a tuple (category, r1, r2, r3, r4, r5) where higher is better.
        Category order: 8 SF, 7 Quads, 6 Full, 5 Flush, 4 Straight, 3 Trips, 2 Two Pair, 1 Pair, 0 High.
        '''
        ranks = [RANK_TO_VALUE[c[0]] for c in cards]
        suits = [c[1] for c in cards]

        # Rank and suit counts
        rank_counts: dict[int, int] = {}
        for v in ranks:
            rank_counts[v] = rank_counts.get(v, 0) + 1

        suit_counts: dict[str, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1

        # Detect flush
        flush_suit = None
        for s, cnt in suit_counts.items():
            if cnt >= 5:
                flush_suit = s
                break

        # Straight helper
        def straight_high(values: list[int]) -> int | None:
            uniq = sorted(set(values))
            # Wheel: A-2-3-4-5
            if 14 in uniq:
                uniq.append(1)
            best = None
            run = 1
            for i in range(1, len(uniq)):
                if uniq[i] == uniq[i - 1] + 1:
                    run += 1
                else:
                    run = 1
                if run >= 5:
                    best = uniq[i]
            return best

        # Straight flush
        if flush_suit is not None:
            flush_cards = [c for c in cards if c[1] == flush_suit]
            flush_ranks = [RANK_TO_VALUE[c[0]] for c in flush_cards]
            sf_high = straight_high(flush_ranks)
            if sf_high is not None:
                return (8, sf_high, 0, 0, 0, 0)

        # Rank groups sorted by (count, rank) desc
        groups = sorted(
            ((cnt, r) for r, cnt in rank_counts.items()),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        counts = [g[0] for g in groups]
        ordered_ranks = [g[1] for g in groups]

        # Four of a kind
        if counts[0] == 4:
            quad_rank = ordered_ranks[0]
            kicker = max(r for r in ranks if r != quad_rank)
            return (7, quad_rank, kicker, 0, 0, 0)

        # Full house
        if counts[0] == 3:
            trip_rank = ordered_ranks[0]
            pair_rank = None
            for cnt, r in groups[1:]:
                if cnt >= 2:
                    pair_rank = r
                    break
            if pair_rank is not None:
                return (6, trip_rank, pair_rank, 0, 0, 0)

        # Flush
        if flush_suit is not None:
            flush_cards = [c for c in cards if c[1] == flush_suit]
            flush_ranks_sorted = sorted(
                (RANK_TO_VALUE[c[0]] for c in flush_cards), reverse=True
            )[:5]
            while len(flush_ranks_sorted) < 5:
                flush_ranks_sorted.append(0)
            return (5, *flush_ranks_sorted[:5])

        # Straight
        st_high = straight_high(ranks)
        if st_high is not None:
            return (4, st_high, 0, 0, 0, 0)

        # Three of a kind
        if counts[0] == 3:
            trip_rank = ordered_ranks[0]
            kickers = sorted(
                (r for r in ranks if r != trip_rank), reverse=True
            )[:2]
            while len(kickers) < 2:
                kickers.append(0)
            return (3, trip_rank, kickers[0], kickers[1], 0, 0)

        # Two pair
        if counts[0] == 2 and counts[1] == 2:
            high_pair = max(ordered_ranks[0], ordered_ranks[1])
            low_pair = min(ordered_ranks[0], ordered_ranks[1])
            kicker = max(r for r in ranks if r != high_pair and r != low_pair)
            return (2, high_pair, low_pair, kicker, 0, 0)

        # One pair
        if counts[0] == 2:
            pair_rank = ordered_ranks[0]
            kickers = sorted(
                (r for r in ranks if r != pair_rank), reverse=True
            )[:3]
            while len(kickers) < 3:
                kickers.append(0)
            return (1, pair_rank, kickers[0], kickers[1], kickers[2], 0)

        # High card
        top = sorted(ranks, reverse=True)[:5]
        while len(top) < 5:
            top.append(0)
        return (0, *top[:5])

    def _estimate_equity(self, my_hand: list[str], board: list[str], iterations: int = 30) -> tuple[float, float]:
        '''
        Monte Carlo-style equity estimate versus a random opponent hand.
        Returns (P(us winning), P(opponent winning)).
        Uses a lightweight hand evaluator instead of eval7 to respect library constraints.
        '''
        if iterations <= 0 or len(my_hand) != 2:
            return 0.5, 0.5

        known = set(my_hand + board)
        deck = [c for c in FULL_DECK if c not in known]
        remaining_board = max(0, 5 - len(board))

        wins = 0
        ties = 0
        trials = 0

        for _ in range(iterations):
            if len(deck) < 2 + remaining_board:
                break
            sample = random.sample(deck, 2 + remaining_board)
            opp_hand = sample[:2]
            extra_board = sample[2:] if remaining_board > 0 else []
            full_board = board + extra_board

            hero_score = self._evaluate_7cards(my_hand + full_board)
            opp_score = self._evaluate_7cards(opp_hand + full_board)

            trials += 1
            if hero_score > opp_score:
                wins += 1
            elif hero_score == opp_score:
                ties += 1

        if trials == 0:
            return 0.5, 0.5

        p_us = (wins + 0.5 * ties) / float(trials)
        p_opp = max(0.0, 1.0 - p_us)
        return p_us, p_opp

    def _init_round_log(self, round_num: int) -> dict:
        if round_num not in self.round_logs:
            self.round_logs[round_num] = {
                "chen": None,
                "used_high_bet": False,
                "opp_raised": False,
                "we_folded": False,
                "preflop_high": False,
                "preflop_low": False,
            }
        return self.round_logs[round_num]

    def _update_classification_if_ready(self, round_num: int) -> None:
        if self.classified_after_round is not None:
            return
        if round_num < 50:
            return

        # High Risk Metric
        if self.highbet_oppwin_total > 0:
            p_raise_given_win = self.highbet_oppwin_raised / float(self.highbet_oppwin_total)
        else:
            p_raise_given_win = 0.0
        if self.highbet_opplose_total > 0:
            p_raise_given_lose = self.highbet_opplose_raised / float(self.highbet_opplose_total)
        else:
            p_raise_given_lose = 0.0
        high_metric = p_raise_given_win - p_raise_given_lose

        # Low Risk Metric
        if self.preflop_high_total > 0:
            p_fold_high = self.preflop_high_fold / float(self.preflop_high_total)
        else:
            p_fold_high = 0.0
        if self.preflop_low_total > 0:
            p_fold_low = self.preflop_low_fold / float(self.preflop_low_total)
        else:
            p_fold_low = 0.0
        low_metric = p_fold_high - p_fold_low

        self.last_high_risk_metric = high_metric
        self.last_low_risk_metric = low_metric

        # Thresholds for classifying opponent risk
        high_risk_threshold = 0.15
        low_risk_threshold = 0.15

        if high_metric >= high_risk_threshold:
            self.opponent_risk_label = "High"
        elif low_metric >= low_risk_threshold:
            self.opponent_risk_label = "Low"
        else:
            self.opponent_risk_label = "Medium"

        self.classified_after_round = round_num

        # Auction adjustment: if they win more auctions, increase our bid band
        if not self.adjusted_auction_bounds and self.auction_rounds > 0:
            if self.auction_wins * 2 < self.auction_rounds:
                self.base_auction_min = 2300
                self.base_auction_max = 2800
                self.adjusted_auction_bounds = True

    # -----------------------------
    # Engine interface
    # -----------------------------

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        '''
        Called when a new round starts. Called NUM_ROUNDS times.
        '''
        round_num = game_info.round_num
        self._init_round_log(round_num)

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        '''
        Called when a round ends. Called NUM_ROUNDS times.
        '''
        round_num = game_info.round_num
        info = self.round_logs.get(round_num, {})
        my_delta = current_state.payoff

        # Track auction wins (we know their card if we won the auction)
        self.auction_rounds += 1
        if current_state.opp_revealed_cards:
            self.auction_wins += 1

        # High-bet experiment outcomes
        if info.get("used_high_bet", False):
            if my_delta > 0:
                self.high_bet_hero_wins += 1
                self.highbet_opplose_total += 1
                if info.get("opp_raised", False):
                    self.highbet_opplose_raised += 1
            elif my_delta < 0:
                self.high_bet_opp_wins += 1
                self.highbet_oppwin_total += 1
                if info.get("opp_raised", False):
                    self.highbet_oppwin_raised += 1

        # Low Risk Metric components: folding behavior for high vs low Chen
        if info.get("preflop_high", False):
            self.preflop_high_total += 1
            if info.get("we_folded", False):
                self.preflop_high_fold += 1
        if info.get("preflop_low", False):
            self.preflop_low_total += 1
            if info.get("we_folded", False):
                self.preflop_low_fold += 1

        # After 50 rounds, classify opponent risk and adjust auction behavior
        self._update_classification_if_ready(round_num)

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        '''
        Decision logic for each action request from the engine.
        '''
        round_num = game_info.round_num
        info = self._init_round_log(round_num)

        # Auction: always bid a high random value in [2k, 2.5k], and after 50 rounds
        # optionally shift the band upwards if we are losing auctions.
        if current_state.street == 'auction':
            lo = self.base_auction_min
            hi = self.base_auction_max

            # Respect current stack size
            hi = min(hi, current_state.my_chips)
            if hi <= 0:
                return ActionBid(0)
            lo = max(0, min(lo, hi))

            if hi > lo:
                bid = random.randint(lo, hi)
            else:
                bid = hi
            return ActionBid(bid)

        # Mark if we are currently facing a bet (opponent has raised this round)
        if current_state.cost_to_call > 0:
            info["opp_raised"] = True

        # Pre-flop: Chen-formula based gating and high-bet experiments
        if current_state.street == 'pre-flop':
            chen = self._chen_score(current_state.my_hand)
            info["chen"] = chen

            if chen >= self.high_chen_threshold:
                info["preflop_high"] = True
            if chen <= self.preflop_fold_threshold:
                info["preflop_low"] = True

            # If Chen is very weak and we are facing a bet, fold pre-flop
            if (chen < self.preflop_fold_threshold
                    and current_state.cost_to_call > 0
                    and current_state.can_act(ActionFold)):
                info["we_folded"] = True
                return ActionFold()

            # High-bet experiment: for up to 25 rounds with strong Chen, raise to [3k, 3.5k]
            if (chen >= self.high_chen_threshold
                    and self.high_bet_experiments < self.high_bet_max_rounds
                    and round_num <= 50
                    and current_state.can_act(ActionRaise)):
                min_raise, max_raise = current_state.raise_bounds
                target = random.randint(3000, 3500)
                amount = max(min_raise, min(max_raise, target))
                if amount > current_state.my_wager:
                    info["used_high_bet"] = True
                    self.high_bet_experiments += 1
                    return ActionRaise(amount)

        # Decide current opponent risk label:
        # - Rounds 1-25: we still experiment; use Medium as default.
        # - Rounds 26-50: use Medium-risk rules as requested.
        # - After classification: use computed label.
        if self.classified_after_round is None:
            if round_num <= 50:
                risk_label = "Medium"
            else:
                risk_label = self.opponent_risk_label
        else:
            risk_label = self.opponent_risk_label

        # Monte Carlo-style win probabilities (P(us), P(opp))
        board = current_state.board
        # Use more iterations when board is more complete
        if len(board) == 0:
            iters = 25
        elif len(board) <= 3:
            iters = 30
        else:
            iters = 40
        p_us, p_opp = self._estimate_equity(current_state.my_hand, board, iterations=iters)

        delta = p_us - p_opp
        opp_minus_us = p_opp - p_us
        abs_delta = abs(delta)
        thr = self.equity_diff_threshold

        # Helper: choose a reasonable raise amount within legal bounds
        def choose_raise_amount() -> int:
            if not current_state.can_act(ActionRaise):
                return 0
            min_raise, max_raise = current_state.raise_bounds
            # Slightly aggressive: between min_raise and midpoint
            midpoint = (min_raise + max_raise) // 2 if max_raise > min_raise else min_raise
            return max(min_raise, midpoint)

        # Helper: call or check as appropriate
        def call_or_check():
            if current_state.cost_to_call == 0 and current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionCall):
                return ActionCall()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionFold):
                info["we_folded"] = True
                return ActionFold()
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        # Apply risk-dependent decision rules
        if risk_label == "High":
            # High Risk opponent:
            # - If P(opp) - P(us) > thr => Fold
            # - If |P(opp) - P(us)| < thr => Call/Check
            # - If P(us) - P(opp) > thr => Raise
            if opp_minus_us > thr:
                if current_state.cost_to_call > 0 and current_state.can_act(ActionFold):
                    info["we_folded"] = True
                    return ActionFold()
                return call_or_check()
            if abs_delta < thr:
                return call_or_check()
            # p_us - p_opp > thr
            if current_state.can_act(ActionRaise):
                amount = choose_raise_amount()
                return ActionRaise(amount)
            return call_or_check()

        if risk_label == "Low":
            # Low Risk opponent:
            # - If P(opp) - P(us) > thr:
            #     If bet < 500 => Call/Check, else Fold
            # - If |P(opp) - P(us)| < thr => Raise
            # - If P(us) - P(opp) > thr => Raise
            if opp_minus_us > thr:
                if current_state.cost_to_call <= 500:
                    return call_or_check()
                if current_state.cost_to_call > 0 and current_state.can_act(ActionFold):
                    info["we_folded"] = True
                    return ActionFold()
                return call_or_check()
            if abs_delta < thr or delta > thr:
                if current_state.can_act(ActionRaise):
                    amount = choose_raise_amount()
                    return ActionRaise(amount)
                return call_or_check()

        # Medium Risk opponent:
        # - If P(opp) - P(us) > thr => Call/Check
        # - If |P(opp) - P(us)| < thr => Call/Check
        # - If P(us) - P(opp) > thr => Raise
        if opp_minus_us > thr or abs_delta < thr:
            return call_or_check()
        # p_us - p_opp > thr
        if current_state.can_act(ActionRaise):
            amount = choose_raise_amount()
            return ActionRaise(amount)
        return call_or_check()


if __name__ == '__main__':
    run_bot(Player(), parse_args())