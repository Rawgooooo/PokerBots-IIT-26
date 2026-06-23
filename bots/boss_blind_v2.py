'''
Boss Blind V2 PokerBot
Plays in three phases with dynamic opponent profiling, pot-odds-aware betting,
and lightweight hand-equity evaluation using eval7.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import math
import eval7

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
ALL_CARDS = [eval7.Card(r + s) for r in RANKS for s in SUITS]


def chen_score(card1, card2):
    """
    Calculates the standard Chen Formula score for starting hands.
    https://en.wikipedia.org/wiki/Texas_hold_%27em_starting_hands#Chen_formula
    Used to estimate the pre-flop strength of a hand.
    """
    val_str = '23456789TJQKA'
    val_map = {'A':10, 'K':8, 'Q':7, 'J':6, 'T':5, '9':4.5, '8':4, '7':3.5, '6':3, '5':2.5, '4':2, '3':1.5, '2':1}
    rank1, suit1 = card1[0], card1[1]
    rank2, suit2 = card2[0], card2[1]
    
    idx1 = val_str.index(rank1)
    idx2 = val_str.index(rank2)
    
    # Sort cards so rank1 > rank2 (highest first)
    if idx1 < idx2:
        idx1, idx2 = idx2, idx1
        rank1, rank2 = rank2, rank1
        suit1, suit2 = suit2, suit1
        
    score = val_map[rank1]
    is_pair = (rank1 == rank2)
    
    if is_pair:
        score = max(5, score * 2)
    
    if suit1 == suit2:
        score += 2
        
    gap = 0
    if not is_pair:
        gap = idx1 - idx2 - 1
        if gap == 0:
            pass
        elif gap == 1:
            score -= 1
        elif gap == 2:
            score -= 2
        elif gap == 3:
            score -= 4
        else:
            score -= 5
            
    # Add 1 point if gap is 0 or 1 and cards are lower than Q
    if not is_pair and gap in [0, 1] and idx1 < val_str.index('Q') and idx2 < val_str.index('Q'):
        score += 1
        
    return math.ceil(score)


def estimate_hand_equity(my_cards, board, iterations=50):
    '''
    Monte Carlo equity estimate versus a random hand using eval7.
    Uses a small iteration count to respect time limits.
    '''
    if iterations <= 0:
        return 0.0

    known_cards = set(eval7.Card(c) for c in my_cards + board)
    deck = [card for card in ALL_CARDS if card not in known_cards]

    hero_cards = [eval7.Card(c) for c in my_cards]
    board_cards = [eval7.Card(c) for c in board]
    remaining_board = 5 - len(board_cards)

    wins = 0
    ties = 0

    for _ in range(iterations):
        # sample opponent hand + remaining board cards without replacement
        sample = random.sample(deck, 2 + max(0, remaining_board))
        opp_cards = sample[:2]
        extra_board = sample[2:] if remaining_board > 0 else []

        full_board = board_cards + extra_board

        hero_score = eval7.evaluate(hero_cards + full_board)
        opp_score = eval7.evaluate(opp_cards + full_board)

        if hero_score > opp_score:
            wins += 1
        elif hero_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / iterations


def calculate_pot_odds(state: PokerState) -> float:
    '''Returns cost_to_call / (pot + cost_to_call).'''
    cost = state.cost_to_call
    if cost <= 0:
        return 0.0
    total = state.pot + cost
    if total <= 0:
        return 1.0
    return cost / float(total)


class Player(BaseBot):
    def __init__(self) -> None:
        self.log_data = [] 
        self.opp_raised_this_round = False
        self.opponent_class = "Medium"
        self.classification_done = False
        self.equity_cache = {}
        self.window_size = 40  # sliding window size for opponent profiling
        self.opp_vpip_this_round = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.opp_raised_this_round = False
        self.opp_vpip_this_round = False
        self.equity_cache = {}

        # Sliding-window opponent profiling: update every 25 rounds once we have data
        if len(self.log_data) >= 20 and game_info.round_num % 25 == 1:
            self._update_opponent_profile()

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        my_payoff = current_state.payoff
        opp_cards = current_state.opp_revealed_cards
        
        they_won = my_payoff < 0
        both_cards_known = len(opp_cards) == 2
        is_high = False
        if both_cards_known:
            score = chen_score(opp_cards[0], opp_cards[1])
            is_high = score >= 10
            
        # They folded if payoff is positive and we didn't see both cards at showdown
        they_folded = (my_payoff > 0 and len(opp_cards) < 2)
        
        # Log this hand for profiling logic at round 151
        self.log_data.append({
            'they_won': they_won,
            'both_cards_known': both_cards_known,
            'is_high': is_high,
            'they_raised': self.opp_raised_this_round,
            'they_folded': they_folded,
            'vpip': self.opp_vpip_this_round,
        })

    def _update_opponent_profile(self):
        '''Recompute opponent_class using a sliding window and VPIP.'''
        if not self.log_data:
            return

        window = min(self.window_size, len(self.log_data))
        recent = self.log_data[-window:]

        # VPIP: how often opponent voluntarily put chips in pre-flop
        vpip_hands = sum(1 for r in recent if r.get('vpip'))
        vpip_rate = vpip_hands / float(window)

        # Classify: Loose -> High Risk, Tight -> Low Risk, otherwise Medium
        if vpip_rate > 0.7:
            self.opponent_class = "High Risk"
        elif vpip_rate < 0.2:
            self.opponent_class = "Low Risk"
        else:
            self.opponent_class = "Medium"

    def _get_hand_equity(self, game_info: GameInfo, current_state: PokerState, iterations: int = 50) -> float:
        '''Cached wrapper around estimate_hand_equity to keep CPU usage low.'''
        key = (
            game_info.round_num,
            current_state.street,
            tuple(current_state.my_hand),
            tuple(current_state.board),
        )
        if key in self.equity_cache:
            return self.equity_cache[key]

        equity = estimate_hand_equity(current_state.my_hand, current_state.board, iterations)
        self.equity_cache[key] = equity
        return equity

    def make_bet(self, amount, current_state):
        """Helper to ensure legal bets, falling back to call/check or fold."""
        if current_state.can_act(ActionRaise):
            min_raise, max_raise = current_state.raise_bounds
            bet = max(min_raise, amount)
            bet = min(max_raise, bet)
            return ActionRaise(bet)
        if current_state.can_act(ActionCall):
            return ActionCall()
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        if current_state.cost_to_call > 0:
            self.opp_raised_this_round = True
            
        round_num = game_info.round_num
        street = current_state.street

        # Track VPIP pre-flop: did opponent voluntarily put chips in the pot?
        if street == 'pre-flop' and not self.opp_vpip_this_round:
            # Opponent's forced blind depends on whether we are BB or SB
            opp_forced = BIG_BLIND if current_state.is_bb else SMALL_BLIND
            opp_invested = STARTING_STACK - current_state.opp_chips
            if opp_invested > opp_forced:
                self.opp_vpip_this_round = True

        # Auction strategy: avoid wasting chips, try to win info cheaply
        if street == 'auction':
            bid_amount = 1
            if bid_amount > current_state.my_chips:
                bid_amount = current_state.my_chips
            return ActionBid(bid_amount)

        # 1) Phase 1: Toned-down Bully Phase (Rounds 1 - 75)
        if round_num <= 75:
            # Still apply pressure but avoid burning full stack every hand
            base_bet = random.randint(200, 800)
            return self.make_bet(base_bet, current_state)

        # 2) Phase 2: Exploratory/Faker Phase (Rounds 76 - 150)
        elif round_num <= 150:
            # Small, somewhat noisy bets to gather reactions without huge risk
            amt = random.randint(50, 400)
            return self.make_bet(amt, current_state)

        # 3) Phase 3: Exploitative Phase (Rounds 151+)
        else:
            chen = chen_score(current_state.my_hand[0], current_state.my_hand[1])
            # Use slightly more iterations on post-flop where board matters
            iters = 40 if street == 'pre-flop' else 60
            equity = self._get_hand_equity(game_info, current_state, iterations=iters)
            pot_odds = calculate_pot_odds(current_state)

            # High Risk Opponent: fold clearly bad hands, otherwise continue aggressively
            if self.opponent_class == 'High Risk':
                bad_hand = (chen < 6) or (equity < max(0.25, pot_odds + 0.05))
                if bad_hand:
                    if current_state.cost_to_call > 0 and current_state.can_act(ActionFold):
                        return ActionFold()
                    if current_state.can_act(ActionCheck):
                        return ActionCheck()
                    if current_state.can_act(ActionCall):
                        return ActionCall()
                    return ActionFold()

                # Strong hand: size raise using equity and pot
                target_base = current_state.pot + current_state.cost_to_call + BIG_BLIND
                bet_amount = max(BIG_BLIND * 2, int(equity * target_base))
                return self.make_bet(bet_amount, current_state)

            # Low Risk Opponent: exploit by bullying, but respect hand quality by street
            elif self.opponent_class == 'Low Risk':
                bad_hand = (chen < 6) or (equity < max(0.30, pot_odds + 0.05))

                # Flop: if our hand is good, raise 1500-2500; otherwise play pot-odds aware
                if street == 'flop':
                    if not bad_hand:
                        amt = random.randint(1500, 2500)
                        return self.make_bet(amt, current_state)

                    # Weak on flop: avoid overpaying
                    if current_state.cost_to_call > 0 and (equity < pot_odds or equity < 0.30):
                        if current_state.can_act(ActionFold):
                            return ActionFold()
                    if current_state.cost_to_call == 0 and current_state.can_act(ActionCheck):
                        return ActionCheck()
                    if current_state.can_act(ActionCall):
                        return ActionCall()
                    if current_state.can_act(ActionCheck):
                        return ActionCheck()
                    return ActionFold()

                # Turn: if our cards are bad, fold in the round after the flop
                if street == 'turn':
                    if bad_hand and current_state.cost_to_call > 0 and current_state.can_act(ActionFold):
                        return ActionFold()

                    if not bad_hand:
                        amt = random.randint(800, 1600)
                        return self.make_bet(amt, current_state)

                    if current_state.cost_to_call == 0 and current_state.can_act(ActionCheck):
                        return ActionCheck()
                    if current_state.can_act(ActionCall):
                        return ActionCall()
                    if current_state.can_act(ActionCheck):
                        return ActionCheck()
                    return ActionFold()

                # Pre-flop and river: use standard equity vs pot-odds logic
                if current_state.cost_to_call > 0 and (equity < pot_odds or bad_hand):
                    if current_state.can_act(ActionFold):
                        return ActionFold()

                if not bad_hand and (equity > pot_odds + 0.10 or current_state.cost_to_call == 0):
                    base = max(current_state.pot, BIG_BLIND * 4)
                    amt = int(equity * base)
                    return self.make_bet(amt, current_state)

                if current_state.cost_to_call == 0 and current_state.can_act(ActionCheck):
                    return ActionCheck()
                if current_state.can_act(ActionCall):
                    return ActionCall()
                if current_state.can_act(ActionCheck):
                    return ActionCheck()
                return ActionFold()

            # Medium Risk Opponent: value bet based on eval7 equity and pot odds
            else:
                if current_state.cost_to_call > 0 and equity < max(0.20, pot_odds - 0.05):
                    if current_state.can_act(ActionFold):
                        return ActionFold()

                if equity > pot_odds + 0.10:
                    base = current_state.pot + current_state.cost_to_call + BIG_BLIND
                    amt = int(equity * base)
                    return self.make_bet(amt, current_state)

                if current_state.cost_to_call == 0 and current_state.can_act(ActionCheck):
                    return ActionCheck()
                if current_state.can_act(ActionCall):
                    return ActionCall()
                if current_state.can_act(ActionCheck):
                    return ActionCheck()
                return ActionFold()

if __name__ == '__main__':
    run_bot(Player(), parse_args())