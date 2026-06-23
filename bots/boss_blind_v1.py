'''
Boss Blind V1 PokerBot
A bot that plays in three phases: highly aggressive, exploratory, and exploitative based on opponent profiling.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import math
import eval7

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

class Player(BaseBot):
    def __init__(self) -> None:
        self.log_data = [] 
        self.opp_raised_this_round = False
        self.opponent_class = "Medium"
        self.classification_done = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.opp_raised_this_round = False
        
        # Profile opponent on round 151 based on history of rounds 1-150
        if game_info.round_num == 151 and not self.classification_done:
            self.classification_done = True
            
            # Variables for evaluating opponent's behavior with High (Chen >= 10) vs Low cards
            S_win_high_raises = 0
            S_win_high_total = 0
            S_lose_high_raises = 0
            S_lose_high_total = 0
            
            S_high_folds = 0
            S_high_total = 0
            S_low_folds = 0
            S_low_total = 0
            
            for r in self.log_data:
                if r['both_cards_known']:
                    is_high = r['is_high']
                    won = r['they_won']
                    raised = r['they_raised']
                    folded = r['they_folded']
                    
                    if is_high:
                        S_high_total += 1
                        if folded: S_high_folds += 1
                        
                        if won:
                            S_win_high_total += 1
                            if raised: S_win_high_raises += 1
                        else:
                            S_lose_high_total += 1
                            if raised: S_lose_high_raises += 1
                    else:
                        S_low_total += 1
                        if folded: S_low_folds += 1
            
            # Calculate betting probability when opponent has high cards and wins vs loses
            p_raise_win_high = (S_win_high_raises / S_win_high_total) if S_win_high_total > 0 else 0
            p_raise_lose_high = (S_lose_high_raises / S_lose_high_total) if S_lose_high_total > 0 else 0
            high_risk_metric = p_raise_win_high - p_raise_lose_high
            
            # Calculate fold probability when opponent has high cards vs low cards
            p_fold_high = (S_high_folds / S_high_total) if S_high_total > 0 else 0
            p_fold_low = (S_low_folds / S_low_total) if S_low_total > 0 else 0
            low_risk_metric = p_fold_high - p_fold_low
            
            # Assign opponent classification
            if high_risk_metric < 0.3:
                # Opponent raises regardless of winning or losing chances -> High Risk / Tight / Very conservative or static
                self.opponent_class = "High Risk"
            elif low_risk_metric < 0.1:
                # Opponent folds about the same whether they have good or bad hands -> Low Risk / Easily bullied
                self.opponent_class = "Low Risk"
            else:
                self.opponent_class = "Medium"

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
            'they_folded': they_folded
        })

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
        
        # 1) Phase 1: Bully Phase (Rounds 1 - 75)
        if round_num <= 75:
            # Always bid max chips in auction for full info
            if street == 'auction':
                return ActionBid(current_state.my_chips)
            else:
                # Agreessively bet large amounts randomly
                amt = random.randint(3500, 4000)
                return self.make_bet(amt, current_state)
                
        # 2) Phase 2: Exploratory/Faker Phase (Rounds 76 - 150)        
        elif round_num <= 150:
            if street == 'auction':
                return ActionBid(current_state.my_chips)
            else:
                # Bet small amounts erratically to see how they react
                amt = random.randint(10, 1000)
                return self.make_bet(amt, current_state)
                
        # 3) Phase 3: Exploitative Phase (Rounds 151+)
        else:
            print(self.opponent_class)
            if street == 'auction':
                return ActionBid(current_state.my_chips)
                
            chen = chen_score(current_state.my_hand[0], current_state.my_hand[1])
            is_good = chen >= 10
            
            # High Risk Opponent: Play strictly mathematically
            if self.opponent_class == 'High Risk':
                if is_good:
                    amt = random.randint(3500, 4000)
                    return self.make_bet(amt, current_state)
                else:
                    if current_state.can_act(ActionCheck) and current_state.cost_to_call == 0:
                        return ActionCheck()
                    if current_state.can_act(ActionFold):
                        return ActionFold()
                    return ActionCheck()
                    
            # Low Risk Opponent: Opponent folds easily, so bully them constantly
            elif self.opponent_class == 'Low Risk':
                amt = random.randint(3500, 4000)
                return self.make_bet(amt, current_state)
                
            # Medium Risk Opponent: Value bet based on known cards
            else:
                known = current_state.my_hand + current_state.board + current_state.opp_revealed_cards
                val_map = {'A':14, 'K':13, 'Q':12, 'J':11, 'T':10, '9':9, '8':8, '7':7, '6':6, '5':5, '4':4, '3':3, '2':2}
                val = sum(val_map[c[0]] for c in known)
                amt = val * 60
                return self.make_bet(amt, current_state)

if __name__ == '__main__':
    run_bot(Player(), parse_args())
