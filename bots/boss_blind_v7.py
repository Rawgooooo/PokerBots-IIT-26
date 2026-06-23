'''
PokerBot v7: The Pot Odds Optimizer
Fixes the auction leak and implements proper Expected Value (EV) decision making.
'''
import random
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

# --- Constants & Lookup ---
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
FULL_DECK = [r + s for r in RANKS for s in SUITS]
RANK_TO_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}

class Player(BaseBot):
    def __init__(self) -> None:
        # Configuration
        self.monte_carlo_iters = 40  # Speed vs Accuracy balance
        
    def _chen_score(self, hand: list[str]) -> float:
        '''Standard Chen formula for pre-flop strength.'''
        if not hand or len(hand) != 2: return 0.0
        
        # Parse cards
        r1, s1 = hand[0][0], hand[0][1]
        r2, s2 = hand[1][0], hand[1][1]
        
        # Value mapping
        val = {'A':10, 'K':8, 'Q':7, 'J':6, 'T':5, '9':4.5, '8':4, '7':3.5, '6':3, '5':2.5, '4':2, '3':1.5, '2':1}
        
        # Sort high to low
        v1, v2 = RANK_TO_VALUE[r1], RANK_TO_VALUE[r2]
        if v1 < v2: 
            r1, r2 = r2, r1
            s1, s2 = s2, s1
            v1, v2 = v2, v1
            
        score = val.get(r1, 0)
        
        # Pairs
        if r1 == r2:
            score = max(5, score * 2)
        elif s1 == s2: # Suited
            score += 2
            
        # Gaps
        gap = v1 - v2 - 1
        if r1 != r2:
            if gap == 1: score -= 1
            elif gap == 2: score -= 2
            elif gap == 3: score -= 4
            elif gap >= 4: score -= 5
            
        # Straight bonus (small cards)
        if gap <= 1 and v1 < 12 and not r1 == r2: # < Q
            score += 1
            
        return score

    def _evaluate_7cards(self, cards: list[str]) -> tuple:
        '''
        Returns numeric tuple representation of hand strength.
        (Score, High1, High2, High3, High4, High5)
        '''
        # Parse
        ranks = sorted([RANK_TO_VALUE[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        
        # Flush check
        suit_counts = {s: suits.count(s) for s in SUITS}
        flush_suit = next((s for s, c in suit_counts.items() if c >= 5), None)
        
        flush_ranks = []
        if flush_suit:
            flush_ranks = sorted([RANK_TO_VALUE[c[0]] for c in cards if c[1] == flush_suit], reverse=True)
            
        # Straight check
        def get_straight_high(rk_list):
            uniques = sorted(set(rk_list))
            if 14 in uniques: uniques = [1] + uniques # Ace low handling
            consecutive = 0
            high_card = 0
            for i in range(len(uniques)-1):
                if uniques[i+1] == uniques[i] + 1:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= 4:
                    high_card = uniques[i+1]
            return high_card

        straight_high = get_straight_high(ranks)
        
        # Straight Flush
        if flush_suit:
            sf_high = get_straight_high(flush_ranks)
            if sf_high > 0:
                return (9, sf_high)

        # Quads
        rank_counts = {r: ranks.count(r) for r in ranks}
        quads = [r for r, c in rank_counts.items() if c == 4]
        if quads:
            kicker = max([r for r in ranks if r != quads[0]])
            return (8, quads[0], kicker)
            
        # Full House
        trips = [r for r, c in rank_counts.items() if c == 3]
        pairs = [r for r, c in rank_counts.items() if c == 2]
        if trips:
            t = trips[0] # Highest trip
            # If two trips, the lower one becomes the pair
            remaining = [r for r in trips if r != t] + pairs
            if remaining:
                p = max(remaining)
                return (7, t, p)
        
        # Flush
        if flush_suit:
            return (6, *flush_ranks[:5])
            
        # Straight
        if straight_high > 0:
            return (5, straight_high)
            
        # Trips
        if trips:
            kickers = sorted([r for r in ranks if r != trips[0]], reverse=True)[:2]
            return (4, trips[0], *kickers)
            
        # Two Pair
        if len(pairs) >= 2:
            p1 = pairs[0]
            p2 = pairs[1]
            kicker = max([r for r in ranks if r != p1 and r != p2])
            return (3, p1, p2, kicker)
            
        # Pair
        if pairs:
            kickers = sorted([r for r in ranks if r != pairs[0]], reverse=True)[:3]
            return (2, pairs[0], *kickers)
            
        # High Card
        return (1, *ranks[:5])

    def _calculate_equity(self, my_hand, board, iters=50):
        '''
        Monte Carlo simulation to estimate Win %
        '''
        deck = [c for c in FULL_DECK if c not in my_hand and c not in board]
        wins = 0
        
        # Adaptive iterations: fewer pre-flop (faster), more post-flop (accuracy)
        if len(board) == 0: iters = 30
        elif len(board) >= 3: iters = 50
        
        for _ in range(iters):
            # Shuffle remaining deck
            random.shuffle(deck)
            
            # Opponent hand
            opp_hand = deck[:2]
            
            # Finish board
            cards_needed = 5 - len(board)
            runout = deck[2:2+cards_needed]
            final_board = board + runout
            
            my_score = self._evaluate_7cards(my_hand + final_board)
            opp_score = self._evaluate_7cards(opp_hand + final_board)
            
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5
                
        return wins / iters

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        
        # 1. PRE-FLOP STRATEGY (Chen Score)
        if current_state.street == 'pre-flop':
            chen = self._chen_score(current_state.my_hand)
            
            # Always check if free
            if current_state.can_act(ActionCheck) and current_state.cost_to_call == 0:
                return ActionCheck()

            # Conservative but opportunistic
            if chen >= 9:
                # Strong hand: Raise or Call
                if current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    # Don't go crazy, small raise to build pot
                    amt = min(max_r, min_r + 20)
                    return ActionRaise(amt)
                return ActionCall()
            
            elif chen >= 6:
                # Playable: Call if cheap
                if current_state.cost_to_call <= 100:
                    return ActionCall()
                return ActionFold()
            
            else:
                return ActionFold()

        # 2. AUCTION STRATEGY (The Fix)
        # Bid proportional to pot size and equity.
        # Don't pay 2000 chips for a 100 chip pot!
        if current_state.street == 'auction':
            equity = self._calculate_equity(current_state.my_hand, current_state.board)
            pot = current_state.pot
            
            # Basic bid: How much of the pot do we "own"?
            # We add a small buffer because knowing the card is valuable.
            bid_amount = int(pot * equity * 1.1)
            
            # Cap bid to prevent stack suicide on draws
            # Never bid more than 1/3 of our remaining stack
            bid_amount = min(bid_amount, current_state.my_chips // 3)
            
            return ActionBid(bid_amount)

        # 3. POST-FLOP STRATEGY (Pot Odds)
        # Calculate Equity
        equity = self._calculate_equity(current_state.my_hand, current_state.board)
        
        # Calculate Pot Odds
        cost = current_state.cost_to_call
        pot = current_state.pot
        
        # Pot odds = Price / (Pot + Price)
        # e.g., Call 50 to win 100. Odds = 50/150 = 33%.
        # If Equity > 33%, we call.
        
        if cost == 0:
            pot_odds = 0
        else:
            pot_odds = cost / (pot + cost)
            
        # EV Decision
        # 1. Check if free
        if current_state.can_act(ActionCheck) and cost == 0:
            # If we have a monster (equity > 80%), slow play or value bet
            if equity > 0.8 and current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                # Value bet ~50% of pot
                bet = max(min_r, min(max_r, int(pot * 0.5)))
                return ActionRaise(bet)
            return ActionCheck()

        # 2. Facing a Bet
        # Add a "safety margin" because our equity is an estimate and opponents aren't random
        required_equity = pot_odds + 0.05 
        
        if equity >= required_equity:
            # We are profitable to call.
            # Should we raise? Only if we are crushing (Equity > 70%)
            if equity > 0.7 and current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                bet = max(min_r, min(max_r, int(pot * 0.75)))
                return ActionRaise(bet)
            return ActionCall()
        
        # 3. Bluff Opportunity (Optional / Advanced)
        # If we have low equity but the opponent checks to us?
        # For now, let's play solid ABC poker to beat v3.
        
        if current_state.can_act(ActionCheck):
            return ActionCheck()
            
        return ActionFold()

if __name__ == '__main__':
    run_bot(Player(), parse_args())