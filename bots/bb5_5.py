from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import eval7
import collections

# --- Constants ---
RANKS = '23456789TJQKA'
SUITS = 'cdhs'

# ==============================================================================
# PRE-FLOP EQUITY LOOKUP TABLE
# ==============================================================================
# This dictionary maps every specific 169 starting hand combinations to its 
# approximate pre-flop equity (%) against a random unknown hand. 
# Format:
#   Pairs:   'AA', 'KK', 'QQ', etc.
#   Suited:  'AKs', 'AQs', 'JTs', etc. (s = suited)
#   Offsuit: 'AKo', 'AQo', 'JTo', etc. (o = offsuit)
#
# Values are percentages (e.g., 0.85 = 85% equity).
# Source: Standard Monte Carlo simulations of Hold'em starting hands vs random.
# ==============================================================================
PREFLOP_EQUITY = {
    # PAIRS
    'AA': 0.852, 'KK': 0.824, 'QQ': 0.799, 'JJ': 0.775, 'TT': 0.751, 
    '99': 0.716, '88': 0.691, '77': 0.662, '66': 0.632, '55': 0.603, 
    '44': 0.570, '33': 0.536, '22': 0.503,
    
    # SUITED (Aces)
    'AKs': 0.670, 'AQs': 0.661, 'AJs': 0.654, 'ATs': 0.647, 'A9s': 0.630, 
    'A8s': 0.621, 'A7s': 0.610, 'A6s': 0.599, 'A5s': 0.599, 'A4s': 0.589, 
    'A3s': 0.580, 'A2s': 0.570,
    
    # SUITED (Kings)
    'KQs': 0.634, 'KJs': 0.626, 'KTs': 0.619, 'K9s': 0.600, 'K8s': 0.589, 
    'K7s': 0.582, 'K6s': 0.571, 'K5s': 0.558, 'K4s': 0.546, 'K3s': 0.537, 
    'K2s': 0.526,
    
    # SUITED (Queens)
    'QJs': 0.603, 'QTs': 0.595, 'Q9s': 0.576, 'Q8s': 0.563, 'Q7s': 0.542, 
    'Q6s': 0.531, 'Q5s': 0.518, 'Q4s': 0.506, 'Q3s': 0.495, 'Q2s': 0.485,
    
    # SUITED (Jacks)
    'JTs': 0.575, 'J9s': 0.555, 'J8s': 0.541, 'J7s': 0.528, 'J6s': 0.504, 
    'J5s': 0.493, 'J4s': 0.481, 'J3s': 0.470, 'J2s': 0.457,
    
    # SUITED (Tens)
    'T9s': 0.540, 'T8s': 0.526, 'T7s': 0.512, 'T6s': 0.498, 'T5s': 0.473, 
    'T4s': 0.461, 'T3s': 0.449, 'T2s': 0.438,
    
    # SUITED (Nines & below)
    '98s': 0.508, '97s': 0.494, '96s': 0.479, '95s': 0.463, '94s': 0.437, '93s': 0.426, '92s': 0.413,
    '87s': 0.480, '86s': 0.465, '85s': 0.449, '84s': 0.432, '83s': 0.406, '82s': 0.395,
    '76s': 0.456, '75s': 0.439, '74s': 0.421, '73s': 0.404, '72s': 0.375,
    '65s': 0.433, '64s': 0.415, '63s': 0.395, '62s': 0.378,
    '54s': 0.410, '53s': 0.393, '52s': 0.374,
    '43s': 0.380, '42s': 0.362,
    '32s': 0.353,
    
    # OFFSUIT (Aces)
    'AKo': 0.653, 'AQo': 0.644, 'AJo': 0.636, 'ATo': 0.629, 'A9o': 0.609, 
    'A8o': 0.600, 'A7o': 0.589, 'A6o': 0.577, 'A5o': 0.576, 'A4o': 0.565, 
    'A3o': 0.556, 'A2o': 0.546,
    
    # OFFSUIT (Kings)
    'KQo': 0.614, 'KJo': 0.606, 'KTo': 0.598, 'K9o': 0.577, 'K8o': 0.566, 
    'K7o': 0.558, 'K6o': 0.546, 'K5o': 0.533, 'K4o': 0.520, 'K3o': 0.510, 
    'K2o': 0.499,
    
    # OFFSUIT (Queens)
    'QJo': 0.582, 'QTo': 0.573, 'Q9o': 0.552, 'Q8o': 0.538, 'Q7o': 0.516, 
    'Q6o': 0.504, 'Q5o': 0.490, 'Q4o': 0.477, 'Q3o': 0.465, 'Q2o': 0.455,
    
    # OFFSUIT (Jacks)
    'JTo': 0.553, 'J9o': 0.530, 'J8o': 0.515, 'J7o': 0.501, 'J6o': 0.476, 
    'J5o': 0.464, 'J4o': 0.451, 'J3o': 0.439, 'J2o': 0.426,
    
    # OFFSUIT (Tens)
    'T9o': 0.514, 'T8o': 0.499, 'T7o': 0.484, 'T6o': 0.469, 'T5o': 0.443, 
    'T4o': 0.430, 'T3o': 0.417, 'T2o': 0.405,
    
    # OFFSUIT (Nines & below)
    '98o': 0.480, '97o': 0.465, '96o': 0.449, '95o': 0.432, '94o': 0.404, '93o': 0.392, '92o': 0.379,
    '87o': 0.450, '86o': 0.434, '85o': 0.417, '84o': 0.398, '83o': 0.370, '82o': 0.358,
    '76o': 0.424, '75o': 0.406, '74o': 0.387, '73o': 0.368, '72o': 0.337,
    '65o': 0.400, '64o': 0.380, '63o': 0.358, '62o': 0.340,
    '54o': 0.374, '53o': 0.356, '52o': 0.336,
    '43o': 0.342, '42o': 0.323,
    '32o': 0.313
}

# Value map to rank cards from 2 (lowest) to A (highest)
CARD_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def get_hand_key(cards: list[str]) -> str:
    '''
    Converts a 2-card hand list (e.g. ['Ah', 'Kd']) into 
    the 169-hand lookup format used in PREFLOP_EQUITY (e.g., 'AKo').
    
    Arguments:
    cards: A list of two strings representing the cards.
    
    Returns:
    A string matching a key in PREFLOP_EQUITY.
    '''
    # We expect exactly 2 cards pre-flop
    if len(cards) != 2:
        return '72o' # Fallback for safety (worst hand)
        
    c1, c2 = cards[0], cards[1]
    
    # Extract rank and suit
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    
    # Figure out which card is the "higher" rank 
    # (e.g., 'A' always comes before 'K' in 'AKs')
    if CARD_VALUES[r1] >= CARD_VALUES[r2]:
        high_r, low_r = r1, r2
    else:
        high_r, low_r = r2, r1
        
    # Is it a pocket pair?
    if high_r == low_r:
        return f"{high_r}{low_r}"
        
    # Is it suited?
    if s1 == s2:
        return f"{high_r}{low_r}s"
    
    # Otherwise, it must be offsuit
    return f"{high_r}{low_r}o"


# ==============================================================================
# POST-FLOP FAST MONTE CARLO (WITH RESTRICTED RANGE)
# ==============================================================================
def estimate_hand_equity_mc(my_cards: list[str], board: list[str], opp_known: list[str], iterations: int = 50) -> float:
    '''
    Evaluates current hand equity by running a Monte Carlo simulation.
    If 'opp_known' contains cards, the simulation instantly restricts the 
    opponent's theoretical hand range to ONLY combinations containing exactly 
    that card. This drastically cuts the state space.
    '''
    if iterations <= 0: 
        return 0.0
        
    known_cards = set(eval7.Card(c) for c in (my_cards + board + opp_known))
    deck = [card for card in (eval7.Card(r+s) for r in RANKS for s in SUITS) if card not in known_cards]
    
    hero = [eval7.Card(c) for c in my_cards]
    brd = [eval7.Card(c) for c in board]
    opp_k = [eval7.Card(c) for c in opp_known]
    
    rem_b = 5 - len(brd)      # Remaining board cards to finish the 5-card runout
    rem_o = 2 - len(opp_k)    # Remaining hole cards opponent needs (usually 1 if we won auction, 2 if neutral)
    
    wins = 0
    ties = 0
    
    for _ in range(iterations):
        samp = random.sample(deck, rem_b + rem_o)
        opp_c = opp_k + samp[:rem_o]
        full_b = brd + samp[rem_o:]
        
        hs = eval7.evaluate(hero + full_b)
        os = eval7.evaluate(opp_c + full_b)
        
        if hs > os: 
            wins += 1
        elif hs == os: 
            ties += 1
            
    return (wins + 0.5 * ties) / iterations

# Helper to normalize bets within pokerbot limits
def make_bet(amount: int, state: PokerState):
    if not state.can_act(ActionRaise):
        if state.can_act(ActionCall): return ActionCall()
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()
    min_r, max_r = state.raise_bounds
    bet = max(min_r, min(max_r, int(amount)))
    return ActionRaise(bet)


class Player(BaseBot):
    '''
    A pokerbot that utilizes pre-computed mathematical equity tables
    for Pre-Flop decision mechanics based on pot odds. Post-flop strategy
    is currently minimal/fallback.
    '''

    def __init__(self) -> None:
        '''
        Called when a new game starts. Called exactly once.
        '''
        self.opp_bids = collections.deque(maxlen=50)
        self.opp_bid_percentages = collections.deque(maxlen=50)
        self.pot_before_auction = None
        self.last_bid = 0
        self.assume_high_mean = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        '''
        Called when a new round starts. Called NUM_ROUNDS times.
        '''
        my_bankroll = game_info.bankroll 
        time_bank = game_info.time_bank
        round_num = game_info.round_num 
        my_cards = current_state.my_hand
        opp_revealed_cards = current_state.opp_revealed_cards
        big_blind = current_state.is_bb 
        pass

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        '''
        Called when a round ends. Called NUM_ROUNDS times.
        '''
        my_delta = current_state.payoff
        street = current_state.street 
        my_cards = current_state.my_hand
        opp_revealed_cards = current_state.opp_revealed_cards

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        '''
        Decides the action for the bot.
        Prioritizes Pre-Flop Equity math compared against calculated Pot Odds.
        '''
        
        # ----------------------------------------------------------------------
        # 1. AUCTION PHASE (Core Logic Phase 3)
        # ----------------------------------------------------------------------
        if current_state.street == 'auction':
            self.pot_before_auction = current_state.pot
            
            # Find Post flop equity using Monte Carlo
            my_cards = current_state.my_hand
            equity = estimate_hand_equity_mc(my_cards, current_state.board, [], iterations=500)
            
            # Calculate means
            if len(self.opp_bids) > 0:
                opp_bid_mean = sum(self.opp_bids) / len(self.opp_bids)
                opp_bid_percentage_mean = sum(self.opp_bid_percentages) / len(self.opp_bid_percentages)
            else:
                opp_bid_mean = 3000
                opp_bid_percentage_mean = 1.0
                
            if game_info.round_num <= 50:
                bid_amount = 3000
            else:
                if equity > 0.80:
                    # Bid max(0, int(90-95% of Opponents rolling mean percentage)) * pot ?
                    # The requirement: "int(90-95% of Opponents rolling mean percentage)" usually means 
                    # applied to the pot size, as percentage is a fraction of the pot.
                    multiplier = random.uniform(0.90, 0.95)
                    bid_amount = max(0, int(multiplier * opp_bid_percentage_mean * current_state.pot))
                elif equity > 0.60: # 60-80%
                    if opp_bid_mean > 1000:
                        bid_amount = int(0.8 * opp_bid_mean)
                    else:
                        bid_amount = random.randint(400, 600)
                elif equity >= 0.25: # 25-60%
                    bid_amount = int(0.8 * opp_bid_mean)
                else: # < 25%
                    bid_amount = 10
            
            bid_amount = min(bid_amount, current_state.my_chips)
            if bid_amount < 0: bid_amount = 0
            
            self.last_bid = bid_amount
            return ActionBid(bid_amount)
            
        # ----------------------------------------------------------------------
        # 2. PRE-FLOP PHASE (Core Logic Phase 1)
        # ----------------------------------------------------------------------
        if current_state.street == 'pre-flop':
            
            # Identify our two-card hand
            my_cards = current_state.my_hand
            
            # Map the cards to the 169-format (e.g., 'AKo')
            hand_key = get_hand_key(my_cards)
            
            # Retrieve our mathematical baseline equity
            # (Defaults to 0.33 if something goes wrong and it can't find it)
            hand_equity = PREFLOP_EQUITY.get(hand_key, 0.33)
            
            # Determine Pot Odds. 
            # Pot Odds = call amount / (pot size + call amount)
            # This represents the strictly mathematical hurdle our hand must jump
            call_amount = current_state.cost_to_call
            pot_size = current_state.pot
            
            if call_amount > 0:
                pot_odds = call_amount / (pot_size + call_amount)
            else:
                # If we don't have to put any chips in, our pot odds are basically 0
                pot_odds = 0.0
            
            # MARGIN of SAFETY: 
            # We don't want to call iff Equity == Pot Odds. That's a 0 EV (Expected Value) play.
            # We want an edge. We'll add a tiny margin.
            MARGIN = 0.05
            
            # DECISION TREE:
            if hand_equity >= (pot_odds + MARGIN):
                
                # We have a strong mathematical advantage. Should we raise?
                # We'll use a very aggressive threshold for raising pre-flop.
                RAISE_THRESHOLD = 0.60 # Top tier hands (pair of 5s+, A8s+, K9s+, etc.)
                
                if hand_equity >= RAISE_THRESHOLD and current_state.can_act(ActionRaise):
                    min_raise, max_raise = current_state.raise_bounds
                    if hand_equity > 0.8:
                        # Bait them: middle bet
                        bet = min_raise + int(0.5 * current_state.pot)
                        bet = max(min_raise, min(bet, max_raise))
                        return ActionRaise(bet)
                    else:
                        # Standard minimum raise
                        return ActionRaise(min_raise)
                else:
                    # Good odds, but not necessarily a premium hand. We'll call.
                    if current_state.can_act(ActionCall):
                        return ActionCall()
                    elif current_state.can_act(ActionCheck):
                        return ActionCheck()
            else:
                # The odds don't justify calling or raising.
                # If we can check for free, we do it. If not, we fold.
                if current_state.can_act(ActionCheck):
                    return ActionCheck()
                else:
                    return ActionFold()

        # ----------------------------------------------------------------------
        # 3. POST-FLOP PHASES (Flop, Turn, River - Core Logic Phase 2)
        # ----------------------------------------------------------------------
        
        # Phase 3: Track Opponent Bids on the Flop
        if current_state.street == 'flop' and self.pot_before_auction is not None:
            pot_at_flop_start = current_state.pot - current_state.my_wager - current_state.opp_wager
            pot_change = pot_at_flop_start - self.pot_before_auction
            
            if self.last_bid > 0:
                if pot_change == self.last_bid:
                    # Opponent bid higher and won
                    opp_bid = 3000
                    if game_info.round_num <= 100:
                        self.assume_high_mean = True
                elif pot_change == 2 * self.last_bid:
                    # Tie
                    opp_bid = self.last_bid
                    self.assume_high_mean = False
                elif pot_change < self.last_bid:
                    # We won, opponent bid lower
                    opp_bid = pot_change
                    self.assume_high_mean = False
                else:
                    # Fallback
                    opp_bid = 3000
            else:
                # We bid 0, Pot change is 0. Unknown.
                opp_bid = 3000
                
            self.opp_bids.append(opp_bid)
            
            pot_at_acc = self.pot_before_auction if self.pot_before_auction > 0 else 1
            self.opp_bid_percentages.append(opp_bid / pot_at_acc)
            
            self.pot_before_auction = None
            
        # Determine Pot Odds
        call_amount = current_state.cost_to_call
        pot_size = current_state.pot
        pot_odds = call_amount / (pot_size + call_amount) if call_amount > 0 else 0.0
        
        # 3a. Determine our State Advantage by tracking who won the auction
        # If we have 1 revealed card, and they have None (from our perspective), we have Advantage.
        # If both of us have 1 revealed card, it was a Bid Draw (Mutual Disadvantage).
        # We don't have absolute knowledge of what they see, but we know if THEY won the auction,
        # they saw our card.
        
        # Actually, if we bid X, and we DID NOT see their card (opp_revealed_cards is empty), 
        # and the auction happened, it means THEY won and THEY saw our card.
        
        # Let's simplify state detection:
        we_know_their_card = len(current_state.opp_revealed_cards) > 0
        
        # The engine logic from engine.py:
        # If bids match exactly -> both see each other's card.
        # If player A beats player B -> Player A sees Player B's card.
        
        # Let's calculate Equity
        # We use fewer iterations if we know their card because the state space is highly restricted (~50 combinations)
        # We use more iterations if we don't know their card because we search against 1326 combos.
        my_cards = current_state.my_hand
        iterations = 25 if we_know_their_card else 40
        equity = estimate_hand_equity_mc(my_cards, current_state.board, current_state.opp_revealed_cards, iterations=iterations)
        
        target_pot = current_state.pot + current_state.cost_to_call
        
        # ==============================================================================
        # DISADVANTAGE STATE & MUTUAL STATE
        # They won the auction (or tied), so THEY SAW OUR CARD.
        # How do we know they won? If the auction happened, and our bid <= their bid.
        # Since we don't know their bid natively without tracking history from 'auction' street,
        # we will use a heuristic: Did we bid 0? If yes, and auction passed, they likely won.
        # Actually, engine.py logic guarantees we ONLY see `opp_revealed_cards` if we tied or won.
        # Wait, if we TIED, we both see.
        # We need a robust way to know if they saw our card.
        # For now, let's assume we maintain a persistent variable `self.disadvantage_state = True` 
        # if we detect auction loss. 
        # ==============================================================================
        
        # To do this safely right now without modifying __init__ tracking too deeply, we'll
        # trigger "Disadvantage Strategy" if we merely SUSPECT they have a read on us 
        # or if it's a Mutual Draw where BOTH cards are revealed.
        # Mutual Draw = len(current_state.opp_revealed_cards) > 0 AND (we bid exactly what they bid).
        
        # Let's implement the core Post-Flop logic branches:
        
        # DEFENSE: If equity is terrible compared to pot odds, fold immediately.
        # (Margin of 0.05). In disadvantage state, we fold even faster.
        if call_amount > 0 and equity < max(0.25, pot_odds - 0.03):
            # Check for BLOCKER WEAPONIZATION
            # Are we in a bad state, but holding a premium blocker?
            # E.g. We hold the Ace of the suit that just flushed the board.
            if len(current_state.board) >= 3:
                # Count suits on board
                suits_on_board = [c[1] for c in current_state.board]
                from collections import Counter
                suit_counts = Counter(suits_on_board)
                max_suit = suit_counts.most_common(1)[0]
                
                if max_suit[1] >= 3: # There is a flush possibility
                    trigger_suit = max_suit[0]
                    # Do we hold the Ace of that suit but don't have a flush?
                    if f'A{trigger_suit}' in my_cards:
                        # Massive Blocker Wepaonization! 
                        # We represent the nut flush, knowing their MC eval will assume we might have it!
                        if current_state.can_act(ActionRaise) and random.random() < 0.8: # 80% frequency
                            return make_bet(int(target_pot * 1.20), current_state)
                            
            # If no blocker, strictly fold.
            if current_state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()


        # OFFENSE / VALUE SIZING
        if equity > 0.65:
            # High Equity. 
            # Are we downsizing to Trap?
            # Trap Logic: If we are extremely strong, but we suspect they know our card, we downsize to 0.33 pot.
            # Otherwise we bet 0.75 pot to extract standard value.
            # (We apply a 50% random frequency to the trap to remain unpredictable)
            if random.random() < 0.5:
                return make_bet(int(target_pot * 0.33), current_state)
            else:
                return make_bet(int(target_pot * 0.75), current_state)
                
        if equity > 0.50:
            # Standard Value Bet
            return make_bet(int(target_pot * 0.33), current_state)

        # PASSIVE PATH
        if current_state.cost_to_call == 0:
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()
        
        if current_state.cost_to_call > 0 and equity >= pot_odds:
            return ActionCall() if current_state.can_act(ActionCall) else ActionFold()

        return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
