# from multiprocessing.connection import Client
# import action_choices
# from action_choices import *
# from location import RealEstateLocation, UtilityLocation, RailroadLocation
# from agent import Agent
# import logging
# logger = logging.getLogger('monopoly_simulator.logging_info.simple_decision_agent')
#
#
# def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
#     """
#     The agent is in the pre-roll phase and must decide what to do (next). This simple dummy agent skips the turn, and
#      doesn't do anything.
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
#     will always be a subset of the action choices for pre_die_roll in the game schema. Your returned action choice must be from
#     allowable_moves; we will check for this when you return.
#     :param code: See the preamble of this file for an explanation of this code
#     :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
#     parameters that will be passed into the function representing that action when it is executed.
#     The dictionary must exactly contain the keys and expected value types expected by that action in
#     action_choices
#     """
#     print('simple agent pre roll client')
#     if action_choices.skip_turn in allowable_moves:
#
#         return (action_choices.skip_turn, dict())
#     else:
#         logger.error("Exception")
#
#
# def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
#     """
#     The agent is in the out-of-turn phase and must decide what to do (next). This simple dummy agent skips the turn, and
#      doesn't do anything.
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
#     will always be a subset of the action choices for out_of_turn in the game schema. Your returned action choice must be from
#     allowable_moves; we will check for this when you return.
#     :param code: See the preamble of this file for an explanation of this code
#     :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
#     parameters that will be passed into the function representing that action when it is executed.
#     The dictionary must exactly contain the keys and expected value types expected by that action in
#     action_choices
#     """
#     print('simple agent oot client')
#     if action_choices.skip_turn in allowable_moves:
#         return (action_choices.skip_turn, dict())
#     else:
#         logger.error("Exception")
#
#
# def make_post_roll_move(player, current_gameboard, allowable_moves, code):
#     """
#     The agent is in the post-roll phase and must decide what to do (next). This simple dummy agent buys the property if it
#     can afford it, otherwise it skips the turn. If we do buy the property, we end the phase by concluding the turn.
#
#     Note that if your agent decides not to buy the property before concluding the turn, the property will move to
#     auction before your turn formally concludes.
#
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
#     will always be a subset of the action choices for post-die-roll in the game schema. Your returned action choice must be from
#     allowable_moves; we will check for this when you return.
#     :param code: See the preamble of this file for an explanation of this code
#     :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
#     parameters that will be passed into the function representing that action when it is executed.
#     The dictionary must exactly contain the keys and expected value types expected by that action in
#     action_choices
#         """
#     print('simple agent post roll client')
#     current_location = current_gameboard['location_sequence'][player.current_position]
#     if action_choices.buy_property in allowable_moves and current_location.price < player.current_cash:
#         logger.debug(player.player_name+': We will attempt to buy '+current_gameboard['location_sequence'][player.current_position].name+' from the bank.')
#         if code == -1:
#             logger.debug('Did not succeed the last time. Concluding actions...')
#             return (action_choices.concluded_actions, dict())
#         params = dict()
#         params['player'] = player
#         params['asset'] = current_gameboard['location_sequence'][player.current_position]
#         params['current_gameboard'] = current_gameboard
#         return (action_choices.buy_property, params)
#     elif action_choices.concluded_actions in allowable_moves:
#         return (action_choices.concluded_actions, dict())
#     else:
#         logger.error("Exception")
#
#
# def make_buy_property_decision(player, current_gameboard, asset):
#     """
#     The decision to be made when the player lands on a location representing a purchaseable asset that is currently
#     owned by the bank. The dummy agent here returns True only if its current cash reserves are not less than the
#     asset's current price. A more sophisticated agent would consider other features in current_gameboard, including
#     whether it would be able to complete the color-set by purchasing the asset etc.
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :return: A Boolean. If True, then you decided to purchase asset from the bank, otherwise False. We allow you to
#     purchase the asset even if you don't have enough cash; however, if you do you will end up with a negative
#     cash balance and will have to handle that if you don't want to lose the game at the end of your move (see notes
#     in handle_negative_cash_balance)
#     """
#     print('simple agent buy client')
#     decision = False
#     if player.current_cash >= asset.price:
#         decision = True
#     return decision
#
#
# def make_bid(player, current_gameboard, asset, current_bid):
#     """
#     Decide the amount you wish to bid for asset in auction, given the current_bid that is currently going. If you don't
#     return a bid that is strictly higher than current_bid you will be removed from the auction and won't be able to
#     bid anymore. Note that it is not necessary that you are actually on the location on the board representing asset, since
#     you will be invited to the auction automatically once a player who lands on a bank-owned asset rejects buying that asset
#     (this could be you or anyone else).
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :param asset: An purchaseable instance of Location (i.e. real estate, utility or railroad)
#     :param current_bid: The current bid that is going in the auction. If you don't bid higher than this amount, the bank
#     will remove you from the auction proceedings. You could also always return 0 to voluntarily exit the auction.
#     :return: An integer that indicates what you wish to bid for asset
#     """
#     print('simple agent bid client')
#     if current_bid < asset.price:
#         new_bid = current_bid + (asset.price-current_bid)/2
#         if new_bid < player.current_cash:
#             return new_bid
#         else:   # We are aware that this can be simplified with a simple return 0 statement at the end. However in the final baseline agent
#                 # the return 0's would be replaced with more sophisticated rules. Think of them as placeholders.
#             return 0 # this will lead to a rejection of the bid downstream automatically
#     else:
#         return 0 # this agent never bids more than the price of the asset
#
#
#
#
# def handle_negative_cash_balance(player, current_gameboard):
#     """
#     You have a negative cash balance at the end of your move (i.e. your post-roll phase is over) and you must handle
#     this issue before we move to the next player's pre-roll. If you do not succeed in restoring your cash balance to
#     0 or positive, bankruptcy proceeds will begin and you will lost the game.
#
#     The dummy agent in this case just decides to go bankrupt by returning -1. A more sophisticated agent would try to
#     do things like selling houses and hotels, properties etc. You must invoke all of these functions yourself since
#     we want to give you maximum flexibility when you are in this situation. Once done, return 1 if you believe you
#     succeeded (see the :return description for a caveat on this)
#
#     :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
#     instantiated with the functions specified by this decision agent).
#     :param current_gameboard: A dict. The global data structure representing the current game board.
#     :return: -1 if you do not try to address your negative cash balance, or 1 if you tried and believed you succeeded.
#     Note that even if you do return 1, we will check to see whether you have non-negative cash balance. The rule of thumb
#     is to return 1 as long as you 'try', or -1 if you don't try (in which case you will be declared bankrupt and lose the game)
#     """
#     print('simple agent handle neg cash client')
#     return -1
#
#
# def _build_decision_agent_methods_dict():
#     """
#     This function builds the decision agent methods dictionary.
#     :return: The decision agent dict. Keys should be exactly as stated in this example, but the functions can be anything
#     as long as you use/expect the exact function signatures we have indicated in this document.
#     """
#     ans = dict()
#     ans['handle_negative_cash_balance'] = handle_negative_cash_balance
#     ans['make_pre_roll_move'] = make_pre_roll_move
#     ans['make_out_of_turn_move'] = make_out_of_turn_move
#     ans['make_post_roll_move'] = make_post_roll_move
#     ans['make_buy_property_decision'] = make_buy_property_decision
#     ans['make_bid'] = make_bid
#     ans['type'] = "decision_agent_methods"
#     return ans
#
#
# decision_agent_methods = _build_decision_agent_methods_dict() # this is the main data structure that is needed by gameplay
#
#
# class ClientAgent(Agent):
#     """
#     An Agent that can be sent requests from a ServerAgent and send back desired moves. Instantiate the client like you
#     would a normal agent, either by passing in the functions to the constructor or by subclassing. The ClientAgent
#     can be used for local play as usual, or if the game has been set up with a ServerAgent, call play_remote_game
#     to have the client receive game information from the server and send back desired moves.
#     """
#
#     def __init__(self):
#         super().__init__(**decision_agent_methods)
#         self.conn = None
#         self.logger = logger
#         self.game_num = 0
#
#     def play_remote_game(self, address=('localhost', 6010), authkey=b"password"):
#         """
#         Connects to a ServerAgent and begins the loop of waiting for requests and responding to them.
#         @param address: Tuple, the address and port number. Defaults to localhost:6000
#         @param authkey: Byte string, the password used to authenticate the client. Must be same as server's authkey.
#             Defaults to "password"
#         """
#
#         self.conn = Client(address, authkey=authkey)
#         result = None
#         while True:
#             func_name, args = self.conn.recv()  # Receive the signal
#
#             # When the tournament begins, we need to
#             if func_name == "start_tournament":
#                 self.logger.info('Tournament starts!')
#                 result = 1
#
#             # Before simulating each game, we have to make sure if we need retrain the network
#             elif func_name == "startup": # args = (current_gameboard, indicator)
#                 # Clear interface history and set the init for interface
#                 self.game_num += 1
#                 self.logger.info(str(self.game_num) + ' th game starts!')
#
#             # When each game ends, we run the KG, but we don not shutdown the connection
#             elif func_name == 'shutdown':
#                 result = 1
#                 self.logger.info(str(self.game_num) + ' th game stops!')
#
#             # When calling agent to make decision
#             elif func_name == 'make_post_roll_move':  # args = (player, current_gameboard, allowable_moves, code)
#                 player, current_gameboard, allowable_moves, code = args
#                 result = make_post_roll_move(player, current_gameboard, allowable_moves, code)
#
#             elif func_name == 'make_out_of_turn_move':  # args = (player, current_gameboard, allowable_moves, code)
#                 player, current_gameboard, allowable_moves, code = args
#                 result = make_out_of_turn_move(player, current_gameboard, allowable_moves, code)
#
#             elif func_name == 'make_pre_roll_move':  # args = (player, current_gameboard, allowable_moves, code)
#                 player, current_gameboard, allowable_moves, code = args
#                 result = make_pre_roll_move(player, current_gameboard, allowable_moves, code)
#
#             elif func_name == 'make_bid':  # args = (player, current_gameboard, allowable_moves, code)
#                 player, current_gameboard, asset, current_bid = args
#                 result = make_bid(player, current_gameboard, asset, current_bid)
#
#             elif func_name == 'make_buy_property_decision':  # args = (player, current_gameboard, allowable_moves, code)
#                 player, current_gameboard, asset = args
#                 result = make_buy_property_decision(player, current_gameboard, asset)
#
#             elif func_name == 'handle_negative_cash_balance':  # args = (player, current_gameboard, allowable_moves, code)
#                 player, current_gameboard = args
#                 result = handle_negative_cash_balance(player, current_gameboard)
#
#             # Send we will close the connection now back to server
#             elif func_name == "end_tournament":
#                 result = 1
#                 self.logger.info('Tournament Finished!')
#
#             else:
#                 result = getattr(self, func_name)(*args)
#
#             # Send the results back to server agent
#             self.conn.send(result)
#
#             # Close connection after each tournament
#             if func_name == "end_tournament":
#                 self.conn.close()
#                 break
#
#
# def main():
#     client = ClientAgent()
#     client.play_remote_game()
#
#
# if __name__ == "__main__":
#     main()
#
