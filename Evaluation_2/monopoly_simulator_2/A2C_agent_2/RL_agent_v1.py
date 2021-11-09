import os, sys
upper_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(upper_path)
sys.path.append('/datadrive/software/Gatech_Agent_Eva_2/Evaluation_2/monopoly_simulator_2')
from monopoly_simulator import action_choices
import A2C_agent_2.agent_helper_functions_agent as agent_helper_functions # helper functions are internal to the agent and will not be recorded in the function log.
from monopoly_simulator import diagnostics
# from monopoly_simulator import hypothetical_simulator
from A2C_agent_2.interface_eva import Interface_eva
import torch
from A2C_agent_2.vanilla_A2C import *
from configparser import ConfigParser
# from novelty_detection import KG_OpenIE
import pickle
from monopoly_simulator.flag_config import flag_config_dict

import logging
logger = logging.getLogger('monopoly_simulator.logging_info.rl_agent')
UNSUCCESSFUL_LIMIT = 2
# export PYTHONHASHSEED=0
"""
All external decision_agent functions must have the exact signatures we have indicated in this document. Beyond
that, we impose no restrictions (you can make the decision agent as complex as you like (including maintaining state),
and we use good faith to ensure you do not manipulate the gameboard. We will have mechanisms to check for inadvertent
changes or inconsistencies that get introduced in the gameboard (due to any reason, including possible subtle errors
in the simulator itself) a short while later.

If you decision agent does maintain state, or some kind of global data structure, please be careful when assigning the
same decision agent (as we do) to each player. We do provide some basic state to you already via 'code' in the make_*_move
functions. Specifically, if code is 1 it means the 'previous' move selected by the player was successful,
and if -1 it means it was unsuccessful. code of -1 is usually returned when an allowable move is invoked with parameters
that preempt the action from happening e.g., the player may decide to mortgage property that is already mortgaged,
which will return the failure code of -1 when the game actually tries to mortgage the property in monopoly_simulator.action_choices.

Be careful to note what each function is supposed to return in addition to adhering to the expected signature. The examples
here are good guides.

Your functions can be called whatever you like, but the keys in decision_agent_methods should not be changed. The
respective functions must adhere in their signatures to the examples here. The agent in this file is simple and rule-based,
 rather than adaptive but capable of taking good actions in a number of eventualities.
 We detail the logic behind each decision in a separate document. This is the agent that will serve as the 'background'
 agent for purposes of evaluation.

"""


def make_pre_roll_move(serial_dict_to_client, memory):
    """
    Many actions are possible in pre_roll but we prefer to save the logic for out_of_turn. The only decision
    we'll make here is whether we want to leave jail (if we're in jail).
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param allowable_moves: A set of functions, each of which is defined in monopoly_simulator.action_choices (imported in this file), and that
    will always be a subset of the action choices for pre_die_roll in the game schema. Your returned action choice must be from
    allowable_moves; we will check for this when you return.
    :param code: See the preamble of this file for an explanation of this code
    :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
    parameters that will be passed into the function representing that action when it is executed.
    The dictionary must exactly contain the keys and expected value types expected by that action in
    monopoly_simulator.action_choices
    """
    ###
    player_name = serial_dict_to_client['player']
    current_gameboard = serial_dict_to_client['current_gameboard']
    player = current_gameboard['players'][player_name]
    allowable_moves = serial_dict_to_client['allowable_moves']
    code = serial_dict_to_client['code']
    return_to_server_dict = dict()
    # ###
    # if player['current_cash'] >= 100: # if we don't have enough money, best to stay put.
    #     param = dict()
    #     param['player'] = player
    #     param['current_gameboard'] = current_gameboard
    #     if 'use_get_out_of_jail_card' in allowable_move_names:
    #         logger.debug(player.player_name+': I am using get out of jail card.')
    #         # player.agent._agent_memory['previous_action'] = action_choices.use_get_out_of_jail_card
    #         return ('use_get_out_of_jail_card', param)
    #     elif 'pay_jail_fine' in allowable_move_names:
    #         logger.debug(player.player_name+': I am going to pay jail fine.')
    #         # player.agent._agent_memory['previous_action'] = action_choices.pay_jail_fine
    #         return ('pay_jail_fine', param)
    #
    # # # if we ran the gamut, and did not return, then it's time to skip turn or conclude actions
    # if 'skip_turn' in allowable_move_names:
    #     # testing hypothetical simulator (will comment when done testing). Note that this was written for the Python 2
    #     # version (the GNOME repo). Make sure to appropriately modify by instantiating agent instead of sending in the
    #     # decision agent methods as being done below.
    #     # player_decision_agents = dict()
    #     # import simple_decision_agent_1
    #     # player_decision_agents['player_1'] = simple_decision_agent_1.decision_agent_methods # the reason I am doing this for all agents is to avoid infinite loops.
    #     # player_decision_agents['player_2'] = simple_decision_agent_1.decision_agent_methods
    #     # player_decision_agents['player_3'] = simple_decision_agent_1.decision_agent_methods
    #     # player_decision_agents['player_4'] = simple_decision_agent_1.decision_agent_methods
    #     # alternate_univ = hypothetical_simulator.initialize_hypothetical_universe(current_gameboard, player_decision_agents)
    #     # logger.debug(player.player_name,' has spawned alternate universe to try out things.')
    #     # hypothetical_winner = hypothetical_simulator.simulate_hypothetical_game(hypothetical_gameboard=alternate_univ,
    #     #                         die_roll_substitute=hypothetical_simulator.die_roll_substitute,num_total_die_rolls=15) # we will only run for fifteen die rolls.
    #     # if hypothetical_winner is None:
    #     #     logger.debug(diagnostics.logger.debug_player_cash_balances(alternate_univ))
    #     # else:
    #     #     logger.debug(hypothetical_winner.player_name)
    #     logger.debug(player.player_name + ': I am skipping turn')
    #     # player.agent._agent_memory['previous_action'] = action_choices.skip_turn
    #     return ('skip_turn', dict())
    # elif 'concluded_actions' in allowable_move_names:
    #     # player.agent._agent_memory['previous_action'] = action_choices.concluded_actions
    #     logger.debug(player.player_name + ': I am concluding actions')
    #     return ('concluded_actions', dict())
    # else:
    #     logger.error("Exception")

    if player['current_cash'] >= 100:  # if we don't have enough money, best to stay put.
        param = dict()
        param['player'] = player_name
        param['current_gameboard'] = "current_gameboard"
        if "use_get_out_of_jail_card" in allowable_moves:
            logger.debug(player_name+': I am using get out of jail card.')
            return_to_server_dict['function'] = "use_get_out_of_jail_card"
            return_to_server_dict['param_dict'] = param
            memory['previous_action'] = "use_get_out_of_jail_card"
            return return_to_server_dict, memory
        elif "pay_jail_fine" in allowable_moves:
            logger.debug(player_name+': I am going to pay jail fine.')
            return_to_server_dict['function'] = "pay_jail_fine"
            return_to_server_dict['param_dict'] = param
            memory['previous_action'] = "pay_jail_fine"
            return return_to_server_dict, memory

    # if we ran the gamut, and did not return, then it's time to skip turn or conclude actions
    if "skip_turn" in allowable_moves:
        # testing hypothetical simulator (will comment when done testing). Note that this was written for the Python 2
        # version (the GNOME repo). Make sure to appropriately modify by instantiating agent instead of sending in the
        # decision agent methods as being done below.
        # player_decision_agents = dict()
        # import simple_decision_agent_1
        # player_decision_agents['player_1'] = simple_decision_agent_1.decision_agent_methods # the reason I am doing this for all agents is to avoid infinite loops.
        # player_decision_agents['player_2'] = simple_decision_agent_1.decision_agent_methods
        # player_decision_agents['player_3'] = simple_decision_agent_1.decision_agent_methods
        # player_decision_agents['player_4'] = simple_decision_agent_1.decision_agent_methods
        # alternate_univ = hypothetical_simulator.initialize_hypothetical_universe(current_gameboard, player_decision_agents)
        # logger.debug(player.player_name,' has spawned alternate universe to try out things.')
        # hypothetical_winner = hypothetical_simulator.simulate_hypothetical_game(hypothetical_gameboard=alternate_univ,
        #                         die_roll_substitute=hypothetical_simulator.die_roll_substitute,num_total_die_rolls=15) # we will only run for fifteen die rolls.
        # if hypothetical_winner is None:
        #     logger.debug(diagnostics.logger.debug_player_cash_balances(alternate_univ))
        # else:
        #     logger.debug(hypothetical_winner.player_name)
        logger.debug(player_name+ ': I am skipping turn')
        memory['previous_action'] = "skip_turn"
        return_to_server_dict['function'] = "skip_turn"
        return_to_server_dict['param_dict'] = dict()
        memory['previous_action'] = "skip_turn"
        return return_to_server_dict, memory

    elif "concluded_actions" in allowable_moves:
        memory['previous_action'] = "concluded_actions"
        logger.debug(player_name+ ': I am concluding actions')
        return_to_server_dict['function'] = "concluded_actions"
        return_to_server_dict['param_dict'] = dict()
        return return_to_server_dict, memory

    else:
        logger.error("Exception")
        raise Exception

def make_out_of_turn_move(serial_dict_to_client, memory, mortgage_percentage, go_increment):

    """
    The agent is in the out-of-turn phase and must decide what to do (next). This simple dummy agent skips the turn, and
     doesn't do anything.
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
    will always be a subset of the action choices for out_of_turn in the game schema. Your returned action choice must be from
    allowable_moves; we will check for this when you return.
    :param code: See the preamble of this file for an explanation of this code
    :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
    parameters that will be passed into the function representing that action when it is executed.
    The dictionary must exactly contain the keys and expected value types expected by that action in
    monopoly_simulator.action_choices
    """
    '''
    Agent V3
    This updated version of the agent can make trade offers with MULTIPLE players simultaneously.
    All strategies available in Agent V2 is still available in V3
    Note that this version of the agent also engages in the trade of only one set of properties like the previous version, ie 
    - only one property will be requested if it is a buy property offer or
    - only one property will be offered if it is a sell property offer or
    - only one property will be offered and one property requested during an exchange property offer.

    Agent V2
    NOTE: The background agent that could make_sell_property_offer is deprecated (available as background_agent_v1_deprecated.py)
    This version of the agent can only make_trade_offer and accept trade offer. Trade involves buy or sell or exchange property offers.
    Accept_sell_property_offer function is still available in case some different agent decides to make a sell property offer.
    Ideally, accept_sell_property_offer() function should never enter allowable moves.
    Make sell property offer can be replicated by making a trade offer that only offers to sell properties in return for cash
    and doesnot involve a buy property or exchange property offer.
    A buy property offer can be duplicated by including only requested properties by offering cash without offering properties.
    Properties and cash can be exchanged which lets both players get an advantage of increasing their respective number of monopolies.
    This version of the agent background_agent_v1 supports making sell property offers in return for cash via make_trade_offer, 
    buy trade offers and exchange property offers.
    Note that this version of the agent engages in the trade of only one set of properties, ie 
    - only one property will be requested if it is a buy property offer or
    - only one property will be offered if it is a sell property offer or
    - only one property will be offered and one property requested during an exchange property offer.

    '''
    player_name = serial_dict_to_client['player']
    current_gameboard = serial_dict_to_client['current_gameboard']
    player = current_gameboard['players'][player_name]
    allowable_moves = serial_dict_to_client['allowable_moves']
    code = serial_dict_to_client['code']

    # if isinstance(code, list):
    #     code_flag = 0
    #     for c in code:
    #         if c == flag_config_dict['failure_code']:
    #             code_flag = 1
    #             break
    #     if code_flag:
    #         if 'count_unsuccessful_tries' in memory:
    #             memory['count_unsuccessful_tries'] += 1
    #         else:
    #             memory['count_unsuccessful_tries'] = 1
    #         logger.debug(player['player_name'] + ' has executed an unsuccessful out of turn action, incrementing unsuccessful_tries ' +
    #                                       'counter to ' + str(memory['count_unsuccessful_tries']))
    # elif code == flag_config_dict['failure_code']:
    #     if 'count_unsuccessful_tries' in memory:
    #         memory['count_unsuccessful_tries'] += 1
    #     else:
    #         memory['count_unsuccessful_tries'] = 1
    #     logger.debug(player['player_name'] + ' has executed an unsuccessful out of turn action, incrementing unsuccessful_tries ' +
    #                                       'counter to ' + str(memory['count_unsuccessful_tries']))
    #
    # if 'count_unsuccessful_tries' in memory and memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
    #     logger.debug('player_1' + ' has reached out of turn unsuccessful action limits.')
    #     if "skip_turn" in allowable_moves:
    #         logger.debug('player_1' + ': I am skipping turn since I have crossed unsuccessful limits.')
    #         memory['previous_action'] = "skip_turn"
    #         return ("skip_turn", dict()), memory
    #     elif "concluded_actions" in allowable_moves:
    #         memory['previous_action'] = action_choices.concluded_actions
    #         logger.debug('player_1'+ ': I am concluding actions since I have crossed unsuccessful limits.')
    #         return ("concluded_actions", dict()), memory
    #     else:
    #         logger.error("Exception")
    #         raise Exception


    # if "accept_trade_offer" in allowable_moves:
    #     param = dict()
    #     param['player'] = 'player_1'
    #     param['current_gameboard'] = "current_gameboard"
    #     logger.debug('(player_1 currently has cash balance of '+str(player['current_cash']) + ')')
    #
    #     if (player['outstanding_trade_offer']['cash_offered'] <= 0 and len(player['outstanding_trade_offer']['property_set_offered'])==0) and \
    #             (player['outstanding_trade_offer']['cash_wanted'] > 0 or len(player['outstanding_trade_offer']['property_set_wanted']) > 0):
    #         logger.debug('Asking for free money or property without money or property in return.')
    #         logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #         pass   #asking for free money or property without anything in return(ie no money and no property offered), -->reject the trade offer
    #
    #     elif player['outstanding_trade_offer']['cash_wanted'] - player['outstanding_trade_offer']['cash_offered'] > player['current_cash']:
    #         logger.debug('Cash wanted from me in the trade offer is more than the cash in hand with me or I am near bankruptcy situation and need to play safe.')
    #         logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #         pass  #cash wanted is more than that offered and the net difference exceeds the cash that the player has --> then reject the tade offer
    #
    #     else:
    #         reject_flag = 0
    #         offered_properties_net_worth = 0
    #         wanted_properties_net_worth = 0
    #         for prop in player['outstanding_trade_offer']['property_set_wanted']:
    #             prop = current_gameboard['locations'][prop]
    #             if prop['is_mortgaged']: #
    #                 reject_flag = 1  #cannot trade mortgaged properties, reject trade offer
    #                 logger.debug('Trade offer invovlves mortgaged properties.')
    #                 logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                 break
    #             else:
    #                 wanted_properties_net_worth += prop['price']
    #
    #         if reject_flag == 0:
    #             for prop in player['outstanding_trade_offer']['property_set_offered']:
    #                 prop = current_gameboard['locations'][prop]
    #                 if prop['is_mortgaged']:
    #                     reject_flag = 1  #from_player cannot offer mortgaged properties, reject trade offer
    #                     logger.debug('Trade offer invovlves mortgaged properties.')
    #                     logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                     break
    #                 else:
    #                     offered_properties_net_worth += prop['price']
    #         if reject_flag == 0:
    #             #GOAL -- increase monopolies
    #             #calculate the net worth of offer vs net worth of request --> makes sense to accept trade only if the offer is greater than request
    #             #net worth of offer = cash + total price of all houses
    #             #positive net_amount_requested implies that the requested net amount is greater than offered net amount
    #             net_offer_worth = (offered_properties_net_worth + player['outstanding_trade_offer']['cash_offered']) - \
    #                               (wanted_properties_net_worth + player['outstanding_trade_offer']['cash_wanted'])
    #             net_amount_requested = -1 * net_offer_worth
    #
    #             count_create_new_monopoly = 0
    #             count_lose_existing_monopoly = 0 ##ideally player doesnot have to worry about losing monopolies since the player who makes the offer
    #             #only requests for lone properties
    #             for prop in player['outstanding_trade_offer']['property_set_offered']:
    #                 prop = current_gameboard['locations'][prop]
    #                 if agent_helper_functions.will_property_complete_set(player,prop,current_gameboard):
    #                     count_create_new_monopoly += 1
    #             for prop in player['outstanding_trade_offer']['property_set_wanted']:
    #                 prop = current_gameboard['locations'][prop]
    #                 if prop['color'] in player['full_color_sets_possessed']:
    #                     count_lose_existing_monopoly += 1
    #
    #             #if you end up losing more monopolies than gaining monopolies (although this condition should never come up) then reject trade offer
    #             if count_lose_existing_monopoly - count_create_new_monopoly > 0:
    #                 logger.debug('Player loses more monopolies than he gains.')
    #                 logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                 reject_flag = 1
    #
    #             #if you end up losing the same number of monopolies as you gain, then accept the offer based on the following multiple conditions.
    #             #Basically you get no new monopolies since ideally you dont lose monopolies (only properties that dont belong to your monopolized color
    #             # groups are only requested from you in the trade.)
    #             elif count_lose_existing_monopoly - count_create_new_monopoly == 0:
    #                 if (player['outstanding_trade_offer']['cash_wanted'] - player['outstanding_trade_offer']['cash_offered']) >= player['current_cash']:
    #                     logger.debug('Cash wanted from me in the trade offer is more than the cash in hand with me or I am near bankruptcy situation and need to play safe.')
    #                     logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                     reject_flag = 1  ##just double checking although this condition was verified before getting here.
    #                 elif player['current_cash'] - (player['outstanding_trade_offer']['cash_wanted'] - player['outstanding_trade_offer']['cash_offered']) < 150/2:
    #                     logger.debug('Cash wanted from me in the trade offer is more than the cash in hand with me or I am near bankruptcy situation and need to play safe.')
    #                     logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                     reject_flag = 1  ##too risky if players cash after transaction drops below half of go_increment value --> hence reject trade offer
    #                 elif (player['current_cash'] - (player['outstanding_trade_offer']['cash_wanted'] - player['outstanding_trade_offer']['cash_offered']) < 150) \
    #                         and net_offer_worth <= 0:
    #                     logger.debug('No gain from accepting trade offer.')
    #                     logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                     reject_flag =1  ##if player has cash > go_increement/2 and < go_increement but net worth of total transaction is negative --> reject trade offer
    #                 else:
    #                     reject_flag =0  ##accept only if you end up getting a higher net worth by accepting the trade although you get no new monopolies
    #
    #
    #             #else you get to monopolize more locations than you had before --> then ACCEPT THE TRADE OFFER
    #             elif count_create_new_monopoly - count_lose_existing_monopoly > 0:
    #                 if (player['outstanding_trade_offer']['cash_wanted'] - player['outstanding_trade_offer']['cash_offered']) >= player['current_cash']:
    #                     logger.debug('Cash wanted from me in the trade offer is more than the cash in hand with me or I am near bankruptcy situation and need to play safe.')
    #                     logger.debug('player_1' + " rejected trade offer from " + player['outstanding_trade_offer']['from_player'])
    #                     reject_flag = 1  ##just double checking although this condition was verified before getting here.
    #                 else:
    #                     reject_flag = 0
    #
    #         if reject_flag == 0:
    #             logger.debug('player_1' + " accepted trade offer from " + player['outstanding_trade_offer']['from_player'])
    #             logger.debug('player_1' + " recieved amount = " + str(player['outstanding_trade_offer']['cash_offered']) + " and offered amount = " +
    #                          str(player['outstanding_trade_offer']['cash_wanted']) + " during trade")
    #             memory['previous_action'] = "accept_trade_offer"
    #             return ("accept_trade_offer", param), memory
    #         elif reject_flag == 1:
    #             #logger.debug('player_1' + " rejected trade offer from " + player.outstanding_trade_offer['from_player'].player_name)
    #             pass

    if "accept_sell_property_offer" in allowable_moves:
        ## Ideally accept_sell_offer should never enter allowable moves since henceforth make_trade_offer also takes care of make_sell_offer and
        ## accept_trade_offer takes care of accept_sell_offer.
        ## This case is included to accomodate a make_sell_property offer raised by an external agent.
        ## Our agent will never make a sell property offer, only makes trade offers which raises an accpet_trade_offer action.
        param = dict()
        param['player'] = 'player_1'
        param['current_gameboard'] = "current_gameboard"
        # we accept an offer under one of two conditions:
        # logger.debug('player_1'+ ': Should I accept the offer by '+player.outstanding_property_offer['from_player'].player_name+' to buy '+\
        #     player.outstanding_property_offer['asset'].name+' for '+str(player.outstanding_property_offer['price'])+'?')
        logger.debug('player_1'+ 'currently has cash balance of '+str(player['current_cash'])+')')
        if player['outstanding_property_offer']['asset'].is_mortgaged or player.outstanding_property_offer['price']>player.current_cash:
            pass # ignore the offer if the property is mortgaged or will result in insolvency. This pass doesn't require 'filling' in.
        elif player['current_cash'] -player['outstanding_property_offer']['price'] >= go_increment and \
            player['outstanding_property_offer']['price']<=player['outstanding_property_offer']['asset']['price']:
            # 1. we can afford it, and it's at or below market rate so let's buy it
            logger.debug('player_1'+ ': I am accepting the offer to buy '+player['outstanding_property_offer']['asset']+' since I can afford' \
                                                    'it and it is being offered at or below market rate.')
            memory['previous_action'] = "accept_sell_property_offer"
            return ("accept_sell_property_offer", param), memory

        elif agent_helper_functions.will_property_complete_set(player, player.outstanding_property_offer['asset'],current_gameboard):
            # 2. less affordable, but we stand to gain by monopoly
            if player.current_cash - player.outstanding_property_offer['price'] >= go_increment / 2: # risky, but worth it
                logger.debug('player_1'+ ': I am accepting the offer to buy '+ player['outstanding_property_offer'][
                    'asset']['name']+ ' since I can afford ' \
                                   'it (albeit barely so) and it will let me complete my color set.')
                memory['previous_action'] = "accept_sell_property_offer"
                return ("accept_sell_property_offer", param), memory

    if player['status'] != 'current_move': # these actions are considered only if it's NOT our turn to roll the dice.
        if "improve_property" in allowable_moves: # beef up full color sets to maximize rent potential.
            param = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
            if param:

                logger.debug('player_1' + ': I am going to improve property '+param['asset']['name'])
                memory['previous_action'] = "improve_property"
                param['player'] = param['player']
                param['asset'] = param['asset']['name']
                param['current_gameboard'] = "current_gameboard"
                return ("improve_property", param), memory

        player_mortgaged_assets_list = list()
        if player['mortgaged_assets']:
            player_mortgaged_assets_list = _set_to_sorted_list_mortgaged_assets(player['mortgaged_assets'])
        for m in player_mortgaged_assets_list:
            m = current_gameboard['locations'][m]
            if player['current_cash'] - (m['mortgage']* (1+ mortgage_percentage)) >= go_increment and "free_mortgage" in allowable_moves:
                # free mortgages till we can afford it. the second condition should not be necessary but just in case.
                param = dict()
                param['player'] = 'player_1'
                param['asset'] = m['name']
                param['current_gameboard'] = "current_gameboard"
                logger.debug('player_1'+ ': I am going to free mortgage on '+ m['name'])
                memory['previous_action'] = "free_mortgage"
                return ("free_mortgage", param), memory

    else:
        #purpose_flags are sent while curating a trade offer to imply why the trade offer was made:
         ## 1 --> low on cash, urgently in need of cash
         ## 2 --> gain monopoly

        if player['current_cash'] < go_increment and "make_trade_offer" in allowable_moves:
            # in this case, the trade offer is a duplication of make_sell_property_offer since the player is in urgent need of cash and
            #cannot strategize a trade
            potential_offer_list = agent_helper_functions.identify_property_trade_offer_to_player(player, current_gameboard)
            potential_request_list = agent_helper_functions.identify_property_trade_wanted_from_player(player, current_gameboard)
            param_list = agent_helper_functions.curate_trade_offer_multiple_players(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag=1)
            #logger.debug(param)
            return_action_list = []
            return_param_list = []

            if param_list and memory['previous_action'] != "make_trade_offer": # we only make one offer per turn. Otherwise we'd
                # be stuck in a loop
                if len(param_list)>1:
                    logger.debug('player_1' + ": I am going to make trade offers to multiple players, ie " + str(len(param_list)) + " players.")
                for param in param_list:
                    # logger.debug('player_1' + ': I am making an offer to trade '+list(param['offer']['property_set_offered'])[0]+' to '+
                    #              param['to_player']+' for '+str(param['offer']['cash_wanted'])+' dollars')

                    param['from_player'] = param['from_player']['player_name']
                    param['to_player'] = param['to_player']['player_name']
                    prop_set_offered = []
                    for item in param['offer']['property_set_offered']:
                        prop_set_offered.append(item)
                    param['offer']['property_set_offered'] = prop_set_offered
                    prop_set_wanted = []
                    for item in param['offer']['property_set_wanted']:
                        prop_set_wanted.append(item)
                    param['offer']['property_set_wanted'] = prop_set_wanted

                    memory['previous_action'] = "make_trade_offer"
                    return_action_list.append("make_trade_offer")
                    return_param_list.append(param)
                return (return_action_list, return_param_list), memory

        elif "make_trade_offer" in allowable_moves:
            # trade offer is being curated to maximise monopolies
            potential_offer_list = agent_helper_functions.identify_property_trade_offer_to_player(player, current_gameboard)
            potential_request_list = agent_helper_functions.identify_property_trade_wanted_from_player(player, current_gameboard)
            param_list = agent_helper_functions.curate_trade_offer_multiple_players(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag=2)
            #logger.debug(param)
            return_action_list = []
            return_param_list = []
            if param_list and memory['previous_action'] != "make_trade_offer":  # we only make one offer per turn. Otherwise we'd
                # be stuck in a loop
                if len(param_list)>1:
                    logger.debug('player_1' + ": I am going to make trade offers to multiple players, ie " + str(len(param_list)) + " players.")
                for param in param_list:
                    logger.debug('player_1'+ ': I am making a trade offer with '+ param['to_player']['player_name'])

                    param['from_player'] = param['from_player']['player_name']
                    param['to_player'] = param['to_player']['player_name']
                    prop_set_offered = []
                    for item in param['offer']['property_set_offered']:
                        prop_set_offered.append(item)
                    param['offer']['property_set_offered'] = prop_set_offered
                    prop_set_wanted = []
                    for item in param['offer']['property_set_wanted']:
                        prop_set_wanted.append(item)
                    param['offer']['property_set_wanted'] = prop_set_wanted

                    memory['previous_action'] = "make_trade_offer"
                    return_action_list.append("make_trade_offer")
                    return_param_list.append(param)
                return (return_action_list, return_param_list), memory


    # if we ran the gamut, and did not return, then it's time to skip turn or conclude actions
    if "skip_turn" in allowable_moves:
        logger.debug('player_1'+ ': I am skipping turn')
        memory['previous_action'] = "skip_turn"
        return ("skip_turn", dict()), memory
    elif "concluded_actions" in allowable_moves:
        logger.debug('player_1'+ ': I am concluding actions')
        memory['previous_action'] = action_choices.concluded_actions
        return ("concluded_actions", dict()), memory
    else:
        logger.error("Exception")
        raise Exception



def make_post_roll_move(serial_dict_to_client, memory, go_increment):
    """
    The agent is in the post-roll phase and must decide what to do (next). The main decision we make here is singular:
    should we buy the property we landed on, if that option is available?

    --If we do buy the property, we end the phase by concluding the turn.

    --If we cannot buy a property, we conclude the turn. If we have negative cash balance, we do not handle it here, but
    in the handle_negative_cash_balance function. This means that the background agent never calls any of
    the mortgage or sell properties here UNLESS we need to mortgage or sell a property in order to buy the current
     one and it is well worth our while.

    Note that if your agent decides not to buy the property before concluding the turn, the property will move to
    auction before your turn formally concludes.

    This background agent never sells a house or hotel in post_roll.

    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param allowable_moves: A set of functions, each of which is defined in monopoly_simulator.action_choices (imported in this file), and that
    will always be a subset of the action choices for post-die-roll in the game schema. Your returned action choice must be from
    allowable_moves; we will check for this when you return.
    :param code: See the preamble of this file for an explanation of this code
    :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
    parameters that will be passed into the function representing that action when it is executed.
    The dictionary must exactly contain the keys and expected value types expected by that action in
    monopoly_simulator.action_choices
        """
    player_name = serial_dict_to_client['player']
    current_gameboard = serial_dict_to_client['current_gameboard']
    player = current_gameboard['players'][player_name]
    allowable_moves = serial_dict_to_client['allowable_moves']
    code = serial_dict_to_client['code']
    return_to_server_dict = dict()

    if 'phase_game' not in memory:
        memory['phase_game'] = 2
        memory['count_unsuccessful_tries'] = 0

    if memory['phase_game'] != 2:
        memory['phase_game'] = 2
        if player['status'] != 'lost':
            memory['count_unsuccessful_tries'] = 0

    if code == flag_config_dict['failure_code']:
        memory['count_unsuccessful_tries'] += 1
        logger.debug('player_1' + ' has executed an unsuccessful postroll action, incrementing unsuccessful_tries ' +
                                          'counter to ' + str(memory['count_unsuccessful_tries']))

    if memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
        logger.debug('player_1' + ' has reached postroll unsuccessful action limits.')
        if "concluded_actions" in allowable_moves:
            memory['previous_action'] = "concluded_actions"
            logger.debug('player_1' + ': I am concluding actions since I have crossed unsuccessful limits.')
            return ("concluded_actions", dict()), memory
        else:
            logger.error("Exception")
            raise Exception

    current_location = current_gameboard['locations'][current_gameboard['location_sequence'][player['current_position']]]
    if "buy_property" in allowable_moves:
        if code == flag_config_dict['failure_code']:
            logger.debug('player_1' +': I did not succeed the last time in buying this property. Concluding actions...')
            return ("concluded_actions", dict()), memory

        params = dict()
        params['player'] = 'player_1'
        params['asset'] = current_location['name']
        params['current_gameboard'] = "current_gameboard"

        if make_buy_property_decision(serial_dict_to_client, go_increment):
            logger.debug('player_1' + ': I am attempting to buy property '+current_location['name'])
            memory['previous_action'] = "buy_property"
            return ("buy_property", params), memory
        else:
            # make_buy returned false, but is there still a chance?
            if agent_helper_functions.will_property_complete_set(player,current_location,current_gameboard):
                # if we can raise enough money, then the 'next' time around we'll succeed in buying
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player,current_location['price'],current_gameboard, True)
                if to_mortgage:
                    params['asset'] = to_mortgage['name']
                    logger.debug('player_1' + ': I am attempting to mortgage property '+ params['asset'])
                    memory['previous_action'] = "mortgage_property"
                    return ("mortgage_property", params), memory

                else: # last chance.
                    to_sell = agent_helper_functions.identify_potential_sale(player, current_gameboard, current_location['price'],True)
                    if to_sell:
                        params['asset'] = to_sell['name']
                        logger.debug(player['player_name']+ ': I am attempting to sell property '+ current_location['name']+' to the bank')
                        memory['previous_action'] = "sell_property"
                        return ("sell_property", params), memory

    if "concluded_actions" in allowable_moves:
        memory['previous_action'] = "concluded_actions"
        return ("concluded_actions", dict()), memory

    else:
        logger.error("Exception")
        raise Exception


def make_buy_property_decision(serial_dict_to_client, go_increment):
    """
    The agent decides to buy the property if:
    (i) it can 'afford' it. Our definition of afford is that we must have at least go_increment cash balance after
    the purchase.
    (ii) we can obtain a full color set through the purchase, and still have positive cash balance afterwards (though
    it may be less than go_increment).

    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :return: A Boolean. If True, then you decided to purchase asset from the bank, otherwise False. We allow you to
    purchase the asset even if you don't have enough cash; however, if you do you will end up with a negative
    cash balance and will have to handle that if you don't want to lose the game at the end of your move (see notes
    in handle_negative_cash_balance)
    """
    player_name = serial_dict_to_client['player']
    current_gameboard = serial_dict_to_client['current_gameboard']
    player = current_gameboard['players'][player_name]
    asset = current_gameboard['locations'][current_gameboard['location_sequence'][player['current_position']]]


    decision = False
    if player['current_cash'] - asset['price'] >= go_increment:  # case 1: can we afford it?
        logger.debug('player_1' + ': I will attempt to buy ' + asset['name'] + ' from the bank.')
        decision = True
    elif asset['price'] <= player['current_cash']and \
            agent_helper_functions.will_property_complete_set(player, asset, current_gameboard):
        logger.debug('player_1' + ': I will attempt to buy ' + asset['name'] + ' from the bank.')
        decision = True

    return decision


def make_bid(serial_dict_to_client):
    """
    Decide the amount you wish to bid for asset in auction, given the current_bid that is currently going. If you don't
    return a bid that is strictly higher than current_bid you will be removed from the auction and won't be able to
    bid anymore. Note that it is not necessary that you are actually on the location on the board representing asset, since
    you will be invited to the auction automatically once a player who lands on a bank-owned asset rejects buying that asset
    (this could be you or anyone else).
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param asset: An purchaseable instance of Location (i.e. real estate, utility or railroad)
    :param current_bid: The current bid that is going in the auction. If you don't bid higher than this amount, the bank
    will remove you from the auction proceedings. You could also always return 0 to voluntarily exit the auction.
    :return: An integer that indicates what you wish to bid for asset
    """
    # player, current_gameboard, asset, current_bid

    player_name = serial_dict_to_client['player']
    current_gameboard = serial_dict_to_client['current_gameboard']
    asset = current_gameboard['locations'][serial_dict_to_client['asset']]
    player = current_gameboard['players'][player_name]
    current_bid = serial_dict_to_client['current_bid']

    if current_bid < asset['price']:
        new_bid = current_bid + (asset['price'] - current_bid) / 2
        if new_bid < player['current_cash']:
            return new_bid
        else:  # We are aware that this can be simplified with a simple return 0 statement at the end. However in the final baseline agent
            # the return 0's would be replaced with more sophisticated rules. Think of them as placeholders.
            return 0  # this will lead to a rejection of the bid downstream automatically
    elif current_bid < player['current_cash'] and agent_helper_functions.will_property_complete_set(player, asset,
                                                                                                 current_gameboard):
        # We are prepared to bid more than the price of the asset only if it doesn't result in insolvency, and
        # if we can get a monopoly this way
        return current_bid + (player['current_cash'] - current_bid) / 4
    else:
        return 0  # no reason to bid


def handle_negative_cash_balance(serial_dict_to_client, memory, property_sell_percentage, mortgage_percentage):
    """
    You have a negative cash balance at the end of your move (i.e. your post-roll phase is over) and you must handle
    this issue before we move to the next player's pre-roll. If you do not succeed in restoring your cash balance to
    0 or positive, bankruptcy proceeds will begin and you will lost the game.

    The background agent tries a number of things to get itself out of a financial hole. First, it checks whether
    mortgaging alone can save it. If not, then it begins selling unimproved properties in ascending order of price, the idea being
    that it might as well get rid of cheap properties. This may not be the most optimal move but it is reasonable.
    If it ends up selling all unimproved properties and is still insolvent, it starts selling improvements, followed
    by a sale of the (now) unimproved properties.

    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :return: -1 if you do not try to address your negative cash balance, or 1 if you tried and believed you succeeded.
    Note that even if you do return 1, we will check to see whether you have non-negative cash balance. The rule of thumb
    is to return 1 as long as you 'try', or -1 if you don't try (in which case you will be declared bankrupt and lose the game)
    """
    player_name = serial_dict_to_client['player']
    current_gameboard = serial_dict_to_client['current_gameboard']
    player = current_gameboard['players'][player_name]

    if player['current_cash'] >= 0:   # prelim check to see if player has negative cash balance
        return (None, flag_config_dict['successful_action']), memory

    mortgage_potentials = list()
    max_sum = 0
    sorted_player_assets_list = _set_to_sorted_list_assets(player['assets'])
    for a in sorted_player_assets_list:
        a = current_gameboard['locations'][a]
        if a['is_mortgaged']:
            continue
        elif a['loc_class'] == 'real_estate' and (a['num_houses']>0 or a['num_hotels']>0):
            continue
        else:
            mortgage_potentials.append((a,a['mortgage']))
            max_sum += a['mortgage']
    if mortgage_potentials and max_sum+player['current_cash'] >= 0: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player['current_cash'] >= 0:
                return (None, flag_config_dict['successful_action']), memory # we're done
            params = dict()
            params['player'] = player['player_name']
            params['asset'] = p[0]['name']
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player['player_name']+ ': I am attempting to mortgage property '+ params['asset'])
            memory['previous_action'] = "mortgage_property"
            return ("mortgage_property", params), memory


    # if we got here, it means we're still in trouble. Next move is to sell unimproved properties. We don't check if
    # the total will cover our debts, since we're desperate at this point.

    # following sale potentials doesnot include properties from monopolized color groups
    sale_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player['assets'])
    for a in sorted_player_assets_list:
        a = current_gameboard['locations'][a]
        if a['color'] in player['full_color_sets_possessed']:
            continue
        elif a['is_mortgaged']:
            sale_potentials.append((a, (a['price'] * property_sell_percentage)-((1 + mortgage_percentage)*a['mortgage']))) #current_gameboard['bank'].property_sell_percentage
        elif a['loc_class'] =='real_estate' and (a['num_houses'] >0 or a['num_hotels'] >0):
            continue
        else:
            sale_potentials.append((a, a['price'] * property_sell_percentage)) #* current_gameboard['bank'].property_sell_percentage

    if sale_potentials: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player['current_cash'] >= 0:
                return (None, flag_config_dict['successful_action']), memory # we're done

            params = dict()
            params['player'] = player['player_name']
            params['asset'] = p[0]['name']
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player['player_name'] + ': I am attempting to sell property '+ p[0]['name'] + ' to the bank')
            memory['previous_action'] = "sell_property"
            return ("sell_property", params),memory


    # if selling properties from non monopolized color groups doesnot relieve the player from debt, then only we start thinking about giving up monopolized groups.
    # If we come across a unimproved property which belongs to a monopoly, we still have to loop through the other properties from the same color group and
    # sell the houses and hotels first because we cannot sell this property when the color group has improved properties
    # We first check if selling houses and hotels one by one on the other improved properties of the same color group relieves the player of his debt. If it does
    # then we return without selling the current property else we sell the property and the player loses monopoly of that color group.
    sale_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player['assets'])
    for a in sorted_player_assets_list:
        a = current_gameboard['locations'][a]
        if a['is_mortgaged']:
            sale_potentials.append((a, (a['price'] * property_sell_percentage)-((1+mortgage_percentage)*a['mortgage'])))
        elif a['loc_class'] =='real_estate' and (a['num_houses']>0 or a['num_hotels']>0):
            continue
        else:
            sale_potentials.append((a,a['price']*property_sell_percentage))

    if sale_potentials:
        sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by sell value in ascending order
        for p in sorted_potentials:
            if player['current_cash'] >= 0:
                return (None, flag_config_dict['successful_action']), memory # we're done

            sorted_player_assets_list = _set_to_sorted_list_assets(player['assets'])
            for prop in sorted_player_assets_list:
                prop = current_gameboard['locations'][prop]
                if prop!=p[0] and prop['color'] ==p[0]['color'] and p[0]['color'] in player['full_color_sets_possessed']:
                    if prop['num_hotels']>0:
                        if player['current_cash'] >= 0:
                            return (None, flag_config_dict['successful_action']), memory
                        params = dict()
                        params['player'] = player['player_name']
                        params['asset'] = prop['name']
                        params['current_gameboard'] = "current_gameboard"
                        params['sell_house'] = False
                        params['sell_hotel'] = True
                        logger.debug(player['player_name']+ ': I am attempting to sell hotel on '+ prop['name'] + ' to the bank')
                        memory['previous_action'] = "sell_house_hotel"
                        return ("sell_house_hotel", params), memory

                    elif prop['num_houses']>0:
                        if player['current_cash'] >= 0:
                            return (None, flag_config_dict['successful_action']), memory
                        params = dict()
                        params['player'] = player['player_name']
                        params['asset'] = prop['name']
                        params['current_gameboard'] = "current_gameboard"
                        params['sell_house'] = True
                        params['sell_hotel'] = False
                        logger.debug(player['player_name']+ ': I am attempting to sell house on '+ prop['name'] + ' to the bank')
                        memory['previous_action'] = "sell_house_hotel"
                        return ("sell_house_hotel", params), memory
                    else:
                        continue

            params = dict()
            params['player'] = player['player_name']
            params['asset'] = p[0]['name']
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player['player_name'] + ': I am attempting to sell property '+ p[0]['name'] + ' to the bank')
            memory['previous_action'] = "sell_property"
            return ("sell_property", params), memory



    #we reach here if the player still hasnot cleared his debt. The above loop has now resulted in some more non monopolized properties.
    #Hence we have to go through the process of looping through these properties once again to decide on the potential properties that can be mortgaged or sold.

    mortgage_potentials = list()
    max_sum = 0
    sorted_player_assets_list = _set_to_sorted_list_assets(player['assets'])
    for a in sorted_player_assets_list:
        a = current_gameboard['locations'][a]
        if a['is_mortgaged']:
            continue
        elif a['loc_class'] == 'real_estate' and (a['num_houses']>0 or a['num_hotels']>0):
            continue
        else:
            mortgage_potentials.append((a,a['mortgage']))
            max_sum += a.mortgage
    if mortgage_potentials and max_sum+player['current_cash'] >= 0: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player['current_cash'] >= 0:
                return (None, flag_config_dict['successful_action']), memory # we're done

            params = dict()
            params['player'] = player['player_name']
            params['asset'] = p[0]['name']
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player['player_name']+ ': I am attempting to mortgage property '+ params['asset'])
            memory['previous_action'] = "mortgage_property"
            return ("mortgage_property", params), memory

    # following sale potentials loops through the properties that have become unmonopolized due to the above loops and
    # doesnot include properties from monopolized color groups
    sale_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player['assets'])
    for a in sorted_player_assets_list:
        a = current_gameboard['locations'][a]
        if a['color'] in player['full_color_sets_possessed']:
            continue
        elif a['is_mortgaged']:
            sale_potentials.append((a, (a['price'] * property_sell_percentage)-((1 + mortgage_percentage)*a['mortgage'])))
        elif a['loc_class'] =='real_estate' and (a['num_houses']>0 or a['num_hotels']>0):
            continue
        else:
            sale_potentials.append((a, a['price'] * property_sell_percentage))

    if sale_potentials: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player['current_cash'] >= 0:
                return (None, flag_config_dict['successful_action']), memory # we're done

            params = dict()
            params['player'] = player['player_name']
            params['asset'] = p[0]['name']
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player['player_name'] + ': I am attempting to sell property '+ p[0]['name'] + ' to the bank')
            memory['previous_action'] = "sell_property"
            return ("sell_property", params), memory

    count = 0
    # if we're STILL not done, then the only option is to start selling houses and hotels from the remaining improved monopolized properties, if we have 'em
    while (player['num_total_houses'] > 0 or player['num_total_hotels'] > 0) and count <3: # often times, a sale may not succeed due to uniformity requirements. We keep trying till everything is sold,
        # or cash balance turns non-negative.
        count += 1 # there is a slim chance that it is impossible to sell an improvement unless the player does something first (e.g., replace 4 houses with a hotel).
        # The count ensures we terminate at some point, regardless.
        sorted_assets_list = _set_to_sorted_list_assets(player['assets'])

        for a in sorted_assets_list:
            a = current_gameboard['locations'][a]
            if a['loc_class'] == 'real_estate' and a['num_houses'] > 0:
                if player['current_cash'] >= 0:
                    return (None, flag_config_dict['successful_action']), memory # we're done

                params = dict()
                params['player'] = player['player_name']
                params['asset'] = a['name']
                params['current_gameboard'] = "current_gameboard"
                params['sell_house'] = True
                params['sell_hotel'] = False
                logger.debug(player['player_name']+ ': I am attempting to sell house on '+ a['name'] + ' to the bank')
                memory['previous_action'] = "sell_house_hotel"
                return ("sell_house_hotel", params), memory

            elif a['loc_class'] == 'real_estate' and a['num_hotels'] > 0:
                if player['current_cash'] >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done
                params = dict()
                params['player'] = player['player_name']
                params['asset'] = a['name']
                params['current_gameboard'] = "current_gameboard"
                params['sell_house'] = False
                params['sell_hotel'] = True
                logger.debug(player['player_name'] + ': I am attempting to sell house on '+ a['name'] + ' to the bank')
                memory['previous_action'] = "sell_house_hotel"
                return ("sell_house_hotel", params), memory

    # final straw
    final_sale_assets = player['assets'].copy()
    sorted_player_assets_list = _set_to_sorted_list_assets(final_sale_assets)
    for a in sorted_player_assets_list:
        a = current_gameboard['locations'][a]
        if player['current_cash'] >= 0:
            return (None, flag_config_dict['successful_action']), memory  # we're done
        params = dict()
        params['player'] = player['player_name']
        params['asset'] = a['name']
        params['current_gameboard'] = "current_gameboard"
        logger.debug(player['player_name'] + ': I am attempting to sell property '+ a['name'] + ' to the bank')
        memory['previous_action'] = "sell_property"
        return ("sell_property", params), memory

    return (None, flag_config_dict['successful_action']), memory # if we didn't suc


def _set_to_sorted_list_mortgaged_assets(player_mortgaged_assets):
    player_m_assets_list = list()
    player_m_assets_dict = dict()
    for item in player_mortgaged_assets:
        player_m_assets_dict[item] = item
    for sorted_key in sorted(player_m_assets_dict):
        player_m_assets_list.append(player_m_assets_dict[sorted_key])
    return player_m_assets_list


def _set_to_sorted_list_assets(player_assets):
    if type(player_assets) == list:
        return sorted(player_assets)
    player_assets_list = list()
    player_assets_dict = dict()
    for item in player_assets:
        player_assets_dict[item.name] = item
    for sorted_key in sorted(player_assets_dict):
        player_assets_list.append(player_assets_dict[sorted_key])
    return player_assets_list


def _build_decision_agent_methods_dict():
    """
    This function builds the decision agent methods dictionary.
    :return: The decision agent dict. Keys should be exactly as stated in this example, but the functions can be anything
    as long as you use/expect the exact function signatures we have indicated in this document.
    """
    ans = dict()
    ans['handle_negative_cash_balance'] = handle_negative_cash_balance
    ans['make_pre_roll_move'] = make_pre_roll_move
    ans['make_out_of_turn_move'] = make_out_of_turn_move
    ans['make_post_roll_move'] = make_post_roll_move
    ans['make_buy_property_decision'] = make_buy_property_decision
    ans['make_bid'] = make_bid
    ans['type'] = "decision_agent_methods"
    return ans


decision_agent_methods = _build_decision_agent_methods_dict() # this is the main data structure that is needed by gameplay

