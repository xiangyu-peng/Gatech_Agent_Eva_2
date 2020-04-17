import initialize_game_elements
from action_choices import roll_die
import numpy as np
from simple_background_agent_becky_p1 import P1Agent
# from simple_background_agent_becky_p2 import P2Agent
# import simple_decision_agent_1
import json
import diagnostics
from interface import Interface
import sys, os
from card_utility_actions import move_player_after_die_roll

import xlsxwriter
import logging
from log_setting import set_log_level, ini_log_level
logger = ini_log_level()
logger = set_log_level()

def cash_negative(game_elements, current_player, num_active_players, a, win_indicator):
    code = current_player.agent.handle_negative_cash_balance(current_player, game_elements)

    # add to game history
    game_elements['history']['function'].append(current_player.agent.handle_negative_cash_balance)
    params = dict()
    params['player'] = current_player
    params['current_gameboard'] = game_elements
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(code)
    #####becky#####
    #
    if code == -1 or current_player.current_cash < 0:
        current_player.begin_bankruptcy_proceedings(game_elements)
        # add to game history
        game_elements['history']['function'].append(current_player.begin_bankruptcy_proceedings)
        params = dict()
        params['self'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)

        num_active_players -= 1
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)

        if num_active_players == 1:
            for p in game_elements['players']:
                if p.status != 'lost':
                    winner = p
                    p.status = 'won'
            for p in game_elements['players']:
                if p.player_name == 'player_1':
                    if p.status == 'won':
                        win_indicator = 1
                    elif p.status == 'lost':
                        win_indicator = -1
                    else:
                        win_indicator = 0
    a.board_to_state(params['current_gameboard'])  # get state s
    return game_elements, num_active_players, a, win_indicator


def after_agent_hypothetical(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params):
    a.board_to_state(game_elements)
    # print('state_space', a.state_space)
    current_player = game_elements['players'][current_player_index]
    # if not current_player.currently_in_jail:
    #got state and masked actions => agent => output actions and move
    #action vector => actions
    move_actions = a.vector_to_actions(game_elements, current_player,actions_vector)
    logger.debug('move_actions =====>'+ str(move_actions))
    current_player.agent.set_move_actions(move_actions)
    current_player.make_post_roll_moves(game_elements)
    #####################################################################

    # add to game history
    game_elements['history']['function'].append(current_player.make_post_roll_moves)
    params = dict()
    params['self'] = current_player
    params['current_gameboard'] = game_elements
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(None)

    logger.debug('now Player after actions is in jail? ' + str(current_player.currently_in_jail))

    win_indicator = 0
    if current_player.current_cash < 0:
        game_elements, num_active_players, a, win_indicator = \
            cash_negative(game_elements, current_player, num_active_players, a, win_indicator)
    else:
        current_player.status = 'waiting_for_move'

    current_player_index = (current_player_index+1)%len(game_elements['players'])

    done_indicator = 0
    if diagnostics.max_cash_balance(game_elements) > 300000: # this is our limit for runaway cash for testing purposes only.
                                                             # We print some diagnostics and return if any player exceeds this.
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)
        done_indicator = 1
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator

