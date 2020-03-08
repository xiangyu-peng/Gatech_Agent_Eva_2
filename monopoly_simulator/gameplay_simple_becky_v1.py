import initialize_game_elements
from action_choices import roll_die
import numpy as np
from card_utility_actions import move_player_after_die_roll
import simple_background_agent_becky_v1
# import simple_decision_agent_1
import json
import diagnostics
from interface import Interface
import sys, os
import copy

import os, sys

class HiddenPrints:
    def __enter__(self):

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#player_1 run the process
def before_agent(game_elements, num_active_players, num_die_rolls, current_player_index, a):
    current_player = game_elements['players'][current_player_index]

    # while current_player.status == 'lost':
    #     current_player_index += 1
    #     current_player_index = current_player_index % len(game_elements['players'])
    #     current_player = game_elements['players'][current_player_index]

    #set current move to current player
    current_player.status = 'current_move'
    # pre-roll for current player + out-of-turn moves for everybody else,
    # till we get num_active_players skip turns in a row.

    skip_turn = 0
    #make make_pre_roll_moves for current player -> player has allowable actions and then call agent.pre-roll-move
    if current_player.make_pre_roll_moves(game_elements) == 2: # 2 is the special skip-turn code #in player.py
        skip_turn += 1

    out_of_turn_player_index = current_player_index + 1
    out_of_turn_count = 0
    while skip_turn != num_active_players and out_of_turn_count<=200:
        out_of_turn_count += 1
        out_of_turn_player = game_elements['players'][out_of_turn_player_index%len(game_elements['players'])]
        if out_of_turn_player.status == 'lost':
            out_of_turn_player_index += 1
            continue
        oot_code = out_of_turn_player.make_out_of_turn_moves(game_elements)
        # add to game history
        game_elements['history']['function'].append(out_of_turn_player.make_out_of_turn_moves)
        params = dict()
        params['self']=out_of_turn_player
        params['current_gameboard']=game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(oot_code)

        if  oot_code == 2:
            skip_turn += 1
        else:
            skip_turn = 0
        out_of_turn_player_index += 1

    # now we roll the dice and get into the post_roll phase,
    # but only if we're not in jail.


    r = roll_die(game_elements['dies'], np.random.choice)
    # add to game history
    game_elements['history']['function'].append(roll_die)
    params = dict()
    params['die_objects'] = game_elements['dies']
    params['choice'] = np.random.choice
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(r)

    num_die_rolls += 1
    game_elements['current_die_total'] = sum(r)
    #####-die-#####
    print('-die- have come up',str(r))
    if not current_player.currently_in_jail:
        check_for_go = True
        move_player_after_die_roll(current_player, sum(r), game_elements, check_for_go)
        # add to game history
        game_elements['history']['function'].append(move_player_after_die_roll)
        params = dict()
        params['player'] = current_player
        params['rel_move'] = sum(r)
        params['current_gameboard'] = game_elements
        params['check_for_go'] = check_for_go
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)

        current_player.process_move_consequences(game_elements)
        # add to game history
        game_elements['history']['function'].append(current_player.process_move_consequences)
        params = dict()
        params['self'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)
        win_indicator = 0
        if current_player.current_cash < 0:
            code = current_player.handle_negative_cash_balance(current_player, game_elements)
            # add to game history
            game_elements['history']['function'].append(current_player.handle_negative_cash_balance)
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
                        if p.player_name == 'player_1':
                            if p.status == 'won':
                                win_indicator = 1
                            elif p.status == 'lost':
                                win_indicator = -1
                            else:
                                win_indicator = 0
            a.board_to_state(params['current_gameboard'])  # get state space

        else:

            # post-roll for current player. No out-of-turn moves allowed at this point.
            #####becky######action space got#####################################
            a = Interface()
            a.board_to_state(params['current_gameboard']) #get state space
            print('state_space =====>', a.state_space)
            allowable_actions,param = current_player.compute_allowable_post_roll_actions(params['current_gameboard'])
            # print('allowed_actions=====>', allowable_actions)
            a.get_masked_actions(allowable_actions, param, current_player)
            # print('masked_actions =====>', a.masked_actions)
            # print('current_player\'s mortgage assets', current_player.mortgaged_assets)

    return game_elements, num_active_players, num_die_rolls, current_player_index, a, params, win_indicator

def after_agent(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params):
    a.board_to_state(game_elements)
    # print('state_space', a.state_space)
    current_player = game_elements['players'][current_player_index]
    if not current_player.currently_in_jail:
        #got state and masked actions => agent => output actions and move
        #action vector => actions
        move_actions = a.vector_to_actions(game_elements, current_player,actions_vector)
        print('move_actions =====>', move_actions)
        current_player.make_post_roll_moves(game_elements, move_actions)
        #####################################################################

        # add to game history
        game_elements['history']['function'].append(current_player.make_post_roll_moves)
        params = dict()
        params['self'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)


    else:
        current_player.currently_in_jail = False # the player is only allowed to skip one turn (i.e. this one)
    #add a indecator to tell the env the game is wining or losing: 0 means losing and 1 means winning
    win_indicator = 0
    if current_player.current_cash < 0:
        code = current_player.handle_negative_cash_balance(current_player, game_elements)
        # add to game history
        game_elements['history']['function'].append(current_player.handle_negative_cash_balance)
        params = dict()
        params['player'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(code)
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
                    if p.player_name == 'player_1':
                        if p.status == 'won':
                            win_indicator = 1
                        elif p.status == 'lost':
                            win_indicator = -1
                        else:
                            win_indicator = 0

    else:
        current_player.status = 'waiting_for_move'

    current_player_index = (current_player_index+1)%len(game_elements['players'])
    #done predictor will decide if we need to terminate game before game ends
    # 0 means keep simulating, 1 means stops
    done_indicator = 0
    if diagnostics.max_cash_balance(game_elements) > 10000: # this is our limit for runaway cash for testing purposes only.
                                                             # We print some diagnostics and return if any player exceeds this.
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)
        done_indicator = 1
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator

# player_2 run the process
def simulate_game_step(game_elements, num_active_players, num_die_rolls, current_player_index):

    current_player = game_elements['players'][current_player_index]
    ##################################################################################################################
    while current_player.status == 'lost':
        current_player_index += 1
        current_player_index = current_player_index % len(game_elements['players'])
        current_player = game_elements['players'][current_player_index]

    # set current move to current player
    current_player.status = 'current_move'
    # pre-roll for current player + out-of-turn moves for everybody else,
    # till we get num_active_players skip turns in a row.
    skip_turn = 0
    # make make_pre_roll_moves for current player -> player has allowable actions and then call agent.pre-roll-move
    if current_player.make_pre_roll_moves(game_elements) == 2:  # 2 is the special skip-turn code #in player.py
        skip_turn += 1

    out_of_turn_player_index = current_player_index + 1
    out_of_turn_count = 0
    while skip_turn != num_active_players and out_of_turn_count <= 200:
        out_of_turn_count += 1
        # print 'checkpoint 1'
        out_of_turn_player = game_elements['players'][out_of_turn_player_index % len(game_elements['players'])]
        if out_of_turn_player.status == 'lost':
            out_of_turn_player_index += 1
            continue
        oot_code = out_of_turn_player.make_out_of_turn_moves(game_elements)
        # add to game history
        game_elements['history']['function'].append(out_of_turn_player.make_out_of_turn_moves)
        params = dict()
        params['self'] = out_of_turn_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(oot_code)

        if oot_code == 2:
            skip_turn += 1
        else:
            skip_turn = 0
        out_of_turn_player_index += 1
##################################################################################################################
    # now we roll the dice and get into the post_roll phase,
    # but only if we're not in jail.

    r = roll_die(game_elements['dies'], np.random.choice)
    # add to game history
    game_elements['history']['function'].append(roll_die)
    params = dict()
    params['die_objects'] = game_elements['dies']
    params['choice'] = np.random.choice
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(r)

    num_die_rolls += 1
    game_elements['current_die_total'] = sum(r)
    print('-die- have come up ', str(r))
    if not current_player.currently_in_jail:
        check_for_go = True
        move_player_after_die_roll(current_player, sum(r), game_elements, check_for_go)
        # add to game history
        game_elements['history']['function'].append(move_player_after_die_roll)
        params = dict()
        params['player'] = current_player
        params['rel_move'] = sum(r)
        params['current_gameboard'] = game_elements
        params['check_for_go'] = check_for_go
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)

        current_player.process_move_consequences(game_elements)
        # add to game history
        game_elements['history']['function'].append(current_player.process_move_consequences)
        params = dict()
        params['self'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)

        # post-roll for current player. No out-of-turn moves allowed at this point.
        current_player.make_post_roll_moves(game_elements, [])

        # add to game history
        game_elements['history']['function'].append(current_player.make_post_roll_moves)
        params = dict()
        params['self'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)

    else:
        current_player.currently_in_jail = False  # the player is only allowed to skip one turn (i.e. this one)
    win_indicator = 0
    if current_player.current_cash < 0:
        code = current_player.handle_negative_cash_balance(current_player, game_elements)
        # add to game history
        game_elements['history']['function'].append(current_player.handle_negative_cash_balance)
        params = dict()
        params['player'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(code)
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
                    if p.player_name == 'player_1':
                        if p.status == 'won':
                            win_indicator = 1
                        elif p.status == 'lost':
                            win_indicator = -1
                        else:
                            win_indicator = 0
    else:
        current_player.status = 'waiting_for_move'

    current_player_index = (current_player_index + 1) % len(game_elements['players'])
    done_indicator = 0
    if diagnostics.max_cash_balance(
            game_elements) > 10000:  # this is our limit for runaway cash for testing purposes only.
        # We print some diagnostics and return if any player exceeds this.
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)
        done_indicator = 1
        # return
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator

######################
def simulate_game_instance(game_elements, num_active_players, np_seed=0):
    """
    Simulate a game instance.
    :param game_elements: The dict output by set_up_board
    :param np_seed: The numpy seed to use to control randomness.
    :return: None
    """
    np.random.seed(np_seed)
    np.random.shuffle(game_elements['players'])
    game_elements['seed'] = np_seed
    game_elements['card_seed'] = np_seed
    game_elements['choice_function'] = np.random.choice

    num_die_rolls = 0
    # game_elements['go_increment'] = 100 # we should not be modifying this here. It is only for testing purposes.
    # One reason to modify go_increment is if your decision agent is not aggressively trying to monopolize. Since go_increment
    # by default is 200 it can lead to runaway cash increases for simple agents like ours.

    print('players will play in the following order: ', '->'.join([p.player_name for p in game_elements['players']]))
    print('Beginning play. Rolling first die...')
    current_player_index = 0
    winner = None
    a = Interface()
    # stop_num = 0
    # game_elements_, num_active_players_, num_die_rolls_, current_player_index_, a_ = copy.deepcopy(game_elements), num_active_players, num_die_rolls, current_player_index,a
    while num_active_players > 1:
        # game_elements, num_active_players, num_die_rolls, current_player_index, a = game_elements_, num_active_players_, num_die_rolls_, current_player_index_,a_
        game_elements, num_active_players, num_die_rolls, current_player_index, a, params, win_indicator = \
            before_agent(game_elements, num_active_players, num_die_rolls, current_player_index,a)
        actions_vector = [0 for i in range(79)] + [1]
        game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
            after_agent(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params)
        game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
            simulate_game_step(game_elements, num_active_players, num_die_rolls, current_player_index)
        # stop_num += 1
        # if stop_num > 2:
        #     break
    #     current_player = game_elements['players'][current_player_index]
    #
    #     while current_player.status == 'lost':
    #         current_player_index += 1
    #         current_player_index = current_player_index % len(game_elements['players'])
    #         current_player = game_elements['players'][current_player_index]
    #
    #     #set current move to current player
    #     current_player.status = 'current_move'
    #     # pre-roll for current player + out-of-turn moves for everybody else,
    #     # till we get num_active_players skip turns in a row.
    #
    #     skip_turn = 0
    #     #make make_pre_roll_moves for current player -> player has allowable actions and then call agent.pre-roll-move
    #     if current_player.make_pre_roll_moves(game_elements) == 2: # 2 is the special skip-turn code #in player.py
    #         skip_turn += 1
    #
    #     out_of_turn_player_index = current_player_index + 1
    #     out_of_turn_count = 0
    #     while skip_turn != num_active_players and out_of_turn_count<=200:
    #         out_of_turn_count += 1
    #         # print 'checkpoint 1'
    #         out_of_turn_player = game_elements['players'][out_of_turn_player_index%len(game_elements['players'])]
    #         if out_of_turn_player.status == 'lost':
    #             out_of_turn_player_index += 1
    #             continue
    #         oot_code = out_of_turn_player.make_out_of_turn_moves(game_elements)
    #         # add to game history
    #         game_elements['history']['function'].append(out_of_turn_player.make_out_of_turn_moves)
    #         params = dict()
    #         params['self']=out_of_turn_player
    #         params['current_gameboard']=game_elements
    #         game_elements['history']['param'].append(params)
    #         game_elements['history']['return'].append(oot_code)
    #
    #         if  oot_code == 2:
    #             skip_turn += 1
    #         else:
    #             skip_turn = 0
    #         out_of_turn_player_index += 1
    #
    #     # now we roll the dice and get into the post_roll phase,
    #     # but only if we're not in jail.
    #
    #
    #     r = roll_die(game_elements['dies'], np.random.choice)
    #     # add to game history
    #     game_elements['history']['function'].append(roll_die)
    #     params = dict()
    #     params['die_objects'] = game_elements['dies']
    #     params['choice'] = np.random.choice
    #     game_elements['history']['param'].append(params)
    #     game_elements['history']['return'].append(r)
    #
    #     num_die_rolls += 1
    #     game_elements['current_die_total'] = sum(r)
    #     print('dies have come up ',str(r))
    #     if not current_player.currently_in_jail:
    #         check_for_go = True
    #         move_player_after_die_roll(current_player, sum(r), game_elements, check_for_go)
    #         # add to game history
    #         game_elements['history']['function'].append(move_player_after_die_roll)
    #         params = dict()
    #         params['player'] = current_player
    #         params['rel_move'] = sum(r)
    #         params['current_gameboard'] = game_elements
    #         params['check_for_go'] = check_for_go
    #         game_elements['history']['param'].append(params)
    #         game_elements['history']['return'].append(None)
    #
    #         current_player.process_move_consequences(game_elements)
    #         # add to game history
    #         game_elements['history']['function'].append(current_player.process_move_consequences)
    #         params = dict()
    #         params['self'] = current_player
    #         params['current_gameboard'] = game_elements
    #         game_elements['history']['param'].append(params)
    #         game_elements['history']['return'].append(None)
    #
    #         # post-roll for current player. No out-of-turn moves allowed at this point.
    #         #####becky######action space got#####################################
    #         a = Interface()
    #         a.board_to_state(params['current_gameboard']) #get state space
    #         # print 'state_space =====>', a.state_space
    #         allowable_actions,param = current_player.compute_allowable_post_roll_actions(params['current_gameboard'])
    #         # print 'allowed_actions=====>', allowable_actions
    #         a.get_masked_actions(allowable_actions, param, current_player)
    #         # print 'masked_actions =====>', a.masked_actions
    #
    #         #got state and masked actions => agent => output actions and move
    #
    #         #action vector => actions
    #         if current_player.player_name == 'player_1':
    #             move_actions = a.vector_to_actions(params['current_gameboard'], current_player)
    #         else:
    #             move_actions = []
    #         # print 'move_actions =====>', move_actions
    #         current_player.make_post_roll_moves(game_elements, move_actions)
    #         #####################################################################
    #
    #         # add to game history
    #         game_elements['history']['function'].append(current_player.make_post_roll_moves)
    #         params = dict()
    #         params['self'] = current_player
    #         params['current_gameboard'] = game_elements
    #         game_elements['history']['param'].append(params)
    #         game_elements['history']['return'].append(None)
    #
    #     else:
    #         current_player.currently_in_jail = False # the player is only allowed to skip one turn (i.e. this one)
    #
    #     if current_player.current_cash < 0:
    #         code = current_player.handle_negative_cash_balance(current_player, game_elements)
    #         # add to game history
    #         game_elements['history']['function'].append(current_player.handle_negative_cash_balance)
    #         params = dict()
    #         params['player'] = current_player
    #         params['current_gameboard'] = game_elements
    #         game_elements['history']['param'].append(params)
    #         game_elements['history']['return'].append(code)
    #         if code == -1 or current_player.current_cash < 0:
    #             current_player.begin_bankruptcy_proceedings(game_elements)
    #             # add to game history
    #             game_elements['history']['function'].append(current_player.begin_bankruptcy_proceedings)
    #             params = dict()
    #             params['self'] = current_player
    #             params['current_gameboard'] = game_elements
    #             game_elements['history']['param'].append(params)
    #             game_elements['history']['return'].append(None)
    #
    #             num_active_players -= 1
    #             diagnostics.print_asset_owners(game_elements)
    #             diagnostics.print_player_cash_balances(game_elements)
    #
    #             if num_active_players == 1:
    #                 for p in game_elements['players']:
    #                     if p.status != 'lost':
    #                         winner = p
    #                         p.status = 'won'
    #     else:
    #         current_player.status = 'waiting_for_move'
    #
    #     current_player_index = (current_player_index+1)%len(game_elements['players'])
    #
    #     if diagnostics.max_cash_balance(game_elements) > 10000: # this is our limit for runaway cash for testing purposes only.
    #                                                              # We print some diagnostics and return if any player exceeds this.
    #         diagnostics.print_asset_owners(game_elements)
    #         diagnostics.print_player_cash_balances(game_elements)
    #         return
    #
    # # let's print some numbers
    # print('printing final asset owners: ')
    # diagnostics.print_asset_owners(game_elements)
    # print('number of dice rolls: ', str(num_die_rolls))
    # print('printing final cash balances: ')
    # diagnostics.print_player_cash_balances(game_elements)

    if winner:
        print('We have a winner: ', winner.player_name)

        return winner.player_name
    #for debugging
    return 'player_3'


def set_up_board(game_schema_file_path, player_decision_agents, num_active_players):
    game_schema = json.load(open(game_schema_file_path, 'r'))
    return initialize_game_elements.initialize_board(game_schema, player_decision_agents, num_active_players)


if __name__ == '__main__':
    # this is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    # control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    # but still relatively simple agent soon.
    n = 0
    win_sum = 0
    while n < 20:
        with HiddenPrints():
            player_decision_agents = dict()
            num_active_players = 2
            player_list = ['player_'+ str(i + 1) for i in range(num_active_players)]
            for player_name in player_list:
                player_decision_agents[player_name] = simple_background_agent_becky_v1.decision_agent_methods
            game_elements = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json',
                                         player_decision_agents, num_active_players)
            winner = simulate_game_instance(game_elements, num_active_players, np_seed=n)
            if winner == 'player_2':
                win_sum += 1
            n+=1
    print(win_sum)
    #just testing history.
    # print len(game_elements['history']['function'])
    # print len(game_elements['history']['param'])
    # print len(game_elements['history']['return'])