import sys, os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation')
####################
from monopoly_simulator import background_agent_v3
from monopoly_simulator.agent import Agent
from monopoly_simulator import initialize_game_elements
from monopoly_simulator.action_choices import roll_die, concluded_actions
import numpy as np
from monopoly_simulator_background.simple_background_agent_becky_p1 import P1Agent
# from monopoly_simulator_background.simple_background_agent_becky_p2 import P2Agent
# from monopoly_simulator_background.background_agent_v2_agent import P2Agent_v2
# import simple_decision_agent_1
import json
from monopoly_simulator import diagnostics
from monopoly_simulator_background.interface import Interface
import sys, os
from monopoly_simulator.card_utility_actions import move_player_after_die_roll
from monopoly_simulator import location
from monopoly_simulator import novelty_generator
import xlsxwriter
import logging

from monopoly_simulator_background.log_setting import ini_log_level, set_log_level
logger = set_log_level()
from monopoly_simulator_background.agent_helper_functions import identify_improvement_opportunity_all

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

        if current_player.player_name == 'player_1':
            if num_active_players == 1:
                win_indicator = 1
            else:
                win_indicator = -1

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

#player_1 run the process
# def before_agent(game_elements, num_active_players, num_die_rolls, current_player_index, a):
def before_agent_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, a, die_roll):
    current_player = game_elements['players'][current_player_index]

    # while current_player.status == 'lost':
    #     current_player_index += 1
    #     current_player_index = current_player_index % len(game_elements['players'])
    #     current_player = game_elements['players'][current_player_index]

    #set current move to current player
    current_player.status = 'current_move'
    # pre-roll for current player + out-of-turn moves for everybody else,
    # till we get num_active_players skip turns in a row.

    logger.debug('Player_1 before move is in jail? '+ str(current_player.currently_in_jail))
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

    if die_roll:
        r = die_roll[-1]
    else:
        r = roll_die(game_elements['dies'], np.random.choice)

    # add to game history
    game_elements['history']['function'].append(roll_die)
    params = dict()
    params['die_objects'] = game_elements['dies']
    params['choice'] = np.random.choice
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(r)

    win_indicator = 0
    num_die_rolls += 1
    game_elements['current_die_total'] = sum(r)

    #####-die-#####
    logger.info('-die- have come up'+ str(r))

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

    else:
        logger.info(current_player.player_name+' is in jail now')
        current_player.currently_in_jail = False

    if current_player.current_cash < 0:
        game_elements, num_active_players, a, win_indicator = \
            cash_negative(game_elements, current_player, num_active_players, a, win_indicator)
        masked_actions = [0, 1]
    else:
        # post-roll for current player. No out-of-turn moves allowed at this point.
        #####becky######action space got#####################################
        a.board_to_state(game_elements)  # get state space
        allowable_actions = current_player.compute_allowable_post_roll_actions(game_elements)
        params_mask = identify_improvement_opportunity_all(current_player, game_elements)
        masked_actions = a.get_masked_actions(allowable_actions, params_mask, current_player)
        logger.debug('Set player_1 to jail ' + str(current_player.currently_in_jail))

    return game_elements, num_active_players, num_die_rolls, current_player_index, a, params, win_indicator, masked_actions

def after_agent_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params):
    # def after_agent(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params):
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

    # add to game history.
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
    if diagnostics.max_cash_balance(game_elements) > 10000: # this is our limit for runaway cash for testing purposes only.
                                                             # We print some diagnostics and return if any player exceeds this.
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)
        done_indicator = 1
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator

# player_2 run the process
# def simulate_game_step(game_elements, num_active_players, num_die_rolls, current_player_index):
def simulate_game_step_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, die_roll, done_indicator, win_indicator, a):
    current_player = game_elements['players'][current_player_index]
    ##################################################################################################################
    if current_player.status == 'lost':
        current_player_index += 1
        current_player_index = current_player_index % len(game_elements['players'])
        current_player = game_elements['players'][current_player_index]
        return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator


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

    if die_roll:
        r = die_roll[current_player_index - 1]
    else:
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
    logger.info('-die- have come up ' + str(r))

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

    else:
        logger.info(current_player.player_name+' is in jail now')
        current_player.currently_in_jail = False  # the player is only allowed to skip one turn (i.e. this one)


    # post-roll for current player. No out-of-turn moves allowed at this point.
    current_player.make_post_roll_moves(game_elements)
    logger.debug('player_2 is in jail? ' + str(current_player.currently_in_jail))
    # add to game history
    game_elements['history']['function'].append(current_player.make_post_roll_moves)
    params = dict()
    params['self'] = current_player
    params['current_gameboard'] = game_elements
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(None)

    win_indicator = 0
    a.set_board(game_elements)


    if current_player.current_cash < 0:
        game_elements, num_active_players, a, win_indicator = \
            cash_negative(game_elements, current_player, num_active_players, a, win_indicator)
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
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator

######################
def simulate_game_instance(game_elements, num_active_players, np_seed=6):
    """
    Simulate a game instance.
    :param game_elements: The dict output by set_up_board
    :param np_seed: The numpy seed to use to control randomness.
    :return: None
    """
    np.random.seed(np_seed)
    # np.random.shuffle(game_elements['players'])
    game_elements['seed'] = np_seed
    game_elements['card_seed'] = np_seed
    game_elements['choice_function'] = np.random.choice

    num_die_rolls = 0
    # game_elements['go_increment'] = 100 # we should not be modifying this here. It is only for testing purposes.
    # One reason to modify go_increment is if your decision agent is not aggressively trying to monopolize. Since go_increment
    # by default is 200 it can lead to runaway cash increases for simple agents like ours.

    logger.debug('players will play in the following order: '+ '->'.join([p.player_name for p in game_elements['players']]))
    logger.debug('Beginning play. Rolling first die...')
    current_player_index = 0
    winner = None

    a = Interface()
    a.set_board(game_elements)
    die_roll = []
    markder = []
    win_indicator = 0
    num = 0
    while win_indicator == 0:
        num += 1

        # game_elements, num_active_players, num_die_rolls, current_player_index, a = game_elements_, num_active_players_, num_die_rolls_, current_player_index_,a_

        game_elements, num_active_players, num_die_rolls, current_player_index, a, params, die_roll, win_indicator, masked_actions = \
            before_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, a, die_roll)
        logger.info(a.state_space)        # game_elements, num_active_players, num_die_rolls, current_player_index, a, params, win_indicator, masked_actions = \
        #     before_agent_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, a, die_roll)
        print('before=>',current_player_index)

        # for i, his in enumerate(game_elements['history']['param']):
        #     if 'card' in his:
        #         if 'amount' in game_elements['history']['param'][i-1]:
        #             print(his['card'].name, game_elements['history']['param'][i-1]['amount'])

        # print(game_elements['history']['param'])
        # if num == 10:
        #     break

        if markder == [1, 1]:
            break

        actions_vector = [1, 0]
        done_indicator = 0
        if win_indicator == 0:
            game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
                after_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params)
            # game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
            #     after_agent_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params)
            print('after',current_player_index)

        loop_num = 1
        die_roll = []
        if win_indicator == 0:
            while loop_num < 4:
                loop_num += 1
                game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, die_roll = \
                    simulate_game_step_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, die_roll, a)
                # game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator = \
                #     simulate_game_step_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, die_roll, done_indicator,
                #                                win_indicator, a)
                print('loop',current_player_index)

        a.save_history(game_elements)

        # if win_indicator != 0:
        #     for i in game_elements['history']['param']:
        #         if 'card' in i:
        #             print(i['card'].name)

    # let's print some numbers
    print(a.loc_history)
    a.get_logging_info(game_elements, 0, num_active_players)
    logger.debug('printing final asset owners: ')
    diagnostics.print_asset_owners(game_elements)
    logger.debug('number of dice rolls: ' + str(num_die_rolls))
    logger.debug('printing final cash balances: ')
    diagnostics.print_player_cash_balances(game_elements)

    if win_indicator == 1:
        logger.info('We have a winner: Player_1')
    elif win_indicator == -1:
        logger.info('We have a winner: Not Player_1')
    else:
        logger.info('No winner!')

    return


def set_up_board(game_schema_file_path, player_decision_agents, num_active_players):
    game_schema = json.load(open(game_schema_file_path, 'r'))
    game_schema['players']['player_states']['player_name'] = game_schema['players']['player_states']['player_name'][: num_active_players]
    return initialize_game_elements.initialize_board(game_schema, player_decision_agents)


# def before_agent_tf_step(game_elements, num_active_players, num_die_rolls, current_player_index, a, die_roll):
def before_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, a, die_roll):
    current_player = game_elements['players'][current_player_index]

    # while current_player.status == 'lost':
    #     current_player_index += 1
    #     current_player_index = current_player_index % len(game_elements['players'])
    #     current_player = game_elements['players'][current_player_index]

    #set current move to current player
    current_player.status = 'current_move'
    # pre-roll for current player + out-of-turn moves for everybody else,
    # till we get num_active_players skip turns in a row.

    logger.debug('Player_1 before move is in jail? '+ str(current_player.currently_in_jail))
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
    # print(game_elements['dies'][0].die_state)
    die_roll.append(r)
    # print('before', r,current_player_index)
    # add to game history
    game_elements['history']['function'].append(roll_die)
    params = dict()
    params['die_objects'] = game_elements['dies']
    params['choice'] = np.random.choice
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(r)

    win_indicator = 0
    num_die_rolls += 1
    game_elements['current_die_total'] = sum(r)

    #####-die-#####
    logger.info('-die- have come up'+ str(r))

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

    else:
        logger.info(current_player.player_name+' is in jail now')
        current_player.currently_in_jail = False

    if current_player.current_cash < 0:

        game_elements, num_active_players, a, win_indicator = \
            cash_negative(game_elements, current_player, num_active_players, a, win_indicator)
        masked_actions = [0, 1]
    else:
        # post-roll for current player. No out-of-turn moves allowed at this point.
        #####becky######action space got#####################################
        a.board_to_state(game_elements)
        allowable_actions = current_player.compute_allowable_post_roll_actions(game_elements)
        params_mask = identify_improvement_opportunity_all(current_player, game_elements)
        masked_actions = a.get_masked_actions(allowable_actions, params_mask, current_player)
        logger.debug('Set player_1 to jail ' + str(current_player.currently_in_jail))

    return game_elements, num_active_players, num_die_rolls, current_player_index, a, params, die_roll, win_indicator, masked_actions

def after_agent_hyp(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params):
    done_hyp = False
    a.board_to_state(game_elements)
    current_player = game_elements['players'][current_player_index]
    move_actions = a.vector_to_actions(game_elements, current_player,actions_vector)
    logger.debug('move_actions =====>'+ str(move_actions))
    # print('move_actions =====>' + str(move_actions))
    current_player.agent.set_move_actions(move_actions)
    code = current_player.make_post_roll_moves(game_elements) #-1 means doesnot work and 1 means successful
    #####################################################################

    # add to game history
    if code == 1:

        game_elements['history']['function'].append(current_player.make_post_roll_moves)
        params = dict()
        params['self'] = current_player
        params['current_gameboard'] = game_elements
        game_elements['history']['param'].append(params)
        game_elements['history']['return'].append(None)

        #if we conclude the actions
        if move_actions[0][0]== concluded_actions:
            done_hyp = True

    a.board_to_state(game_elements)  # get state space
    allowable_actions = current_player.compute_allowable_post_roll_actions(game_elements)
    params_mask = identify_improvement_opportunity_all(current_player, game_elements)
    masked_actions = a.get_masked_actions(allowable_actions, params_mask, current_player)

    win_indicator = 0
    if current_player.current_cash < 0:
        game_elements, num_active_players, a, win_indicator = \
            cash_negative(game_elements, current_player, num_active_players, a, win_indicator)
    else:
        current_player.status = 'waiting_for_move'

    current_player_index = (current_player_index+1)%len(game_elements['players'])

    done_indicator = 0
    if diagnostics.max_cash_balance(game_elements) > 10000: # this is our limit for runaway cash for testing purposes only.
                                                             # We print some diagnostics and return if any player exceeds this.
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)
        done_indicator = 1


    return game_elements, num_active_players, num_die_rolls, current_player_index, a, params, masked_actions, done_hyp,done_indicator

def after_agent_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, actions_vector, a, params):
    a.board_to_state(game_elements)
    # print('state_space', a.state_space)
    current_player = game_elements['players'][current_player_index]
    print(current_player.assets)
    # if not current_player.currently_in_jail:
    #got state and masked actions => agent => output actions and move
    #action vector => actions
    move_actions = a.vector_to_actions(game_elements, current_player,actions_vector)
    logger.debug('move_actions =====>'+ str(move_actions[0][0]))
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
    if diagnostics.max_cash_balance(game_elements) > 10000: # this is our limit for runaway cash for testing purposes only.
                                                             # We print some diagnostics and return if any player exceeds this.
        diagnostics.print_asset_owners(game_elements)
        diagnostics.print_player_cash_balances(game_elements)
        done_indicator = 1
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator

# player_2 run the process

def simulate_game_step_tf_nochange(game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, die_roll, a):
    current_player = game_elements['players'][current_player_index]
    ##################################################################################################################
    if current_player.status == 'lost':
        current_player_index += 1
        current_player_index = current_player_index % len(game_elements['players'])
        current_player = game_elements['players'][current_player_index]
        die_roll.append([])
        return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, die_roll


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
    die_roll.append(r)
    # add to game history
    game_elements['history']['function'].append(roll_die)
    params = dict()
    params['die_objects'] = game_elements['dies']
    params['choice'] = np.random.choice
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(r)

    num_die_rolls += 1
    game_elements['current_die_total'] = sum(r)
    logger.info('-die- have come up ' + str(r))

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

    else:
        logger.debug(current_player.player_name+' is in jail now')
        current_player.currently_in_jail = False  # the player is only allowed to skip one turn (i.e. this one)


    # post-roll for current player. No out-of-turn moves allowed at this point.
    current_player.make_post_roll_moves(game_elements)
    logger.debug('player_2 is in jail? ' + str(current_player.currently_in_jail))
    # add to game history
    game_elements['history']['function'].append(current_player.make_post_roll_moves)
    params = dict()
    params['self'] = current_player
    params['current_gameboard'] = game_elements
    game_elements['history']['param'].append(params)
    game_elements['history']['return'].append(None)

    win_indicator = 0
    a.set_board(game_elements)
    print('cash', current_player.player_name)
    if current_player.current_cash < 0:
        game_elements, num_active_players, a, win_indicator = \
            cash_negative(game_elements, current_player, num_active_players, a, win_indicator)
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
    return game_elements, num_active_players, num_die_rolls, current_player_index, done_indicator, win_indicator, die_roll




def inject_novelty(current_gameboard, novelty_schema=None):
    """
    Function for illustrating how we inject novelty
    ONLY FOR ILLUSTRATIVE PURPOSES
    :param current_gameboard: the current gameboard into which novelty will be injected. This gameboard will be modified
    :param novelty_schema: the novelty schema json, read in from file. It is more useful for running experiments at scale
    rather than in functions like these. For the most part, we advise writing your novelty generation routines, just like
    we do below, and for using the novelty schema for informational purposes (i.e. for making sense of the novelty_generator.py
    file and its functions.
    :return: None
    """

    ###Below are examples of Level 1, Level 2 and Level 3 Novelties
    ###Uncomment only the Level of novelty that needs to run (i.e, either Level1 or Level 2 or Level 3). Do not mix up novelties from different levels.


    #Level 1 Novelty
    # numberDieNovelty = novelty_generator.NumberClassNovelty()
    # numberDieNovelty.die_novelty(current_gameboard, 2, die_state_vector=[[1,2,3,4,5,6],[1,2,3,4,5,6]])
    # classDieNovelty = novelty_generator.TypeClassNovelty()
    # # die_state_distribution_vector = ['uniform','uniform','biased','biased']
    # die_state_distribution_vector = ['uniform', 'uniform']
    # die_type_vector = ['odd_only','even_only']
    # classDieNovelty.die_novelty(current_gameboard, die_state_distribution_vector, die_type_vector)


    # classCardNovelty = novelty_generator.TypeClassNovelty()
    # novel_cc = dict()
    # novel_cc["street_repairs"] = "alternate_contingency_function_1"
    # novel_chance = dict()
    # novel_chance["general_repairs"] = "alternate_contingency_function_1"
    # classCardNovelty.card_novelty(current_gameboard, novel_cc, novel_chance)



    #Level 2 Novelty
    #The below combination reassigns property groups and individual properties to different colors.
    #On playing the game it is verified that the newly added property to the color group is taken into account for monopolizing a color group,
    # i,e the orchid color group now has Baltic Avenue besides St. Charles Place, States Avenue and Virginia Avenue. The player acquires a monopoly
    # only on the ownership of all the 4 properties in this case.
    inanimateNovelty = novelty_generator.InanimateAttributeNovelty()
    # inanimateNovelty.map_property_set_to_color(current_gameboard, [current_gameboard['location_objects']['Park Place'], current_gameboard['location_objects']['Boardwalk']], 'Brown')
    # inanimateNovelty.map_property_to_color(current_gameboard, current_gameboard['location_objects']['Baltic-Avenue'], 'Orchid')
    # #setting new rents for Indiana Avenue
    # inanimateNovelty.rent_novelty(current_gameboard['location_objects']['Indiana Avenue'], {'rent': 50, 'rent_1_house': 150})
    # asset_lists = ["Mediterranean Avenue", "Baltic Avenue", "Reading Railroad", "Oriental Avenue", "Vermont Avenue", "Connecticut Avenue", "St. Charles Place", "Electric Company", "States Avenue", "Virginia Avenue", "Pennsylvania Railroad", "St. James Place", "Tennessee Avenue", "New York Avenue", "Kentucky Avenue", "Indiana Avenue", "Illinois Avenue", "B&O Railroad", "Atlantic Avenue", "Ventnor Avenue", "Water Works", "Marvin Gardens", "Pacific Avenue", "North Carolina Avenue", "Pennsylvania Avenue", "Short Line", "Park Place", "Boardwalk"]
    # for asset in asset_lists:
    #     inanimateNovelty.price_novelty(current_gameboard['location_objects'][asset], 1499)
    inanimateNovelty.price_novelty(current_gameboard['location_objects']['Baltic Avenue'], 1499)


    # Level 3 Novelty
    # Change game board

    # granularityNovelty = novelty_generator.GranularityRepresentationNovelty()
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Baltic-Avenue'], 6)
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['States-Avenue'], 20)
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Tennessee-Avenue'], 27)

    # spatialNovelty = novelty_generator.SpatialRepresentationNovelty()
    # spatialNovelty.color_reordering(current_gameboard, ['Boardwalk', 'Park Place'], 'Blue')
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Park Place'], 52)
    return current_gameboard
if __name__ == '__main__':
    # this is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    # control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    # but still relatively simple agent soon.
    # file_path = '/media/becky/GNOME-p3/monopoly_simulator/gameplay.log'
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    ini_log_level()
    set_log_level()
    # logger.debug('tf_file')
    player_decision_agents = dict()
    num_active_players = 4

    player_decision_agents['player_1'] = P1Agent()

    name_num = 1
    while name_num < num_active_players:
        name_num += 1
        player_decision_agents['player_' + str(name_num)] = Agent(**background_agent_v3.decision_agent_methods)


    game_elements = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-1.json',
                                 player_decision_agents, num_active_players)

    # inject_novelty(game_elements)
    simulate_game_instance(game_elements, num_active_players, np_seed=5)
