import numpy as np
from location import RailroadLocation
from flag_config import flag_config_dict
import diagnostics
import logging
logger = logging.getLogger('monopoly_simulator.logging_info.card_utility_actions')

"""
This is an important file that contains many functions (not including internal functions that start with _) that
either correspond to an action or contingency contained in a card (e.g., go to jail)  or an action that must be taken
when we land on an 'action' location (such as community chest, wherein we must pick a card from the community chest
card pack)
"""


def _set_send_to_jail(player, current_gameboard):
    """
    Internal function that sets the currently_in_jail field of player to true.
    :param player: player who is currently in jail and needs a status update
    :param current_gameboard: The global game board data structure
    :return: None
    """
    player.currently_in_jail = True


def go_to_jail(player, current_gameboard):
    """
    The player will be moved to jail. The player will not receive go_increment, even if they pass go.
    :param player: Player instance.
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('execute go_to_jail action for '+player.player_name)
    player.send_to_jail(current_gameboard)
    # add to game history
    current_gameboard['history']['function'].append(player.send_to_jail)
    params = dict()
    params['self'] = player
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


def _set_to_sorted_list_func(set_cards):
    cc_card_dict = dict()
    while len(set_cards) != 0:
        popped_card = set_cards.pop()
        try:
            cc_card_dict[str(popped_card.name)].append(popped_card)
        except:
            cc_card_dict[str(popped_card.name)] = [popped_card]
    list_cards = []

    for sorted_key in sorted(cc_card_dict):   #sorts dictionary based on dict keys
        for j in range(len(cc_card_dict[sorted_key])):
            list_cards.append(cc_card_dict[sorted_key][j])
    return list_cards


def pick_card_from_community_chest(player, current_gameboard):
    """
    Pick the card  from the community chest pack and execute the action
    Note: get_out_of_jail_free card is treated a little differently, since we must remove it from the card pack.
    :param player: an instance of Player. This is the player that will be picking the card.
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug(player.player_name+' is picking card from community chest.')
    card_rand = np.random.RandomState(current_gameboard['card_seed'])
    set_cc_cards_copy = current_gameboard['community_chest_cards'].copy()
    list_community_chest_cards = _set_to_sorted_list_func(set_cc_cards_copy)
    card = card_rand.choice(list_community_chest_cards)
    current_gameboard['picked_community_chest_cards'].append(current_gameboard['community_chest_card_objects'][card.name])
    # card = card_rand.choice(list(current_gameboard['community_chest_cards']))
    current_gameboard['card_seed'] += 1
    logger.debug(player.player_name+' picked card '+card.name)
    if card.name == 'get_out_of_jail_free':
        logger.debug('removing get_out_of_jail card from community chest pack')
        current_gameboard['community_chest_cards'].remove(card)
        card.action(player, card, current_gameboard, pack='community_chest')
        current_gameboard['history']['function'].append(card.action)
        params = dict()
        params['player'] = player
        params['card'] = card
        params['current_gameboard'] = current_gameboard
        params['pack'] = 'community_chest'
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)
    else:
        card.action(player, card, current_gameboard) # all card actions except get out of jail free must take this signature
        # add to game history
        current_gameboard['history']['function'].append(card.action)
        params = dict()
        params['player'] = player
        params['card'] = card
        params['current_gameboard'] = current_gameboard
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)


def pick_card_from_chance(player, current_gameboard):
    """
    Pick the card  from the chance pack and execute the action
    Note: get_out_of_jail_free card is treated a little differently, since we must remove it from the card pack.
    :param player: an instance of Player. This is the player that will be picking the card.
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug(player.player_name+ ' is picking card from chance.')
    card_rand = np.random.RandomState(current_gameboard['card_seed'])
    set_chance_cards_copy = current_gameboard['chance_cards'].copy()
    list_chance_cards = _set_to_sorted_list_func(set_chance_cards_copy)
    card = card_rand.choice(list_chance_cards)
    current_gameboard['picked_chance_cards'].append(current_gameboard['chance_card_objects'][card.name])
    # card = card_rand.choice(list(current_gameboard['chance_cards']))
    current_gameboard['card_seed'] += 1
    logger.debug(player.player_name+ ' picked card '+ card.name)
    if card.name == 'get_out_of_jail_free':
        logger.debug('removing get_out_of_jail card from chance pack')
        current_gameboard['chance_cards'].remove(card)
        card.action(player, card, current_gameboard, pack='chance')
        current_gameboard['history']['function'].append(card.action)
        params = dict()
        params['player'] = player
        params['card'] = card
        params['current_gameboard'] = current_gameboard
        params['pack'] = 'chance'
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)
    else:
        card.action(player, card, current_gameboard) # all card actions except get out of jail free must take this signature
        # add to game history
        current_gameboard['history']['function'].append(card.action)
        params = dict()
        params['player'] = player
        params['card'] = card
        params['current_gameboard'] = current_gameboard
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)


def move_player(player, card, current_gameboard):
    """
    moves the player to the destination specified on the community chest or chance card
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing move_player for '+player.player_name)
    logger.debug('destination specified on card is '+card.destination.name)
    new_position = card.destination.start_position
    jail_position = current_gameboard['jail_position']
    if new_position == jail_position:
        player.send_to_jail(current_gameboard)
        # add to game history
        current_gameboard['history']['function'].append(player.send_to_jail)
        params = dict()
        params['self'] = player
        params['current_gameboard'] = current_gameboard
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)
    else:
        _move_player__check_for_go(player, new_position, current_gameboard)


def set_get_out_of_jail_card_status(player, card, current_gameboard, pack):
    """
    Depending on whether we took the card out of community chest or chance, we update the requisite field for the player.
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing set_get_out_of_jail_card_status for '+player.player_name)
    if pack == 'community_chest' and card.name == 'get_out_of_jail_free': # remember, this is an object equality test
        player.has_get_out_of_jail_community_chest_card = True
        logger.debug(player.player_name+' now has get_out_of_jail community_chest card')
    elif pack == 'chance' and card.name == 'get_out_of_jail_free': # remember, this is an object equality test
        player.has_get_out_of_jail_chance_card = True
        logger.debug(player.player_name+ ' now has get_out_of_jail chance card')
    else: # if we arrive here, it means that the card we have is either not get out of jail free, or something else has gone wrong.
        logger.debug('something has gone wrong in set_get_out_of_jail_card_status. That is all I know.')
        logger.error("Exception")
        raise Exception


def bank_cash_transaction(player, card, current_gameboard):
    """
    Player either receives or gives an amount to the bank, as specified on the card.
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure. In this function it is unused.
    :return: None
    """
    logger.debug('executing bank_cash_transaction for '+ player.player_name)
    if card.amount < 0:
        player.charge_player(-1*card.amount, current_gameboard, bank_flag=True)
        # add to game history
        current_gameboard['history']['function'].append(player.charge_player)
        params = dict()
        params['self'] = player
        params['amount'] = -1*card.amount
        params['description'] = 'bank cash transaction'
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)
    elif card.amount > 0:
        code = player.receive_cash(card.amount, current_gameboard, bank_flag=True)
        # add to game history
        if code == flag_config_dict['successful_action']:
            current_gameboard['history']['function'].append(player.receive_cash)
            params = dict()
            params['self'] = player
            params['amount'] = card.amount
            params['description'] = 'bank cash transaction'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(code)
        elif code == flag_config_dict['failure_code']:
            logger.debug('Transaction broke due to insufficient funds. Player does not receive the stated funds.')
    else:
        logger.debug('Something broke in bank_cash_transaction. That is all I know.')
        logger.error("Exception")
        raise Exception


def player_cash_transaction(player, card, current_gameboard):
    """
    Player either receives or gives an amount to each player, as specified on the card.
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing player_cash_transaction for '+ player.player_name)
    if card.amount_per_player < 0:
        for p in current_gameboard['players']:
            if p == player or p.status == 'lost':
                continue

            code = p.receive_cash(-1*card.amount_per_player, current_gameboard, bank_flag=False)
            # add to game history
            if code == flag_config_dict['successful_action']:
                current_gameboard['history']['function'].append(p.receive_cash)
                params = dict()
                params['self'] = p
                params['amount'] = -1*card.amount_per_player
                params['description'] = 'player cash transaction'
                current_gameboard['history']['param'].append(params)
                current_gameboard['history']['return'].append(code)
            else:
                logger.debug("Not sure what happened! Something broke!")
                logger.error("Exception")
                raise Exception

            player.charge_player(-1*card.amount_per_player, current_gameboard, bank_flag=False)
            # add to game history
            current_gameboard['history']['function'].append(player.charge_player)
            params = dict()
            params['self'] = player
            params['amount'] = -1*card.amount_per_player
            params['description'] = 'player cash transaction'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(None)

    elif card.amount_per_player > 0:
        for p in current_gameboard['players']:
            if p == player or p.status == 'lost':
                continue

            code = player.receive_cash(card.amount_per_player, current_gameboard, bank_flag=False)
            # add to game history
            if code == flag_config_dict['successful_action']:
                current_gameboard['history']['function'].append(player.receive_cash)
                params = dict()
                params['self'] = player
                params['amount'] = card.amount_per_player
                params['description'] = 'player cash transaction'
                current_gameboard['history']['param'].append(params)
                current_gameboard['history']['return'].append(code)
            else:
                logger.debug("Not sure what happened! Something broke!")
                logger.error("Exception")
                raise Exception

            p.charge_player(card.amount_per_player, current_gameboard, bank_flag=False)
            # add to game history
            current_gameboard['history']['function'].append(p.charge_player)
            params = dict()
            params['self'] = p
            params['amount'] = card.amount_per_player
            params['description'] = 'player cash transaction'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(None)


def contingent_bank_cash_transaction(player, card, current_gameboard):
    """
    Calls the contingency action specified by the card. In the default board, it is one of calculate_street_repair_cost or
    calculate_general_repair_cost.
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing contingent_bank_cash_transaction for '+ player.player_name)
    card.contingency(player, card, current_gameboard)
    # add to game history
    current_gameboard['history']['function'].append(card.contingency)
    params = dict()
    params['player'] = player
    params['card'] = card
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


def calculate_street_repair_cost(player, card, current_gameboard): # assesses, not just calculates
    """
    Assesses street repair cost using a pre-defined formula and then charges it to the player.
    :param player: Player instance.
    :param card: Card instance. In this function it is unused.
    :param current_gameboard: A dict. The global gameboard data structure. In this function it is unused.
    :return: None
    """
    logger.debug('executing calculate_street_repair_cost for '+player.player_name)
    cost_per_house = 40
    cost_per_hotel = 115
    cost = player.num_total_houses*cost_per_house+player.num_total_hotels*cost_per_hotel
    player.charge_player(cost, current_gameboard, bank_flag=True)
    # add to game history
    current_gameboard['history']['function'].append(player.charge_player)
    params = dict()
    params['self'] = player
    params['amount'] = cost
    params['description'] = 'street repair'
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


def move_player__check_for_go(player, card, current_gameboard):
    """
    Move the player to the destination specified on the card, and check for go in the process.
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing move_player__check_for_go for '+player.player_name)
    logger.debug('destination specified on card is '+ card.destination.name)
    new_position = card.destination.start_position
    _move_player__check_for_go(player, new_position, current_gameboard)


def move_to_nearest_utility__pay_or_buy__check_for_go(player, card, current_gameboard):
    """
    Move the player to the 'nearest' utility (remember to check backwards as well!). The player can buy this property
    if it is owned by the bank, but if not, the player has to pay according to a specific rule in the card (see
    the Monopoly card/rules description in the repo for the rule). Note that the payment could differ from what the
    player would be paying if he/she had landed 'normally' (i.e. after a dice roll) on that utility.
    :param player: Player instance.
    :param card: Card instance. In this function it is unused.
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing move_to_nearest_utility__pay_or_buy__check_for_go '+ player.player_name)
    utility_positions = current_gameboard['utility_positions']
    min_utility_position = utility_positions[0]
    min_utility_distance = _calculate_board_distance(player.current_position, min_utility_position)
    for u in utility_positions:
        dist = _calculate_board_distance(player.current_position, u)
        if dist < min_utility_distance:
            min_utility_distance = dist
            min_utility_position = u

    logger.debug('The utility position that player is being moved to is '+current_gameboard['location_sequence'][min_utility_position].name)

    go_position = current_gameboard['go_position']
    go_increment = current_gameboard['go_increment']

    if 'auxiliary_check' in current_gameboard:
        current_gameboard['auxiliary_check'](player, min_utility_position, current_gameboard)

    if _has_player_passed_go(player.current_position, min_utility_position, go_position):
        if 'auxiliary_check_for_go' in current_gameboard:
            current_gameboard['auxiliary_check_for_go'](player, min_utility_position, current_gameboard)
        code = player.receive_cash(go_increment, current_gameboard, bank_flag=True)
        # add to game history
        if code == flag_config_dict['successful_action']:
            current_gameboard['history']['function'].append(player.receive_cash)
            params = dict()
            params['self'] = player
            params['amount'] = go_increment
            params['description'] = 'go increment'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(code)
        else:
            logger.debug('Current cash balance with the bank = '+ str(current_gameboard['bank'].total_cash_with_bank))
            logger.debug("Player supposed to receive go increment, but bank has no sufficient funds, hence unable to pay player." +
                         "Player will have to pass GO position without receiving go increment!")

    player.update_player_position(min_utility_position, current_gameboard) # update this only after checking for go
    # add to game history
    current_gameboard['history']['function'].append(player.update_player_position)
    params = dict()
    params['self'] = player
    params['new_position'] = min_utility_position
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)

    current_loc = current_gameboard['location_sequence'][player.current_position]

    if current_loc.loc_class != 'utility':  # simple check
        logger.debug('location is supposed to be a utility...what happened?')
        logger.error("Exception")
        raise Exception

    if 'bank.Bank' in str(type(current_loc.owned_by)): # we're forced to use this hack to avoid an import.
        logger.debug('utility is owned by bank. Player will have option to purchase.')
        player.process_move_consequences(current_gameboard)
        # add to game history
        current_gameboard['history']['function'].append(player.process_move_consequences)
        params = dict()
        params['self'] = player
        params['current_gameboard'] = current_gameboard
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)
        return
    else:
        amount_due = current_gameboard['current_die_total']*10
        player.charge_player(amount_due, current_gameboard, bank_flag=False)
        # add to game history
        current_gameboard['history']['function'].append(player.charge_player)
        params = dict()
        params['self'] = player
        params['amount'] = amount_due
        params['number of utilities'] = current_loc.owned_by.num_utilities_possessed
        params['description'] = 'utility dues'
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)

        current_owner = current_loc.owned_by
        code = current_owner.receive_cash(amount_due, current_gameboard, bank_flag=False)
        if code == flag_config_dict['successful_action']:
            # add to game history
            current_gameboard['history']['function'].append(current_owner.receive_cash)
            params = dict()
            params['self'] = current_owner
            params['amount'] = amount_due
            params['number of utilities'] = current_loc.owned_by.num_utilities_possessed
            params['description'] = 'utility dues'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(code)
        else:
            logger.debug("Not sure what happened! Something broke!")
            logger.error("Exception")
            raise Exception


def move_to_nearest_railroad__pay_double_or_buy__check_for_go(player, card, current_gameboard):
    """
    Move the player to the 'nearest' railroad (remember to check backwards as well!). The player can buy this property
    if it is owned by the bank, but if not, the player has to pay according to a specific rule in the card (see
    the Monopoly card/rules description in the repo for the rule). Note that the payment could differ from what the
    player would be paying if he/she had landed 'normally' (i.e. after a dice roll) on that railroad.
    :param player: Player instance.
    :param card: Card instance. In this function it is unused.
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing move_to_nearest_railroad__pay_double_or_buy__check_for_go for '+ player.player_name)
    railroad_positions = current_gameboard['railroad_positions']
    min_railroad_position = railroad_positions[0]
    min_railroad_distance = _calculate_board_distance(player.current_position, railroad_positions[0])
    for u in railroad_positions:
        dist = _calculate_board_distance(player.current_position, u)
        if dist < min_railroad_distance:
            min_railroad_distance = dist
            min_railroad_position = u

    logger.debug('The railroad position that player is being moved to is '+ current_gameboard['location_sequence'][
        min_railroad_position].name)

    go_position = current_gameboard['go_position']
    go_increment = current_gameboard['go_increment']

    if 'auxiliary_check' in current_gameboard:
        current_gameboard['auxiliary_check'](player, min_railroad_position, current_gameboard)

    if _has_player_passed_go(player.current_position, min_railroad_position, go_position):
        if 'auxiliary_check_for_go' in current_gameboard:
            current_gameboard['auxiliary_check_for_go'](player, min_railroad_position, current_gameboard)
        code = player.receive_cash(go_increment, current_gameboard, bank_flag=True)
        # add to game history
        if code == flag_config_dict['successful_action']:
            current_gameboard['history']['function'].append(player.receive_cash)
            params = dict()
            params['self'] = player
            params['amount'] = go_increment
            params['description'] = 'go increment'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(code)
        else:
            logger.debug('Current cash balance with the bank = '+ str(current_gameboard['bank'].total_cash_with_bank))
            logger.debug("Player supposed to receive go increment, but bank has no sufficient funds, hence unable to pay player." +
                         "Player will have to pass GO position without receiving go increment!")

    player.update_player_position(min_railroad_position, current_gameboard) # update this only after checking for go
    # add to game history
    current_gameboard['history']['function'].append(player.update_player_position)
    params = dict()
    params['self'] = player
    params['new_position'] = min_railroad_position
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)

    current_loc = current_gameboard['location_sequence'][player.current_position]

    if current_loc.loc_class != 'railroad': # simple check
        logger.debug('location is supposed to be a railroad...what happened?')
        logger.error("Exception")
        raise Exception

    if 'bank.Bank' in str(type(current_loc.owned_by)):  # we're forced to use this hack to avoid an import.
        logger.debug('railroad is owned by bank. Player will have option to purchase.')
        player.process_move_consequences(current_gameboard)
        # add to game history
        current_gameboard['history']['function'].append(player.process_move_consequences)
        params = dict()
        params['self'] = player
        params['current_gameboard'] = current_gameboard
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)
        return
    else:
        amount_due = 2 * RailroadLocation.calculate_railroad_dues(current_loc, current_gameboard)
        player.charge_player(amount_due, current_gameboard, bank_flag=False)
        # add to game history
        current_gameboard['history']['function'].append(player.charge_player)
        params = dict()
        params['self'] = player
        params['amount'] = amount_due
        params['number of railroads'] = current_loc.owned_by.num_railroads_possessed
        params['description'] = 'railroad dues'
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)

        current_owner = current_loc.owned_by
        code = current_owner.receive_cash(amount_due, current_gameboard, bank_flag=False)
        if code == flag_config_dict['successful_action']:
            # add to game history
            current_gameboard['history']['function'].append(current_owner.receive_cash)
            params = dict()
            params['self'] = current_owner
            params['amount'] = amount_due
            params['number of railroads'] = current_loc.owned_by.num_railroads_possessed
            params['description'] = 'railroad dues'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(code)
        else:
            logger.debug("Not sure what happened! Something broke!")
            logger.error("Exception")
            raise Exception


def calculate_general_repair_cost(player, card, current_gameboard):
    """
    Assesses street repair cost using a pre-defined formula and then charges it to the player.
    :param player: Player instance.
    :param card: Card instance. In this function it is unused.
    :param current_gameboard: A dict. The global gameboard data structure. In this function it is unused.
    :return: None
    """
    logger.debug('executing calculate_general_repair_cost action for '+ player.player_name)
    cost_per_house = 25
    cost_per_hotel = 100
    cost = player.num_total_houses * cost_per_house + player.num_total_hotels * cost_per_hotel
    player.charge_player(cost, current_gameboard, bank_flag=True)
    # add to game history
    current_gameboard['history']['function'].append(player.charge_player)
    params = dict()
    params['self'] = player
    params['amount'] = cost
    params['description'] = 'general repair'
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


def move_player_relative(player, card, current_gameboard):
    """
    The new relative position by which to move the player is specified in the card.
    :param player: Player instance.
    :param card: Card instance
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    logger.debug('executing move_player_relative action for '+player.player_name)
    move_player_after_die_roll(player, card.new_relative_position, current_gameboard, True)
    # add to game history
    current_gameboard['history']['function'].append(move_player_after_die_roll)
    params = dict()
    params['player'] = player
    params['rel_move'] = card.new_relative_position
    params['current_gameboard'] = current_gameboard
    params['check_for_go'] = True
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


def move_player_after_die_roll(player, rel_move, current_gameboard, check_for_go=True):
    """
     This is a utility function used in gameplay, rather than card draw.
     The goal of the function is to move the player by rel_move steps forward from the player's current position.
     if check_for_go is disabled, we will not check for go and the player's cash will not be incremented as it would be
     if we did check and the player passed go.
     It's important to note that if we are 'visiting' in jail, this function will not set the player.currently_in_jail field to True, since it shouldn't.
    :param player: Player instance. This is the player to move.
    :param rel_move: An integer. The number of steps by which to move the player forward.
    :param current_gameboard: A dict. The global gameboard data structure.
    :param check_for_go: A boolean. If True, as set by default, then we will check for go and increment player cash by
    go_increment if we land on go or pass it.
    :return:  None
    """
    logger.debug('executing move_player_after_die_roll for '+player.player_name+' by '+str(rel_move)+' relative steps forward.')
    num_locations = len(current_gameboard['location_sequence'])
    go_position = current_gameboard['go_position']
    go_increment = current_gameboard['go_increment']

    new_position = (player.current_position+rel_move) % num_locations

    if 'auxiliary_check' in current_gameboard:
        current_gameboard['auxiliary_check'](player, new_position, current_gameboard)

    if check_for_go:
        if _has_player_passed_go(player.current_position, new_position, go_position):
            if 'auxiliary_check_for_go' in current_gameboard:
                current_gameboard['auxiliary_check_for_go'](player, new_position, current_gameboard)
            logger.debug(player.player_name+' passes Go.')
            code = player.receive_cash(go_increment, current_gameboard, bank_flag=True)
            # add to game history
            if code == flag_config_dict['successful_action']:
                current_gameboard['history']['function'].append(player.receive_cash)
                params = dict()
                params['self'] = player
                params['amount'] = go_increment
                params['description'] = 'go increment'
                current_gameboard['history']['param'].append(params)
                current_gameboard['history']['return'].append(code)
            else:
                logger.debug('Current cash balance with the bank = '+ str(current_gameboard['bank'].total_cash_with_bank))
                logger.debug("Player supposed to receive go increment, but bank has no sufficient funds, hence unable to pay player." +
                             "Player will have to pass GO position without receiving go increment!")

    player.update_player_position(new_position, current_gameboard)  # update this only after checking for go
    # add to game history
    current_gameboard['history']['function'].append(player.update_player_position)
    params = dict()
    params['self'] = player
    params['new_position'] = new_position
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


"""
All functions below are for internal use only and should never be invoked externally.
"""
def _has_player_passed_go(current_position, new_position, go_position):
    """
    Function to determine whether the player passes, or is on, Go if the player moves from current position to new position.
    :param current_position: An integer. Specifies the position from which the player is moving.
    :param new_position: An integer. Specifies the position to which the player is moving.
    :param go_position: An integer. Specifies the go position. In the default board, it is just set to 0.
    :return: A boolean. True if the player is on, or has passed, go, and False otherwise.
    """
    if new_position == go_position: # we've landed on go
        return True

    elif new_position == current_position:  # we've gone all round the board
        return True

    elif current_position < new_position:
        if new_position >= go_position > current_position:
            return True

    elif current_position > new_position:
        if go_position > current_position or go_position <= new_position:
            return True

    return False # we've exhausted the possibilities. If it reaches here, we haven't passed go.


def _calculate_board_distance(position_1, position_2):
    """
    Calculate minimum distance between position_1 or position_2 in terms of locations in between.
    :param position_1: An integer.
    :param position_2: An integer.
    :return: returns shortest distance (forward or backward; hence this is NOT necessarily equal to the 'dice' total it would take
    to get here)
    """
    if position_1 - position_2 < 0 :
        return position_2 - position_1
    else:
        return position_1 - position_2


def _move_player__check_for_go(player, new_position, current_gameboard):
    """
    A private helper function which moves the player and checks for go.
    :param player: Player instance.
    :param new_position: An integer. Specifies the position to which the player is moving.
    :param current_gameboard: A dict. The global gameboard data structure
    :return: None
    """
    # the private version
    go_position = current_gameboard['go_position']
    go_increment = current_gameboard['go_increment']

    if 'auxiliary_check' in current_gameboard:
        current_gameboard['auxiliary_check'](player, new_position, current_gameboard)

    if _has_player_passed_go(player.current_position, new_position, go_position):
        if 'auxiliary_check_for_go' in current_gameboard:
            current_gameboard['auxiliary_check_for_go'](player, new_position, current_gameboard)
        code = player.receive_cash(go_increment, current_gameboard, bank_flag=True)
        # add to game history
        if code == flag_config_dict['successful_action']:
            current_gameboard['history']['function'].append(player.receive_cash)
            params = dict()
            params['self'] = player
            params['amount'] = go_increment
            params['description'] = 'go increment'
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(code)
        else:
            logger.debug('Current cash balance with the bank = '+ str(current_gameboard['bank'].total_cash_with_bank))
            logger.debug("Player supposed to receive go increment, but bank has no sufficient funds, hence unable to pay player." +
                         "Player will have to pass GO position without receiving go increment!")

    player.update_player_position(new_position, current_gameboard) # update this only after checking for go
    # add to game history
    current_gameboard['history']['function'].append(player.update_player_position)
    params = dict()
    params['self'] = player
    params['new_position'] = new_position
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)

    player.process_move_consequences(current_gameboard)
    # add to game history
    current_gameboard['history']['function'].append(player.process_move_consequences)
    params = dict()
    params['self'] = player
    params['current_gameboard'] = current_gameboard
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)


def check_for_game_termination(current_gameboard, tot_time):
    """
    checks for game termination conditions, eg: runaway cash limit for player current cash, or to keep a tab on max time limit for a game
    :param current_gameboard: global gameboard data structure
    :param tot_time: total time the game has taken until this function was called
    :return: bool, true if game termination condition is met, else false
    """
    if diagnostics.max_cash_balance(current_gameboard) > 6000:   # this is our limit for runaway cash for testing purposes only.
        logger.debug("Game terminated since max cash balance exceeded limit.")
        return True
    else:
        return False


def check_for_winner(current_gameboard):
    """
    If the game is forced to terminate when it reaches game termination conditions, it is possible that there is more that 1 player remaining
    in the game as the game did not terminate normally in which case the last man standing would have been the winner.
    This function returns a winner in such a situation when there is more than one player remaining and the game is forced to terminate.
    From among the players left in the game (ie, those that did not lose by that time), this function returns the winner as the player
    with the "highest networth" when the game is forced to conclude.
    :param current_gameboard: Global gameboard data structure
    :return: winning player
    """
    winner = None
    max_global_networth = 0
    for pl in current_gameboard['players']:
        if pl.status == 'win':                  # in case the game entered game termination condition right after the winner was found, return original winner
            return winner
    for pl in current_gameboard['players']:   # in case there are no winners, we find the winner as the player with highest networth
        if pl.status != 'lost':
            networth_player = 0
            networth_player += pl.current_cash
            if pl.assets:
                for prop in pl.assets:
                    if prop.loc_class == 'real_estate':
                        networth_player += prop.price
                        networth_player += prop.num_houses*prop.price_per_house
                        networth_player += prop.num_hotels*prop.price_per_house*(current_gameboard['bank'].house_limit_before_hotel + 1)
                    elif prop.loc_class == 'railroad':
                        networth_player += prop.price
                    elif prop.loc_class == 'utility':
                        networth_player += prop.price
            if networth_player > max_global_networth:
                winner = pl
                max_global_networth = networth_player
            elif networth_player > 0 and winner is not None and networth_player == max_global_networth:
                # in the highly unlikely event that 2 players end up with the same networth, tie breaker is the player with higher current cash
                # if current cash also ends up being the same, then whoever entered the loop first just ends up getting the advantage of being the winner
                if pl.current_cash > winner.current_cash:
                    winner = pl

    return winner
