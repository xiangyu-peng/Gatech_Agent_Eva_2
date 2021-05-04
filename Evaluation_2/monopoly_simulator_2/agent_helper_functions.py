import logging
logger = logging.getLogger('monopoly_simulator.logging_info.agent_helper_func')

def will_property_complete_set(player, asset, current_gameboard):
    """
    A helper function that checks if the asset passed into this function will complete a color group for the player resulting
    in a monopoly.
    :param player: Player instance
    :param asset: Location instance
    :return: Boolean. True if the asset will complete a color set for the player, False otherwise. For railroads
    (or utilities), returns true only if player owns all other railroads (or utilities)
    """
    if type(asset) != dict:
        if asset.color is None:
            if asset.loc_class == 'railroad':
                if player.num_railroads_possessed == 3:
                    return True
                else:
                    return False
            elif asset.loc_class == 'utility':
                if player.num_utilities_possessed == 1:
                    return True
                else:
                    return False
            else:
                logger.error('This asset does not have a color and is neither utility nor railroad')
                logger.error("Exception")
        else:

            c = asset.color
            c_assets = current_gameboard['color_assets'][c]
            for c_asset in c_assets:
                if c_asset == asset:
                    continue
                else:
                    if c_asset not in player.assets:
                        return False
            return True  # if we got here, then eve

    else:
        if asset['color'] is None:
            if asset['loc_class'] == 'railroad':
                if player['num_railroads_possessed'] == 3:
                    return True
                else:
                    return False
            elif asset['loc_class'] == 'utility':
                if player['num_utilities_possessed'] == 1:
                    return True
                else:
                    return False
            else:
                logger.error('This asset does not have a color and is neither utility nor railroad')
                logger.error("Exception")
        else:
            c = asset['color']
            c_assets = []
            for name in current_gameboard['locations'].keys():
                if current_gameboard['locations'][name]['color'] == c:
                    c_assets.append(name)
            # c_assets = current_gameboard['color_assets'][c]
            for c_asset in c_assets:
                if c_asset == asset['name']:
                    continue
                else:
                    if c_asset not in player['assets']:
                        return False
            return True # if we

def identify_potential_mortgage(player, amount_to_raise, lone_constraint=False):
    """
    We return the property with the lowest mortgage such that it still exceeds or equals amount_to_raise, and if
    applicable, satisfies the lone constraint.
    :param player: Player instance. The potential mortgage has to be an unmortgaged property that this player owns.
    :param amount_to_raise: Integer. The amount of money looking to be raised from this mortgage.
    :param lone_constraint: Boolean. If true, we will limit our search to properties that meet the 'lone' constraint i.e.
    the property (if a railroad or utility) must be the only railroad or utility possessed by the player, or if colored,
    the property must be the only asset in its color class to be possessed by the player.
    :return: None, if a mortgage cannot be identified, otherwise a Location instance (representing the potential mortgage)
    """
    potentials = list()
    for a in player.assets:
        if a.is_mortgaged:
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        elif a.mortgage < amount_to_raise:
            continue
        elif lone_constraint:
            if is_property_lone(player, a):
                continue
        # a is a potential mortgage, and its mortgage price meets our fundraising bar.
        potentials.append((a,a.mortgage))

    if len(potentials) == 0:
        return None # nothing got identified
    else:
        sorted_potentials = sorted(potentials, key=lambda x: x[1]) # sort by mortgage in ascending order
        return sorted_potentials[0][0]


def identify_potential_sale(player, current_gameboard, amount_to_raise, lone_constraint=False):
    """
    All potential sales considered here will be to the bank. The logic is very similar to identify_potential_mortgage.
    We try to identify the cheapest property that will meet our fundraising bar (and if applicable, satisfy lone_constraint)
    :param player: Player instance. The potential sale has to be an unmortgaged property that this player owns.
    :param current_gameboard: The gameboard data structure
    :param amount_to_raise: Integer. The amount of money looking to be raised from this sale.
    :param lone_constraint: Boolean. If true, we will limit our search to properties that meet the 'lone' constraint i.e.
    the property (if a railroad or utility) must be the only railroad or utility possessed by the player, or if colored,
    the property must be the only asset in its color class to be possessed by the player.
    :return: None, if a sale cannot be identified, otherwise a Location instance (representing the potential sale)
    """
    potentials = list()
    for a in player.assets:
        if a.is_mortgaged: # technically, we can sell a property even if it is mortgaged. If your agent wants to make
            # this distinction, you should modify this helper function. Note that cash received will be lower than
            # price/2 however, since you have to free the mortgage before you can sell.
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        elif a.price*current_gameboard['bank'].property_sell_percentage < amount_to_raise:
            continue
        elif lone_constraint:
            if is_property_lone(player, a):
                continue
        # a is a potential sale, and its sale price meets our fundraising bar.
        potentials.append((a, a.price*current_gameboard['bank'].property_sell_percentage))

    if len(potentials) == 0:
        return None  # nothing got identified
    else:
        sorted_potentials = sorted(potentials, key=lambda x: x[1])  # sort by sale price in ascending order
        return sorted_potentials[0][0]


def is_property_lone(player, asset, current_gameboard=None):
    if type(asset) == dict:
        if asset['color'] is None:
            if asset['loc_class'] == 'railroad':
                if player['num_railroads_possessed'] == 1:
                    return True
            elif asset['loc_class'] == 'utility':
                if player['num_utilities_possessed'] == 1:
                    return True
            else:
                logger.error('This asset does not have a color and is neither utility nor railroad')
                logger.error("Exception")
        else:
            c = asset['color']
            for c_asset in player['assets']:
                if c_asset == asset['name']:
                    continue
                else:
                    if current_gameboard['locations'][c_asset]['loc_class'] == 'real_estate' and current_gameboard['locations'][c_asset]['color'] == c:  # player has another property with this color
                        return False
            return True  # if we got here, then only this asset (of its color class) is possessed by player.
    ####
    if asset.color is None:
        if asset.loc_class == 'railroad':
            if player.num_railroads_possessed == 1:
                return True
        elif asset.loc_class == 'utility':
            if player.num_utilities_possessed == 1:
                return True
        else:
            logger.error('This asset does not have a color and is neither utility nor railroad')
            logger.error("Exception")
    else:
        c = asset.color
        for c_asset in player.assets:
            if c_asset == asset:
                continue
            else:
                if c_asset.loc_class == 'real_estate' and c_asset.color == c: # player has another property with this color
                    return False
        return True # if we got here, then only this asset (of its color class) is possessed by player.


def identify_improvement_opportunity(player, current_gameboard):
    """
    Identify an opportunity to improve a property by building a house or hotel. This is a 'strategic' function; there
    are many other ways/strategies to identify improvement opportunities than the one we use here.
    :param player:
    :param current_gameboard:
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.improve_property by the calling function.
    """
    if type(player) == dict:
        potentials = list()
        for c in player['full_color_sets_possessed']:
            c_assets = []
            for name in current_gameboard['locations']:
                if current_gameboard['locations'][name]['color'] == c:
                    c_assets.append(current_gameboard['locations'][name])
            # c_assets = current_gameboard['color_assets'][c]
            for asset in c_assets:
                if can_asset_be_improved(asset,
                                         c_assets) and asset['price_per_house'] <= player['current_cash']:  # player must be able to afford the improvement
                    potentials.append((asset, asset_incremental_improvement_rent(asset) - asset['price_per_house']))
        if potentials:
            sorted_potentials = sorted(potentials, key=lambda x: (x[1], x[0]['name']),
                                       reverse=True)  # sort in descending order
            param = dict()
            param['player'] = player['player_name']
            param['asset'] = sorted_potentials[0][0]
            param['current_gameboard'] = "current_gameboard"
            param['add_house'] = True
            param['add_hotel'] = False
            if param['asset'].num_houses == current_gameboard['bank'].house_limit_before_hotel:
                param['add_hotel'] = True
                param['add_house'] = False
            return param
        else:
            return None


    potentials = list()
    for c in player.full_color_sets_possessed:
        c_assets = current_gameboard['color_assets'][c]
        for asset in c_assets:
            if can_asset_be_improved(asset,c_assets) and asset.price_per_house<=player.current_cash: # player must be able to afford the improvement
                potentials.append((asset,asset_incremental_improvement_rent(asset)-asset.price_per_house))
    if potentials:
        sorted_potentials = sorted(potentials, key=lambda x: (x[1], x[0].name), reverse=True) # sort in descending order
        param = dict()
        param['player'] = player
        param ['asset'] = sorted_potentials[0][0]
        param['current_gameboard'] = current_gameboard
        param['add_house'] = True
        param['add_hotel'] = False
        if param ['asset'].num_houses == current_gameboard['bank'].house_limit_before_hotel:
            param['add_hotel'] = True
            param['add_house'] = False
        return param
    else:
        return None


def identify_sale_opportunity_to_player(player, current_gameboard):
    """
    Identify an opportunity to sell a property currently owned by player to another player by making a
    sell property offer. This is a 'strategic' function; there
    are many other ways/strategies to identify such sales than the one we use here. All we do is identify if
    there is a player who needs a single property to complete a full color set and if that property is a 'lone'
    property for us. If such a player exists for some such
    property that we own, we offer it to the player at 50% markup. We do not offer mortgaged properties for sale.
    For simplicity, we do not offer railroads or utilities for sale either. Other agents may consider more sophisticated
    strategies to handle railroads and utilities.
    :param player:
    :param current_gameboard:
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function.
    """
    if type(asset) != dict:
        potentials = list()
        for a in player.assets:
            if a.loc_class != 'real_estate' or a.is_mortgaged:
                continue
            if a.color in player.full_color_sets_possessed:
                continue
            if is_property_lone(player, a):
                for p in current_gameboard['players']:
                    if p == player or p.status == 'lost':
                        continue
                    elif will_property_complete_set(p, a, current_gameboard):
                        # we make an offer!
                        param = dict()
                        param['from_player'] = player
                        param['asset'] = a
                        param['to_player'] = p
                        param['price'] = a.price * 1.5  # 50% markup on market price.
                        if param['price'] < param['to_player'].current_cash / 2:
                            param['price'] = param['to_player'].current_cash / 2  # how far would you go for a monopoly?
                        elif param['price'] > param['to_player'].current_cash:
                            # no point offering this to the player; they don't have money.
                            continue
                        potentials.append((param, param['price']))

        if not potentials:
            return None
        else:
            sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=True)  # sort in descending order
            return sorted_potentials[0][0]

    else:
        potentials = list()
        for a in player['assets']:
            a = current_gameboard['locations'][a]
            if a['loc_class'] != 'real_estate' or a['is_mortgaged']:
                continue
            if a['color'] in player['full_color_sets_possessed']:
                continue
            if is_property_lone(player, a, current_gameboard):
                for p in current_gameboard['players'].keys():
                    if p == player['player_name'] or current_gameboard['players'][p]['status'] == 'lost':
                        continue
                    elif will_property_complete_set(current_gameboard['players'][p], a, current_gameboard):
                        # we make an offer!
                        param = dict()
                        param['from_player'] = player
                        param['asset'] = a
                        param['to_player'] = p
                        param['price'] = a['price'] * 1.5  # 50% markup on market price.
                        if param['price'] < param['to_player']['current_cash'] / 2:
                            param['price'] = param['to_player']['current_cash'] / 2  # how far would you go for a monopoly?
                        elif param['price'] > param['to_player']['current_cash']:
                            # no point offering this to the player; they don't have money.
                            continue
                        potentials.append((param, param['price']))

        if not potentials:
            return None
        else:
            sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=True)  # sort in descending order
            return sorted_potentials[0][0]


def can_asset_be_improved(asset, same_color_assets):
    """
    This function does not check if all the same colored assets are owned by the same player. This is something that
    should have been checked much earlier in the code. All that we check here is whether it is permissible to improve
    asset under the assumption that the asset, and all other assets of that color, belong to one player. We also do
    not check here whether the game board is in an incorrect state (i.e. if somehow the uniform development rule
    has been violated).
    We are also not checking affordability of the improvement since the player is not specified.
    :param asset: asset that needs to be improved
    :param same_color_assets: other assets of the same color
    :return: True if asset can be improved else False
    """
    if type(asset) == dict:
        if asset['loc_class'] != 'real_estate' or asset['is_mortgaged']:
            return False
        if asset['num_hotels'] > 0:
            return False  # we can't improve any further
        if asset['num_houses'] == 0:
            return True
        count = 0
        for c_asset in same_color_assets:
            if c_asset['color'] != asset['color']:
                logger.error('asset color is not the same as the color of the set')
                logger.error(
                    "Exception")  # if this has happened, it probably indicates a problem in the code. That's why we don't return false
            if c_asset['name'] == asset['name']:
                continue
            if c_asset['num_hotels'] > 0 and asset['num_houses'] == 4:
                return True  # we can build a hotel on asset
            if c_asset['num_houses'] > asset['num_houses']:
                return True
            if c_asset['num_houses'] == asset['num_houses']:
                count += 1

        if count == len(
                same_color_assets) - 1:  # every asset in same_color has the same no. of houses as the current asset, hence
            # it can be improved (either by building another house, or a hotel).
            return True

        return False





    if asset.loc_class != 'real_estate' or asset.is_mortgaged:
        return False
    if asset.num_hotels > 0:
        return False # we can't improve any further
    if asset.num_houses == 0:
        return True
    count = 0
    for c_asset in same_color_assets:
        if c_asset.color != asset.color:
            logger.error('asset color is not the same as the color of the set')
            logger.error("Exception") # if this has happened, it probably indicates a problem in the code. That's why we don't return false
        if c_asset == asset:
            continue
        if c_asset.num_hotels > 0 and asset.num_houses == 4 :
            return True # we can build a hotel on asset
        if c_asset.num_houses > asset.num_houses:
            return True
        if c_asset.num_houses == asset.num_houses:
            count += 1

    if count == len(same_color_assets) - 1: # every asset in same_color has the same no. of houses as the current asset, hence
        # it can be improved (either by building another house, or a hotel).
        return True

    return False


def asset_incremental_improvement_rent(asset):
    """
    If we were to incrementally improve this asset, how much extra rent would we get?
    :param asset: the property to be (hypothetically) incrementally improved
    :return: Integer representing the additional rent we get if we were to incrementally improve this property. Note that
    we do not check if we 'can' improve it, we return assuming that we can.
    """
    if asset.num_hotels > 0:
        logger.error("Exception") # there is no incremental improvement possible. how did we get here?
    if asset.num_houses == 4:
        return asset.rent_hotel-asset.rent_4_houses
    elif asset.num_houses == 3:
        return asset.rent_4_houses - asset.rent_3_houses
    elif asset.num_houses == 2:
        return asset.rent_3_houses - asset.rent_2_houses
    elif asset.num_houses == 1:
        return asset.rent_2_houses - asset.rent_1_house
    else:
        return asset.rent_1_house - (asset.rent*2) # remember, if the house can be improved, then it is monopolized, so twice the rent is being charged even without houses.


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


def identify_property_trade_offer_to_player(player, current_gameboard):
    """
    Identify an opportunity to sell a property currently owned by player to another player by making a
    trade offer. This is a 'strategic' function; there
    are many other ways/strategies to identify such sales than the one we use here. All we do is identify if
    there is a player who needs a single property to complete a full color set and if that property is a 'lone'
    property for us. If such a player exists for some such
    property that we own, we offer it to the player at 50% markup. We do not offer mortgaged properties for sale.
    For simplicity, we do not offer railroads or utilities for sale either. Other agents may consider more sophisticated
    strategies to handle railroads and utilities.
    :param player: player who wants to offer its properties
    :param current_gameboard: The gameboard data structure
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function.
    """
    if type(player) != dict:
        potentials = list()
        sorted_player_assets = _set_to_sorted_list_assets(player.assets)
        for a in sorted_player_assets:
            if a.loc_class != 'real_estate' or a.is_mortgaged:
                continue
            if a.color in player.full_color_sets_possessed:
                continue
            if is_property_lone(player, a):
                for p in current_gameboard['players']:
                    if p == player or p.status == 'lost':
                        continue
                    elif will_property_complete_set(p, a, current_gameboard):
                        # we make an offer!
                        param = dict()
                        param['from_player'] = player
                        param['asset'] = a
                        param['to_player'] = p
                        param['price'] = a.price*1.5 # 50% markup on market price.
                        if param['price'] < param['to_player'].current_cash / 2:
                            param['price'] = param['to_player'].current_cash / 2  # how far would you go for a monopoly?
                        elif param['price'] > param['to_player'].current_cash:
                            # no point offering this to the player; they don't have money.
                            continue
                        potentials.append((param, param['price']))

        if not potentials:
            return None
        else:
            sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=True)  # sort in descending order
            return sorted_potentials

    else:
        potentials = list()
        sorted_player_assets = _set_to_sorted_list_assets(player['assets'])
        for a in sorted_player_assets:
            a = current_gameboard['locations'][a]
            if a['loc_class'] != 'real_estate' or a['is_mortgaged']:
                continue
            if a['color'] in player['full_color_sets_possessed']:
                continue
            if is_property_lone(player, a, current_gameboard):
                for p in current_gameboard['players'].keys():
                    p = current_gameboard['players'][p]
                    if p['player_name'] == player['player_name'] or p['status'] == 'lost':
                        continue
                    elif will_property_complete_set(p, a, current_gameboard):
                        # we make an offer!
                        param = dict()
                        param['from_player'] = player
                        param['asset'] = a
                        param['to_player'] = p
                        param['price'] = a['price'] * 1.5  # 50% markup on market price.
                        if param['price'] < param['to_player']['current_cash'] / 2:
                            param['price'] = param['to_player']['current_cash'] / 2  # how far would you go for a monopoly?
                        elif param['price'] > param['to_player']['current_cash']:
                            # no point offering this to the player; they don't have money.
                            continue
                        potentials.append((param, param['price']))

        if not potentials:
            return None
        else:
            sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=True)  # sort in descending order
            return sorted_potentials

def identify_property_trade_wanted_from_player(player, current_gameboard):
    """
    Identify an opportunity to buy a property currently owned another player by making a
    trade offer for the purpose of increasing the number of monopolies. This is a 'strategic' function; there
    are many other ways/strategies to identify such sales than the one we use here. All we do is identify if
    there is a player who has a lone property which will help us acquire a monopoly for that color group.
    If such a player exists for some such
    property, we request to buy it from the player using trading strategies based on the situation.
    We do not request to buy mortgaged properties.
    For simplicity, we do not request for railroads or utilities either. Other agents may consider more sophisticated
    strategies to handle railroads and utilities.
    :param player: player who wants properties from other players and hence invokes this function
    :param current_gameboard: The gameboard data structure
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function.
    """
    if type(player) == dict:
        potentials = list()
        for p in current_gameboard['players']:
            p = current_gameboard['players'][p]
            if p['player_name'] == player['player_name'] or p['status'] == 'lost':
                continue
            else:
                sorted_player_assets = _set_to_sorted_list_assets(p['assets'])
                for a in sorted_player_assets:
                    a = current_gameboard['locations'][a]
                    if a['loc_class'] != 'real_estate' or a['is_mortgaged']:
                        continue
                    if a['color'] in p['full_color_sets_possessed']:
                        continue
                    if is_property_lone(p, a, current_gameboard):
                        if will_property_complete_set(player, a, current_gameboard):
                            # we request for the property!
                            param = dict()
                            param['from_player'] = p
                            param['asset'] = a
                            param['to_player'] = player
                            param[
                                'price'] = a['price'] * 0.75  # willing to buy at 0.75 times the market price since other properties leading to monopoly are being offered
                            # and this is a good enough incentive to release a lone property at the advantage of gaining a monopoly. Hence only 0.75*market_price is offered.
                            # But if I have no properties to offer to the other player while curating the trade offer in the curate_trade_offer function,
                            # then this basically becomes a buy offer and the price offered will be higher than the market price since the other player has no other incentive
                            # to sell it to me besides a price better than the market price. (this is taken care of in the curate trade offer.)
                            if param['price'] > player['current_cash']:
                                # I don't have money to buy this property even at a reduced price.
                                continue
                            potentials.append((param, param['price']))

        if not potentials:
            return None
        else:
            sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=False)  # sort in ascending order
            return sorted_potentials




    potentials = list()
    for p in current_gameboard['players']:
        if p == player or p.status == 'lost':
            continue
        else:
            sorted_player_assets = _set_to_sorted_list_assets(p.assets)
            for a in sorted_player_assets:
                if a.loc_class != 'real_estate' or a.is_mortgaged:
                    continue
                if a.color in p.full_color_sets_possessed:
                    continue
                if is_property_lone(p, a):
                    if will_property_complete_set(player, a, current_gameboard):
                        # we request for the property!
                        param = dict()
                        param['from_player'] = p
                        param['asset'] = a
                        param['to_player'] = player
                        param['price'] = a.price*0.75 # willing to buy at 0.75 times the market price since other properties leading to monopoly are being offered
                        # and this is a good enough incentive to release a lone property at the advantage of gaining a monopoly. Hence only 0.75*market_price is offered.
                        #But if I have no properties to offer to the other player while curating the trade offer in the curate_trade_offer function,
                        # then this basically becomes a buy offer and the price offered will be higher than the market price since the other player has no other incentive
                        # to sell it to me besides a price better than the market price. (this is taken care of in the curate trade offer.)
                        if param['price'] > player.current_cash:
                            # I don't have money to buy this property even at a reduced price.
                            continue
                        potentials.append((param, param['price']))

    if not potentials:
        return None
    else:
        sorted_potentials = sorted(potentials, key=lambda x: x[1], reverse=False)  # sort in ascending order
        return sorted_potentials


def curate_trade_offer(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag):
    """
    Generates a trade offer from the potential offer list, potential request list and the purpose indicated by
    the purpose flag for background_agent_v2
    :param player: player who wants to make the trade offer
    :param potential_offer_list: List of potential property offers that can be made to other players because they are lone properties
    for this player and will result in a monopoly to the player to whom it is being offered. These offers are sorted in descending order of the
    amount of cash this player would get if the other player accepts the offer. The first property yields the most cash.
    :param potential_request_list: List of potential properties that this player would like to buy because they are lone properties for the players
    to whom this player makes a trade offer and will lead this player into getting a monopoly. These are sorted in the ascending order of the amount
    this player is willing to pay for the property. The property that costs this player the least is on the top of the list.
    :param current_gameboard: The gameboard data structure
    :purpose_flag: indicates the purpose of making the trade.
    purpose_flag=1 implies that it is purely a make_sell_property_offer because
    the player is urgently in need of cash or is trying to see if it can get to sell one of its lone properties at a high premium because the other
    player can get a monopoly from this offer.
    purpose_flag=2 implies that it can be a buy offer or exchange of properties and cash to increase number of monopolies.
    :return: a parameter dictionary or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function.
    """
    param = dict()
    trade_offer = dict()
    trade_offer['property_set_offered'] = set()
    trade_offer['property_set_wanted'] = set()
    trade_offer['cash_offered'] = 0
    trade_offer['cash_wanted'] = 0
    if purpose_flag == 1:
        #goal - somehow increase current cash since I am almost broke
        #we just made an offer to a player whose property can yield us maximum money and will also enable that player to gain a monopoly at the cost of a premium price.
        if not potential_offer_list:
            return None
        else:
            trade_offer['property_set_offered'].add(potential_offer_list[0][0]['asset'])
            trade_offer['cash_wanted'] = potential_offer_list[0][0]['price']
            param['from_player'] = player
            param['to_player'] = potential_offer_list[0][0]['to_player']
            param['offer'] = trade_offer
    elif purpose_flag == 2:
        #goal - increase monopolies since I have sufficient cash in hand
        #constraint - I will only trade and request only 1 property at a time.
        if not potential_offer_list and not potential_request_list:
            return None

        elif not potential_offer_list and potential_request_list:
            #constraint = request only 1 property
            #nothing to offer (I have no lone properties that could lead other players into getting a monopoly) but I want some
            #properties that could lead me into getting a monopoly, then I will request the property at 1.125 times the market price
            #provided that offer doesnot make me broke (Note that if I had properties to offer to the other player to help him or her get a monopoly
            #then I would only pay 0.75 times the market price. But now its my need, so I am willing to pay 1.125 times the market price.)
            if player.current_cash - potential_request_list[0][0]['price']*1.5 > current_gameboard['go_increment']/2:
                trade_offer['cash_offered'] = potential_request_list[0][0]['price']*1.5  #(0.75*1.5=1.125)
                trade_offer['property_set_wanted'].add(potential_request_list[0][0]['asset'])
                param['from_player'] = player
                param['to_player'] = potential_request_list[0][0]['from_player']
                param['offer'] = trade_offer
            else:
                return None #No cash to make the trade

        elif not potential_request_list and potential_offer_list:
            #try giving away one lone property for a very high premium which was already calculated in identify_property_trade_offer_to_player function
            trade_offer['property_set_offered'].add(potential_offer_list[0][0]['asset'])
            trade_offer['cash_wanted'] = potential_offer_list[0][0]['price']
            param['from_player'] = player
            param['to_player'] = potential_offer_list[0][0]['to_player']
            param['offer'] = trade_offer

        else:
            # want to make a trade offer such that I lose less cash and gain max cash and also get a monopoly
            # potential_request_list is sorted in ascending order of cash offered for the respective requested property
            # potential_offer_list is sorted in descending order of cash received for the respective offered property
            found_player_flag = 0
            saw_players = []
            for req in potential_request_list:
                player_2 = req[0]['from_player']
                if player_2 not in saw_players:
                    saw_players.append(player_2)
                    for off in potential_offer_list:
                        if off[0]['to_player'] == player_2:
                            found_player_flag = 1
                            trade_offer['property_set_offered'].add(off[0]['asset'])
                            trade_offer['cash_wanted'] = off[0]['price']
                            trade_offer['property_set_wanted'].add(req[0]['asset'])
                            trade_offer['cash_offered'] = req[0]['price']
                            param['from_player'] = player
                            param['to_player'] = player_2
                            param['offer'] = trade_offer
                            break
                else:
                    continue
                if found_player_flag == 1:
                    break

            #if found_player_flag=0, that means we werent able to establish a one on one trade offer, ie I couldnot find a player to whom I could sell my property
            #in return for their property
            if found_player_flag == 0:
                #exchange property not possible --> so we just try to buy property if we have enough cash in order to gain more monopolies at 1.125 times the market price
                if player.current_cash - potential_request_list[0][0]['price']*1.5 > current_gameboard['go_increment']/2:
                    trade_offer['cash_offered'] = potential_request_list[0][0]['price']*1.5  #(0.75*1.5=1.125)
                    trade_offer['property_set_wanted'].add(potential_request_list[0][0]['asset'])
                    param['from_player'] = player
                    param['to_player'] = potential_request_list[0][0]['from_player']
                    param['offer'] = trade_offer
                else:
                    return None #No cash to make the trade

    return param


def curate_trade_offer_multiple_players(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag):
    """
    Generates trade offers from the potential offer list, potential request list and the purpose indicated by
    the purpose flag for background_agent_v3.
    Allows a player to make trade offers to MULTIPLE PLAYERS simultaneously.
    :param player: player who wanted to make the trade offer
    :param potential_offer_list: List of potential property offers that can be made to other players because they are lone properties
    for this player and will result in a monopoly to the player to whom it is being offered. These offers are sorted in descending order of the
    amount of cash this player would get if the other player accepts the offer. The first property yields the most cash.
    :param potential_request_list: List of potential properties that this player would like to buy because they are lone properties for the players
    to whom this player makes a trade offer and will lead this player into getting a monopoly. These are sorted in the ascending order of the amount
    this player is willing to pay for the property. The property that costs this player the least is on the top of the list.
    :param current_gameboard: The gameboard data structure
    :purpose_flag: indicates the purpose of making the trade.
    purpose_flag=1 implies that it is purely a make_sell_property_offer because
    the player is urgently in need of cash or is trying to see if it can get to sell one of its lone properties at a high premium because the other
    player can get a monopoly from this offer.
    purpose_flag=2 implies that it can be a buy offer or exchange of properties and cash to increase number of monopolies.
    :return: a list of parameter dictionaries or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function. Each parameter dictionary in the list corresponds to the
    trade offers to multiple players
    """
    param = dict()
    trade_offer = dict()
    trade_offer['property_set_offered'] = set()
    trade_offer['property_set_wanted'] = set()
    trade_offer['cash_offered'] = 0
    trade_offer['cash_wanted'] = 0
    offer_list_multiple_players = []

    if purpose_flag == 1:
        #goal - somehow increase current cash since I am almost broke
        #we just made an offer to a player whose property can yield us maximum money and will also enable that player to gain a monopoly at the cost of a premium price.
        if not potential_offer_list:
            return None
        else:
            player_list = []
            offer_list_multiple_players = []
            for item in potential_offer_list:
                if item[0]['to_player'] not in player_list:
                    param = dict()
                    trade_offer = dict()
                    trade_offer['property_set_offered'] = set()
                    trade_offer['property_set_wanted'] = set()
                    trade_offer['cash_offered'] = 0
                    trade_offer['cash_wanted'] = 0

                    trade_offer['property_set_offered'].add(item[0]['asset'])
                    trade_offer['cash_wanted'] = item[0]['price']
                    param['from_player'] = player
                    param['to_player'] = item[0]['to_player']
                    param['offer'] = trade_offer
                    player_list.append(item[0]['to_player'])
                    offer_list_multiple_players.append(param)

    elif purpose_flag == 2:
        #goal - increase monopolies since I have sufficient cash in hand
        #constraint - I will only trade and request only 1 property at a time.
        if not potential_offer_list and not potential_request_list:
            return None

        elif not potential_offer_list and potential_request_list:
            #constraint = request only 1 property
            #nothing to offer (I have no lone properties that could lead other players into getting a monopoly) but I want some
            #properties that could lead me into getting a monopoly, then I will request the property at 1.125 times the market price
            #provided that offer doesnot make me broke (Note that if I had properties to offer to the other player to help him or her get a monopoly
            #then I would only pay 0.75 times the market price. But now its my need, so I am willing to pay 1.25 times the market price.)
            player_list = []
            offer_list_multiple_players = []
            cash_to_be_offered_during_trade = 0
            for item in potential_request_list:
                if item[0]['from_player'] not in player_list:
                    param = dict()
                    trade_offer = dict()
                    trade_offer['property_set_offered'] = set()
                    trade_offer['property_set_wanted'] = set()
                    trade_offer['cash_offered'] = 0
                    trade_offer['cash_wanted'] = 0
                    cash_to_be_offered_during_trade += item[0]['price']*1.5
                    if player.current_cash - cash_to_be_offered_during_trade > current_gameboard['go_increment']/2:
                        trade_offer['cash_offered'] = item[0]['price']*1.5  #(0.75*1.5=1.125)
                        trade_offer['property_set_wanted'].add(item[0]['asset'])
                        param['from_player'] = player
                        param['to_player'] = item[0]['from_player']
                        param['offer'] = trade_offer
                        player_list.append(item[0]['from_player'])
                        offer_list_multiple_players.append(param)
                    else:
                        cash_to_be_offered_during_trade -= item[0]['price']*1.5

            if len(offer_list_multiple_players)==0:
                logger.debug("Wanted to make a trade offer but donot have the money or properties for it, so cannot make one.")
                return None #No cash to make the trade

        elif not potential_request_list and potential_offer_list:
            #try giving away one lone property for a very high premium which was already calculated in identify_property_trade_offer_to_player function
            player_list = []
            offer_list_multiple_players = []
            for item in potential_offer_list:
                if item[0]['to_player'] not in player_list:
                    param = dict()
                    trade_offer = dict()
                    trade_offer['property_set_offered'] = set()
                    trade_offer['property_set_wanted'] = set()
                    trade_offer['cash_offered'] = 0
                    trade_offer['cash_wanted'] = 0
                    if type(item[0]['asset']) == dict:
                        trade_offer['property_set_offered'].add(item[0]['asset']['name'])
                    else:
                        trade_offer['property_set_offered'].add(item[0]['asset'])
                    trade_offer['cash_wanted'] = item[0]['price']
                    param['from_player'] = player
                    param['to_player'] = item[0]['to_player']
                    param['offer'] = trade_offer
                    player_list.append(item[0]['to_player'])
                    offer_list_multiple_players.append(param)

            if len(offer_list_multiple_players)==0:
                logger.debug("Wanted to make a trade offer but donot have the money or properties for it, so cannot make one.")
                return None #No cash to make the trade

        else:
            # want to make a trade offer such that I lose less cash and gain max cash and also get a monopoly
            # potential_request_list is sorted in ascending order of cash offered for the respective requested property
            # potential_offer_list is sorted in descending order of cash received for the respective offered property
            found_player_flag = 0
            saw_players = []
            offer_list_multiple_players = []
            for req in potential_request_list:
                player_2 = req[0]['from_player']
                if player_2 not in saw_players:
                    saw_players.append(player_2)
                    for off in potential_offer_list:
                        if off[0]['to_player'] == player_2:
                            param = dict()
                            trade_offer = dict()
                            trade_offer['property_set_offered'] = set()
                            trade_offer['property_set_wanted'] = set()
                            trade_offer['cash_offered'] = 0
                            trade_offer['cash_wanted'] = 0

                            found_player_flag = 1
                            trade_offer['property_set_offered'].add(off[0]['asset'])
                            trade_offer['cash_wanted'] = off[0]['price']
                            trade_offer['property_set_wanted'].add(req[0]['asset'])
                            trade_offer['cash_offered'] = req[0]['price']
                            param['from_player'] = player
                            param['to_player'] = player_2
                            param['offer'] = trade_offer
                            offer_list_multiple_players.append(param)
                            break

                else:
                    continue

            #if found_player_flag=0, that means we werent able to establish a one on one trade offer, ie I couldnot find a player to whom I could sell my property
            #in return for their property
            if found_player_flag == 0:
                #exchange property not possible --> so we just try to buy property if we have enough cash in order to gain more monopolies
                player_list = []
                offer_list_multiple_players = []
                cash_to_be_offered_during_trade = 0
                for item in potential_request_list:
                    if item[0]['from_player'] not in player_list:
                        param = dict()
                        trade_offer = dict()
                        trade_offer['property_set_offered'] = set()
                        trade_offer['property_set_wanted'] = set()
                        trade_offer['cash_offered'] = 0
                        trade_offer['cash_wanted'] = 0
                        cash_to_be_offered_during_trade += item[0]['price']*1.5
                        if player.current_cash - cash_to_be_offered_during_trade > current_gameboard['go_increment']/2:
                            trade_offer['cash_offered'] = item[0]['price']*1.5  #(0.75*1.5=1.125)
                            trade_offer['property_set_wanted'].add(item[0]['asset'])
                            param['from_player'] = player
                            param['to_player'] = item[0]['from_player']
                            param['offer'] = trade_offer
                            player_list.append(item[0]['from_player'])
                            offer_list_multiple_players.append(param)
                        else:
                            cash_to_be_offered_during_trade -= item[0]['price']*1.5

                if len(offer_list_multiple_players)==0:
                    logger.debug("Wanted to make a trade offer but donot have the money or properties for it, so cannot make one.")
                    return None #No cash to make the trade

    return offer_list_multiple_players


def curate_trade_offer_multiple_players_aggressive(player, potential_offer_list, potential_request_list, current_gameboard, purpose_flag):
    """
    Generates trade offers from the potential offer list, potential request list and the purpose indicated by
    the purpose flag for background_agent_v4.
    Allows a player to make trade offers to MULTIPLE PLAYERS simultaneously.
    :param player: player who wanted to make the trade offer
    :param potential_offer_list: List of potential property offers that can be made to other players because they are lone properties
    for this player and will result in a monopoly to the player to whom it is being offered. These offers are sorted in descending order of the
    amount of cash this player would get if the other player accepts the offer. The first property yields the most cash.
    :param potential_request_list: List of potential properties that this player would like to buy because they are lone properties for the players
    to whom this player makes a trade offer and will lead this player into getting a monopoly. These are sorted in the ascending order of the amount
    this player is willing to pay for the property. The property that costs this player the least is on the top of the list.
    :param current_gameboard: The gameboard data structure
    :purpose_flag: indicates the purpose of making the trade.
    purpose_flag=1 implies that it is purely a make_sell_property_offer because
    the player is urgently in need of cash or is trying to see if it can get to sell one of its lone properties at a high premium because the other
    player can get a monopoly from this offer.
    purpose_flag=2 implies that it can be a buy offer or exchange of properties and cash to increase number of monopolies.
    :return: a list of parameter dictionaries or None. The parameter dictionary, if returned, can be directly sent into
    action_choices.make_sell_property_offer by the calling function. Each parameter dictionary in the list corresponds to the
    trade offers to multiple players
    """
    param = dict()
    trade_offer = dict()
    trade_offer['property_set_offered'] = set()
    trade_offer['property_set_wanted'] = set()
    trade_offer['cash_offered'] = 0
    trade_offer['cash_wanted'] = 0
    offer_list_multiple_players = []

    if purpose_flag == 1:
        #goal - somehow increase current cash since I am almost broke
        #we just made an offer to a player whose property can yield us maximum money and will also enable that player to gain a monopoly at the cost of a premium price.
        if not potential_offer_list:
            return None
        else:
            player_list = []
            offer_list_multiple_players = []
            for item in potential_offer_list:
                if item[0]['to_player'] not in player_list:
                    param = dict()
                    trade_offer = dict()
                    trade_offer['property_set_offered'] = set()
                    trade_offer['property_set_wanted'] = set()
                    trade_offer['cash_offered'] = 0
                    trade_offer['cash_wanted'] = 0

                    trade_offer['property_set_offered'].add(item[0]['asset'])
                    if item[0]['price'] < item[0]['to_player'].current_cash*0.7 and item[0]['to_player'].current_cash*0.3 > current_gameboard['go_increment']:
                        item[0]['price'] = item[0]['to_player'].current_cash*0.7   ## trying to sell for an insame premium as the other player has a lot of cash and maybe ready to
                        #accept the offer for a monopoly
                    elif item[0]['price'] < item[0]['to_player'].current_cash*0.6 and item[0]['to_player'].current_cash*0.4 > current_gameboard['go_increment']:
                        item[0]['price'] = item[0]['to_player'].current_cash*0.6
                    trade_offer['cash_wanted'] = item[0]['price'] ## if above 2 conditions are not satisfied then the price calculation strategy is from the helper functions
                    param['from_player'] = player
                    param['to_player'] = item[0]['to_player']
                    param['offer'] = trade_offer
                    player_list.append(item[0]['to_player'])
                    offer_list_multiple_players.append(param)

    elif purpose_flag == 2:
        #goal - increase monopolies since I have sufficient cash in hand
        #constraint - I will only trade and request 1 property at a time but can simultaneously trade with multiple players.
        if not potential_offer_list and not potential_request_list:
            return None

        elif not potential_offer_list and potential_request_list:
            ##strategy, if the player from whom you are requesting property has very low cash, then he will be ready to give you his property only if the net worth
            #of the offer is positive, ie I should pay greater than market price.
            #but if he has a lot of cash, then even requesting the property for a price lower than market price is worth a try.
            player_list = []
            offer_list_multiple_players = []
            cash_to_be_offered_during_trade = 0
            for item in potential_request_list:
                if item[0]['from_player'] not in player_list:
                    param = dict()
                    trade_offer = dict()
                    trade_offer['property_set_offered'] = set()
                    trade_offer['property_set_wanted'] = set()
                    trade_offer['cash_offered'] = 0
                    trade_offer['cash_wanted'] = 0
                    if item[0]['from_player'].current_cash < current_gameboard['go_increment']:
                        #if player has very low cash, player will accept buy offer only if networth is positive
                        #hence price offered must be > market price
                        item[0]['price'] = item[0]['price']*1.5   ##(1.125*market price)
                    else:
                        item[0]['price'] = item[0]['price']    #0.75*market price

                    cash_to_be_offered_during_trade += item[0]['price']
                    if player.current_cash - cash_to_be_offered_during_trade > current_gameboard['go_increment']/2:
                        trade_offer['cash_offered'] = item[0]['price']
                        trade_offer['property_set_wanted'].add(item[0]['asset'])
                        param['from_player'] = player
                        param['to_player'] = item[0]['from_player']
                        param['offer'] = trade_offer
                        player_list.append(item[0]['from_player'])
                        offer_list_multiple_players.append(param)
                    else:
                        cash_to_be_offered_during_trade -= item[0]['price']

            if len(offer_list_multiple_players)==0:
                logger.debug("Wanted to make a trade offer but donot have the money or properties for it, so cannot make one.")
                return None #No cash to make the trade

        elif not potential_request_list and potential_offer_list:
            #try giving away one lone property for a very high premium which was already calculated in identify_property_trade_offer_to_player function
            player_list = []
            offer_list_multiple_players = []
            for item in potential_offer_list:
                if item[0]['to_player'] not in player_list:
                    param = dict()
                    trade_offer = dict()
                    trade_offer['property_set_offered'] = set()
                    trade_offer['property_set_wanted'] = set()
                    trade_offer['cash_offered'] = 0
                    trade_offer['cash_wanted'] = 0

                    trade_offer['property_set_offered'].add(item[0]['asset'])
                    if item[0]['price'] < item[0]['to_player'].current_cash*0.7 and item[0]['to_player'].current_cash*0.3 > current_gameboard['go_increment']:
                        item[0]['price'] = item[0]['to_player'].current_cash*0.7   ## trying to sell for an insame premium as the other player has a lot of cash and maybe ready to
                        #accept the offer for a monopoly
                    elif item[0]['price'] < item[0]['to_player'].current_cash*0.6 and item[0]['to_player'].current_cash*0.4 > current_gameboard['go_increment']:
                        item[0]['price'] = item[0]['to_player'].current_cash*0.6
                    trade_offer['cash_wanted'] = item[0]['price']
                    param['from_player'] = player
                    param['to_player'] = item[0]['to_player']
                    param['offer'] = trade_offer
                    player_list.append(item[0]['to_player'])
                    offer_list_multiple_players.append(param)

            if len(offer_list_multiple_players)==0:
                logger.debug("Wanted to make a trade offer but donot have the money or properties for it, so cannot make one.")
                return None #No cash to make the trade

        else:
            # want to make a trade offer such that I lose less cash and gain max cash and also get a monopoly
            # potential_request_list is sorted in ascending order of cash offered for the respective requested property
            # potential_offer_list is sorted in descending order of cash received for the respective offered property
            found_player_flag = 0
            saw_players = []
            offer_list_multiple_players = []
            for req in potential_request_list:
                player_2 = req[0]['from_player']
                if player_2 not in saw_players:
                    saw_players.append(player_2)
                    for off in potential_offer_list:
                        if off[0]['to_player'] == player_2:
                            param = dict()
                            trade_offer = dict()
                            trade_offer['property_set_offered'] = set()
                            trade_offer['property_set_wanted'] = set()
                            trade_offer['cash_offered'] = 0
                            trade_offer['cash_wanted'] = 0

                            found_player_flag = 1
                            trade_offer['property_set_offered'].add(off[0]['asset'])
                            trade_offer['cash_wanted'] = off[0]['price']
                            trade_offer['property_set_wanted'].add(req[0]['asset'])
                            trade_offer['cash_offered'] = req[0]['price']
                            param['from_player'] = player
                            param['to_player'] = player_2
                            param['offer'] = trade_offer
                            offer_list_multiple_players.append(param)
                            break

                else:
                    continue

            #if found_player_flag=0, that means we werent able to establish a one on one trade offer, ie I couldnot find a player to whom I could sell my property
            #in return for their property
            if found_player_flag == 0:
                #exchange property not possible --> so we just try to buy property if we have enough cash in order to gain more monopolies
                player_list = []
                offer_list_multiple_players = []
                cash_to_be_offered_during_trade = 0
                for item in potential_request_list:
                    if item[0]['from_player'] not in player_list:
                        param = dict()
                        trade_offer = dict()
                        trade_offer['property_set_offered'] = set()
                        trade_offer['property_set_wanted'] = set()
                        trade_offer['cash_offered'] = 0
                        trade_offer['cash_wanted'] = 0
                        if item[0]['from_player'].current_cash < current_gameboard['go_increment']:
                            #if player has very low cash, player will accept buy offer only if networth is positive
                            #hence price offered must be > market price
                            item[0]['price'] = item[0]['price']*1.5   ##(1.125*market price)
                        else:
                            item[0]['price'] = item[0]['price']    #0.75*market price

                        cash_to_be_offered_during_trade += item[0]['price']
                        if player.current_cash - cash_to_be_offered_during_trade > current_gameboard['go_increment']/2:
                            trade_offer['cash_offered'] = item[0]['price']
                            trade_offer['property_set_wanted'].add(item[0]['asset'])
                            param['from_player'] = player
                            param['to_player'] = item[0]['from_player']
                            param['offer'] = trade_offer
                            player_list.append(item[0]['from_player'])
                            offer_list_multiple_players.append(param)
                        else:
                            cash_to_be_offered_during_trade -= item[0]['price']

                if len(offer_list_multiple_players)==0:
                    logger.debug("Wanted to make a trade offer but donot have the money or properties for it, so cannot make one.")
                    return None #No cash to make the trade

    for param in offer_list_multiple_players:
        if param['offer']['property_set_wanted'] and param['offer']['property_set_offered']:
            if param['offer']['cash_wanted'] - param['offer']['cash_offered'] < param['to_player'].current_cash*0.7 and param['to_player'].current_cash*0.3 > current_gameboard['go_increment']:
                param['offer']['cash_wanted'] = param['to_player'].current_cash*0.7 + param['offer']['cash_offered']
            elif param['offer']['cash_wanted'] - param['offer']['cash_offered'] < param['to_player'].current_cash*0.6 and param['to_player'].current_cash*0.4 > current_gameboard['go_increment']:
                param['offer']['cash_wanted'] = param['to_player'].current_cash*0.6 + param['offer']['cash_offered']
            elif param['offer']['cash_wanted'] - param['offer']['cash_offered'] < param['to_player'].current_cash*0.5 and param['to_player'].current_cash*0.5 > current_gameboard['go_increment']:
                param['offer']['cash_wanted'] = param['to_player'].current_cash*0.5 + param['offer']['cash_offered']

    return offer_list_multiple_players
