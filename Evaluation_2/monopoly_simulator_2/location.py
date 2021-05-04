from flag_config import flag_config_dict
from bank import Bank
import logging
logger = logging.getLogger('monopoly_simulator.logging_info.location')

class Location(object):

    def __init__(self, loc_class, name, start_position, end_position, color):
        """
         Super-class that all locations on the board will be sub-classed to, and that has the common attributes.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        """
        self.loc_class = loc_class
        self.name = name
        self.start_position = start_position
        self.end_position = end_position
        if color == 'None':
            self.color = None
        else:
            self.color = color

        self.calculate_mortgage_owed = Bank.calculate_mortgage_owed

    def transfer_property_to_bank(self, player, current_gameboard):
        """
        This function is called when the player is selling the property back to the bank. If the property is mortgaged
        then we deduct the mortgage-freeing cost from the cash that the player would have received if the property had
        not been mortgaged.
        :param player: Player instance. The property will be taken from this player and transferred to the bank
        :param current_gameboard: A dict. The global gameboard data structure
        :return: An integer. Specifies the amount due to the player for selling this property to the bank
        """

        cash_due = self.price * current_gameboard['bank'].property_sell_percentage   # changed hardcoded value to a bank parameter
        cash_owed = 0
        if self.loc_class == 'real_estate' and (self.num_houses > 0 or self.num_hotels > 0):
            logger.debug('Bank error!'+ self.name+' being sold has improvements on it. Raising Exception')
            logger.error("Exception")
            raise Exception

        if self.is_mortgaged:
            cash_owed = self.calculate_mortgage_owed(self, current_gameboard)

        if cash_due >= cash_owed:
            if current_gameboard['bank'].total_cash_with_bank < cash_due - cash_owed:    # i.e. bank does not have enough money to pay the player what is due
                logger.debug("Bank has insufficient funds!!!  Rejected Transaction!!")
                return flag_config_dict['failure_code']
            else:
                player.remove_asset(self)
                if self.is_mortgaged:
                    self.is_mortgaged = False
                # add to game history
                current_gameboard['history']['function'].append(player.remove_asset)
                params = dict()
                params['self'] = player
                params['asset'] = self
                current_gameboard['history']['param'].append(params)
                current_gameboard['history']['return'].append(None)
                self.owned_by = current_gameboard['bank']
                return cash_due - cash_owed

        else:
            player.remove_asset(self)
            if self.is_mortgaged:
                self.is_mortgaged = False
            # add to game history
            current_gameboard['history']['function'].append(player.remove_asset)
            params = dict()
            params['self'] = player
            params['asset'] = self
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(None)
            self.owned_by = current_gameboard['bank']
            return 0 # foreclosure.


    def transfer_property_between_players(self, from_player, to_player, current_gameboard):
        """
        Remove property from possession of from_player and transfer to to_player. Note that there is no cash transfer
        happening here; any such cash transfer must be done outside the function.
        :param from_player: Player instance.
        :param to_player: Player instance.
        :param current_gameboard: A dict. The global gameboard data structure
        :return: None
        """
        # from_player.remove_asset(self)
        # # add to game history
        # current_gameboard['history']['function'].append(from_player.remove_asset)
        # params = dict()
        # params['self'] = from_player
        # params['asset'] = self
        # current_gameboard['history']['param'].append(params)
        # current_gameboard['history']['return'].append(None)

        self.update_asset_owner(to_player, current_gameboard)
        # add to game history
        current_gameboard['history']['function'].append(self.update_asset_owner)
        params = dict()
        params['self'] = self
        params['player'] = to_player
        params['current_gameboard'] = current_gameboard
        current_gameboard['history']['param'].append(params)
        current_gameboard['history']['return'].append(None)

    def update_asset_owner(self, player, current_gameboard):
        """
        If the asset is non-purchaseable, we will raise an exception. A more elegant way (we'll make this change
        in a close future edition) is to have a PurchaseableLocation class sitting between the purchaseable sub-classes
        like real estate and Location, and to add update_asset_owner as a method of PurchaseableLocation.
        Note that we remove the asset from the previous owner's portfolio if it is not owned by the bank.
        :param player: Player instance. The player who now owns this asset (self)
        :param current_gameboard: A dict. The global gameboard data structure
        :return: None
        """
        logger.debug('attempting to update asset '+ self.name+ ' to reflect new owner: '+ player.player_name)
        if self.loc_class == 'real_estate' or self.loc_class == 'railroad' or self.loc_class == 'utility':
            if self.owned_by == player:
                logger.debug(player.player_name+' already owns this asset! Raising exception...')
                logger.error("Exception")
                raise Exception
            elif type(self.owned_by) != Bank: # not owned by this player or by the bank.
                logger.debug('Asset is owned by '+self.owned_by.player_name+'. Attempting to remove...')
                self.owned_by.remove_asset(self)
                # add to game history
                current_gameboard['history']['function'].append(self.owned_by.remove_asset)
                params = dict()
                params['self'] = self.owned_by
                params['asset'] = self
                current_gameboard['history']['param'].append(params)
                current_gameboard['history']['return'].append(None)

                self.owned_by = current_gameboard['bank'] # this is temporary, but we want to enforce safe behavior

            self.owned_by = player
            player.add_asset(self, current_gameboard) # if the property is mortgaged, this will get reflected in the new owner's portfolio
            # add to game history
            current_gameboard['history']['function'].append(player.add_asset)
            params = dict()
            params['self'] = player
            params['asset'] = self
            params['current_gameboard'] = current_gameboard
            current_gameboard['history']['param'].append(params)
            current_gameboard['history']['return'].append(None)

            logger.debug('Asset ownership update succeeded.')
        else:
            logger.debug('Asset ',self.name+' is non-purchaseable!')
            logger.error("Exception")
            raise Exception


class DoNothingLocation(Location):
    def __init__(self, loc_class, name, start_position, end_position, color):
        """
        This is a location (such as free parking) where nothing happens. It has loc_class 'do_nothing' in the game
        schema. The attributes are the same as in the schema.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        """
        super().__init__(loc_class, name, start_position, end_position, color)

    def serialize(self):
        loc_dict = dict()
        loc_dict['loc_class'] = self.loc_class
        loc_dict['name'] = self.name
        loc_dict['start_position'] = self.start_position
        loc_dict['end_position'] = self.end_position
        loc_dict['color'] = self.color
        return loc_dict


class ActionLocation(Location):
    def __init__(self, loc_class, name, start_position, end_position, color, perform_action):
        """
        This is a location that is associated with a non tax-paying action such as
        picking a card from community chest or chance. It has loc_class 'action' in the game
        schema. The attributes are the same as in the schema.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        :param perform_action: A function from card_utility_actions. This is the action that will be performed when
        the player lands on this location.
        """
        super().__init__(loc_class, name, start_position, end_position, color)
        self.perform_action = perform_action

    def serialize(self):
        loc_dict = dict()
        loc_dict['loc_class'] = self.loc_class
        loc_dict['name'] = self.name
        loc_dict['start_position'] = self.start_position
        loc_dict['end_position'] = self.end_position
        loc_dict['color'] = self.color
        loc_dict['perform_actiom'] = self.perform_action.__name__
        return loc_dict


class RealEstateLocation(Location):

    def __init__(self, loc_class, name, start_position, end_position, color, rent_1_house, rent_hotel,
                 price, rent_3_houses, rent, mortgage, price_per_house, rent_4_houses, rent_2_houses, owned_by,
                 num_houses, num_hotels):

        """
        This is a real estate location. It has loc_class 'real_estate' in the game
        schema. The attributes are the same as in the schema.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        :param rent_1_house: An integer. The rent that must be paid if there is one house on the property.
        :param rent_hotel: An integer. The rent that must be paid if there is a hotel on the property (currently, at most 1 hotel is allowed/property).
        :param price: An integer. The purchase price of the property if the bank is the owner.
        :param rent_3_houses: An integer. The rent that must be paid if there are three houses on the property.
        :param rent: An integer. The rent that must be paid if the property is unimproved (no houses or hotels)
        :param mortgage: An integer. The amount that you can mortgage the property for.
        :param price_per_house: An integer. The cost of setting up a house on the property.
        :param rent_4_houses: An integer. The rent that must be paid if there are four houses on the property.
        :param rent_2_houses: An integer. The rent that must be paid if there are two houses on the property.
        :param owned_by: An instance of Player or Bank. Specifies who owns the property
        :param num_houses: An integer. Number of houses currently set up on the property.
        :param num_hotels: An integer. Number of hotels currently set up on the property.
        """
        super().__init__(loc_class, name, start_position, end_position, color)
        self.rent_1_house = float(rent_1_house)
        self.rent_2_houses = float(rent_2_houses)
        self.rent_3_houses = float(rent_3_houses)
        self.rent_4_houses = float(rent_4_houses)
        self.rent_hotel = float(rent_hotel)
        self.rent = float(rent)
        self.price = float(price)
        self.price_per_house = float(price_per_house)
        self.mortgage = float(mortgage)
        self.owned_by = owned_by
        self.num_houses = num_houses
        self.num_hotels = num_hotels
        self.is_mortgaged = False

        obj = dict()
        obj[1] = self.rent_1_house
        obj[2] = self.rent_2_houses
        obj[3] = self.rent_3_houses
        obj[4] = self.rent_4_houses
        self._house_rent_dict = obj

    @staticmethod
    def calculate_rent(asset, current_gameboard):
        """
        When calculating the rent, note that a real estate can either have a hotel OR houses OR be
        unimproved-monopolized OR be unimproved-non-monopolized. Rent is calculated based on which of these
        situations applies.
        :param asset: rent of this asset is to be calculated
        :param current_gameboard: The global gameboard data structure
        :return: An integer. The rent due.
        """
        logger.debug('calculating rent for '+asset.name)
        ans = asset.rent # unimproved-non-monopolized rent (the default)
        if asset.num_hotels == 1:
            logger.debug('property has a hotel. Updating rent.')
            ans = asset.rent_hotel
        elif asset.num_houses > 0: # later we can replace these with reflections
            logger.debug('property has '+str(asset.num_houses)+' houses. Updating rent.')
            ans = asset._house_rent_dict[asset.num_houses] # if for some reason you have more than 4 houses, you'll get a key error
        elif asset.color in asset.owned_by.full_color_sets_possessed:
            ans = asset.rent*current_gameboard['bank'].monopolized_property_rent_factor # charge twice the rent on unimproved monopolized properties.
            logger.debug('property has color '+ asset.color+ ' which is monopolized by '+asset.owned_by.player_name+'. Updating rent.')
        logger.debug('rent is calculated to be '+str(ans))
        return ans

    def serialize(self):
        loc_dict = dict()
        loc_dict['loc_class'] = self.loc_class
        loc_dict['name'] = self.name
        loc_dict['start_position'] = self.start_position
        loc_dict['end_position'] = self.end_position
        loc_dict['color'] = self.color
        loc_dict['rent_1_house'] = self.rent_1_house
        loc_dict['rent_2_houses'] = self.rent_2_houses
        loc_dict['rent_3_houses'] = self.rent_3_houses
        loc_dict['rent_4_houses'] = self.rent_4_houses
        loc_dict['rent_hotel'] = self.rent_hotel
        loc_dict['rent'] = self.rent
        loc_dict['price'] = self.price
        loc_dict['price_per_house'] = self.price_per_house
        loc_dict['mortgage'] = self.mortgage
        if isinstance(self.owned_by, Bank):
            loc_dict['owned_by'] = 'bank'
        else:
            loc_dict['owned_by'] = self.owned_by.player_name
        loc_dict['num_houses'] = self.num_houses
        loc_dict['num_hotels'] = self.num_hotels
        loc_dict['is_mortgaged'] = self.is_mortgaged
        loc_dict['house_rent_dict'] = self._house_rent_dict
        return loc_dict


class TaxLocation(Location):
    def __init__(self, loc_class, name, start_position, end_position, color, amount_due):
        """
        This is a tax (luxury or income) location. It has loc_class 'tax' in the game
        schema. The attributes are the same as in the schema.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        :param amount_due: An integer. The amount of tax that is due when the player is at this location.
        """
        super().__init__(loc_class, name, start_position, end_position, color)
        self.amount_due = float(amount_due)

    def serialize(self):
        loc_dict = dict()
        loc_dict['loc_class'] = self.loc_class
        loc_dict['name'] = self.name
        loc_dict['start_position'] = self.start_position
        loc_dict['end_position'] = self.end_position
        loc_dict['color'] = self.color
        loc_dict['amount_due'] = self.amount_due
        return loc_dict

    @staticmethod
    def calculate_tax(location, player, current_gameboard):
        return location.amount_due

class RailroadLocation(Location):
    def __init__(self, loc_class, name, start_position, end_position, color, price, mortgage, owned_by):
        """
        This is a railroad location. It has loc_class 'railroad' in the game
        schema. The attributes are the same as in the schema.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        :param price: An integer. The purchase price of the property if the bank is the owner.
        :param mortgage: An integer. The amount that you can mortgage the property for.
        :param owned_by: An instance of Player or Bank. Specifies who owns the property
        """
        super().__init__(loc_class, name, start_position, end_position, color)
        self.price = float(price)
        self.mortgage = float(mortgage)
        self.owned_by = owned_by
        self.is_mortgaged = False

        obj = dict()
        obj[1] = 25
        obj[2] = 50
        obj[3] = 100
        obj[4] = 200
        self._railroad_dues = obj

    @staticmethod
    def calculate_railroad_dues(asset, current_gameboard):
        """
        Compute dues if a player lands on railroad owned by another player.
        :return: An integer. Specifies railroad dues
        """
        logger.debug('calculating railroad dues for '+asset.name)
        if asset.owned_by.num_railroads_possessed > 4 or asset.owned_by.num_railroads_possessed < 0:
            logger.debug('Error! num railroads possessed by '+ asset.owned_by.player_name+ ' is '+ \
                str(asset.owned_by.num_railroads_possessed)+', which is impossible')

            logger.error("Exception")
            raise Exception
        dues = asset._railroad_dues[asset.owned_by.num_railroads_possessed]

        logger.debug('railroad dues are '+str(dues))
        return dues

    def serialize(self):
        loc_dict = dict()
        loc_dict['loc_class'] = self.loc_class
        loc_dict['name'] = self.name
        loc_dict['start_position'] = self.start_position
        loc_dict['end_position'] = self.end_position
        loc_dict['color'] = self.color
        loc_dict['price'] = self.price
        loc_dict['mortgage'] = self.mortgage
        if isinstance(self.owned_by, Bank):
            loc_dict['owned_by'] = 'bank'
        else:
            loc_dict['owned_by'] = self.owned_by.player_name
        loc_dict['is_mortgaged'] = self.is_mortgaged
        loc_dict['railroad_dues'] = self._railroad_dues
        return loc_dict


class UtilityLocation(Location):
    def __init__(self, loc_class, name, start_position, end_position, color, price, mortgage, owned_by):
        """
        This is a utility location. It has loc_class 'utility' in the game
        schema. The attributes are the same as in the schema.
        :param loc_class: A string. The location class/type as specified in the schema.
        :param name: A string. The name of the location
        :param start_position: An integer. Specifies (inclusively) the index on the location_sequence of the current
        gameboard where this location begins.
        :param end_position: An integer. Specifies (non-inclusively) the index on the location_sequence of the current
        gameboard where this location ends. In the default board, it is always start_position+1
        :param color: A string or None. If string, it specifies the color of the location.
        :param price: An integer. The purchase price of the property if the bank is the owner.
        :param mortgage: An integer. The amount that you can mortgage the property for.
        :param owned_by: An instance of Player or Bank. Specifies who owns the property
        """
        super().__init__(loc_class, name, start_position, end_position, color)
        self.price = float(price)
        self.mortgage = float(mortgage)
        self.owned_by = owned_by
        self.is_mortgaged = False

        obj = dict()
        obj[1] = 4
        obj[2] = 10
        self._die_multiples = obj

    @staticmethod
    def calculate_utility_dues(asset, current_gameboard, die_total):
        """
        Compute dues if a player lands on utility owned by another player.
        :param die_total: An integer. The dice total (if there's more than 1 dice as there is in the default game)
        :return: An integer. Specifies utility dues.
        """
        logger.debug('calculating utility dues for '+ asset.name)
        if asset.owned_by.num_utilities_possessed > 2 or asset.owned_by.num_utilities_possessed < 0:
                logger.debug('Error! num utilities possessed by '+asset.owned_by.player_name+' is '+ \
                    str(asset.owned_by.num_utilities_possessed)+ ', which is impossible')

                logger.error("Exception")
                raise Exception

        dues = die_total*asset._die_multiples[asset.owned_by.num_utilities_possessed]
        logger.debug('utility dues are '+ str(dues))
        return dues

    def serialize(self):
        loc_dict = dict()
        loc_dict['loc_class'] = self.loc_class
        loc_dict['name'] = self.name
        loc_dict['start_position'] = self.start_position
        loc_dict['end_position'] = self.end_position
        loc_dict['color'] = self.color
        loc_dict['price'] = self.price
        loc_dict['mortgage'] = self.mortgage
        if isinstance(self.owned_by, Bank):
            loc_dict['owned_by'] = 'bank'
        else:
            loc_dict['owned_by'] = self.owned_by.player_name
        loc_dict['is_mortgaged'] = self.is_mortgaged
        loc_dict['die_multiples'] = self._die_multiples
        return loc_dict
