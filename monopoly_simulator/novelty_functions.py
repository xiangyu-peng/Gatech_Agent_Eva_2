import logging
import novelty_generator
from log_setting import ini_log_level, set_log_level
logger = set_log_level()


def alternate_contingency_function_1(player, card, current_gameboard):
    """
    This function has the exact signature as calculate_street_repair_cost.

    Note that this function is being provided as an example. There may be other alternate contingency functions that will
    be written in this file with the exact same syntax but with different logic or values. This function may itself
    undergo changes (but the syntax and function it substitutes will not change).
    :return:
    """
    logger.debug('calculating alternative street repair cost for '+ player.player_name)
    cost_per_house = 70
    cost_per_hotel = 145
    cost = player.num_total_houses * cost_per_house + player.num_total_hotels * cost_per_hotel
    player.charge_player(cost)
    # add to game history
    current_gameboard['history']['function'].append(player.charge_player)
    params = dict()
    params['self'] = player
    params['amount'] = cost
    current_gameboard['history']['param'].append(params)
    current_gameboard['history']['return'].append(None)

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
    numberDieNovelty = novelty_generator.NumberClassNovelty()
    numberDieNovelty.die_novelty(current_gameboard, 2, die_state_vector=[[1,2,3,4],[1,2,3,4]])
    classDieNovelty = novelty_generator.TypeClassNovelty()
    # die_state_distribution_vector = ['uniform','uniform','biased','biased']
    die_state_distribution_vector = ['uniform', 'uniform']
    die_type_vector = ['odd_only','even_only']
    classDieNovelty.die_novelty(current_gameboard, die_state_distribution_vector, die_type_vector)
    classCardNovelty = novelty_generator.TypeClassNovelty()
    novel_cc = dict()
    novel_cc["street_repairs"] = "alternate_contingency_function_1"
    novel_chance = dict()
    novel_chance["general_repairs"] = "alternate_contingency_function_1"
    classCardNovelty.card_novelty(current_gameboard, novel_cc, novel_chance)


    '''
    #Level 2 Novelty
    #The below combination reassigns property groups and individual properties to different colors.
    #On playing the game it is verified that the newly added property to the color group is taken into account for monopolizing a color group,
    # i,e the orchid color group now has Baltic Avenue besides St. Charles Place, States Avenue and Virginia Avenue. The player acquires a monopoly
    # only on the ownership of all the 4 properties in this case.
    inanimateNovelty = novelty_generator.InanimateAttributeNovelty()
    inanimateNovelty.map_property_set_to_color(current_gameboard, [current_gameboard['location_objects']['Park Place'], current_gameboard['location_objects']['Boardwalk']], 'Brown')
    inanimateNovelty.map_property_to_color(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 'Orchid')
    #setting new rents for Indiana Avenue
    inanimateNovelty.rent_novelty(current_gameboard['location_objects']['Indiana Avenue'], {'rent': 50, 'rent_1_house': 150})
    '''

    '''
    #Level 3 Novelty
    granularityNovelty = novelty_generator.GranularityRepresentationNovelty()
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 6)
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['States Avenue'], 20)
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Tennessee Avenue'], 27)
    spatialNovelty = novelty_generator.SpatialRepresentationNovelty()
    spatialNovelty.color_reordering(current_gameboard, ['Boardwalk', 'Park Place'], 'Blue')
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Park Place'], 52)
    '''