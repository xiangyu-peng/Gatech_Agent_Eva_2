"""
This file is edited for better evaluation.
We assign our agent to  player_1.
Changes are as follows,
1). Add relative path
2). Call our agent only once a tournament
"""
###GT - Add relative path#####
import sys, os
upper_path = os.path.abspath('..')
upper_path_2level = upper_path.replace('/Evaluation/GNOME-p3','')
sys.path.append(upper_path)
sys.path.append(upper_path_2level)
from A2C_agent.server_agent import ServerAgent
##############################

import numpy as np
from monopoly_simulator import gameplay_v2
from monopoly_simulator import novelty_generator
from monopoly_simulator import location
from monopoly_simulator.logging_info import log_file_create
import os
import shutil
import json


def play_tournament_without_novelty(tournament_log_folder=None, meta_seed=5, num_games=100):
    """
    Tournament logging is not currently supported, but will be soon.
    :param tournament_log_folder: String. The path to a folder.
    :param meta_seed: This is the seed we will use to generate a sequence of seeds, that will (in turn) spawn the games in gameplay/simulate_game_instance
    :param num_games: The number of games to simulate in a tournament
    :return: None. Will print out the win-loss metrics, and will write out game logs
    """

    if not tournament_log_folder:
        print("No logging folder specified, cannot log tournaments. Provide a logging folder path.")
        raise Exception

    np.random.seed(meta_seed)
    big_list = list(range(0,1000000))
    np.random.shuffle(big_list)
    tournament_seeds = big_list[0:num_games]
    winners = list()
    count = 1

    folder_name = "../tournament_logs" + tournament_log_folder
    try:
        os.makedirs(folder_name)
        print('Logging gameplay')
    except:
        print('Given logging folder already exists. Clearing folder before logging new files.')
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)

    metadata_dict = {
        "function": "play_tournament_without_novelty",
        "parameters": {
            "meta_seed": meta_seed,
            "num_game": num_games
        }
    }

    json_filename = folder_name + "tournament_meta_data.json"
    out_file = open(json_filename, "w")
    json.dump(metadata_dict, out_file, indent=4)
    out_file.close()
    #####GT - Call agent here#####
    agent = ServerAgent()
    f_name = "meta_seed_" + str(meta_seed) + '_without_novelty'
    agent.start_tournament(f_name)
    ##############################
    for t in tournament_seeds:
        print('Logging gameplay for seed: ', str(t), ' ---> Game ' + str(count))
        filename = folder_name + "meta_seed_" + str(meta_seed) + '_num_games_' + str(count) + '.log'
        logger = log_file_create(filename)
        #####GT - add agent to parameter#####
        winners.append(gameplay_v2.play_game_in_tournament(t,agent=agent))
        #####################################
        handlers_copy = logger.handlers[:]
        for handler in handlers_copy:
            logger.removeHandler(handler)
            handler.close()
            handler.flush()
        count += 1

    print(winners)
    wins = 0
    for i in winners:
        if i == 'player_1':
            wins += 1

    print(wins)
    agent.end_tournament()


def play_tournament_with_novelty_1(tournament_log_folder=None, meta_seed=5, num_games=100, novelty_index=23):
    """

    :param tournament_log_folder:
    :param meta_seed:
    :param num_games:
    :param novelty_index: an integer between 1 and num_games-1. We will play this many games BEFORE introducing novelty.
    :return:
    """

    if not tournament_log_folder:
        print("No logging folder specified, cannot log tournaments. Provide a logging folder path.")
        raise Exception

    np.random.seed(meta_seed)
    big_list = list(range(0, 1000000))
    np.random.shuffle(big_list)
    tournament_seeds = big_list[0:num_games]
    winners = list()
    count = 1

    folder_name = "../tournament_logs" + tournament_log_folder
    try:
        os.makedirs(folder_name)
        print('Logging gameplay')
    except:
        print('Given logging folder already exists. Clearing folder before logging new files.')
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)

    metadata_dict = {
        "function": "play_tournament_with_novelty_1",
        "parameters": {
            "meta_seed": meta_seed,
            "novelty_index": novelty_index,
            "num_game": num_games
        }
    }

    json_filename = folder_name + "tournament_meta_data.json"
    out_file = open(json_filename, "w")
    json.dump(metadata_dict, out_file, indent=4)
    out_file.close()
    #####GT - Call agent here#####
    agent = ServerAgent()
    f_name = "meta_seed_" + str(meta_seed) + '_with_novelty'
    agent.start_tournament(f_name)
    ##############################
    for t in range(0,novelty_index):
        print('Logging gameplay without novelty for seed: ', str(t), ' ---> Game ' + str(count))
        filename = folder_name + "meta_seed_" + str(meta_seed) + '_without_novelty' + '_num_games_' + str(count) + '.log'
        logger = log_file_create(filename)
        #####GT - add agent to parameter#####
        winners.append(gameplay_v2.play_game_in_tournament(tournament_seeds[t],agent=agent))
        #####################################
        handlers_copy = logger.handlers[:]
        for handler in handlers_copy:
            logger.removeHandler(handler)
            handler.close()
            handler.flush()
        count += 1

    new_winners = list()
    for t in range(novelty_index, len(tournament_seeds)):
        print('Logging gameplay with novelty for seed: ', str(t), ' ---> Game ' + str(count))
        filename = folder_name + "meta_seed_" + str(meta_seed) + '_with_novelty' + '_num_games_' + str(count) + '.log'
        logger = log_file_create(filename)
        #####GT - add agent to parameter#####
        new_winners.append(gameplay_v2.play_game_in_tournament(tournament_seeds[t], class_novelty_1,agent=agent))
        #####################################
        handlers_copy = logger.handlers[:]
        for handler in handlers_copy:
            logger.removeHandler(handler)
            handler.close()
            handler.flush()
        count += 1

    print('pre_novelty winners', winners)
    print('post_novelty_winners', new_winners)

    # GT -Close the connection #
    agent.end_tournament()



def class_novelty_1(current_gameboard):
    # Level 1 Novelty

    # Dice
    # numberDieNovelty = novelty_generator.NumberClassNovelty()
    # numberDieNovelty.die_novelty(current_gameboard, 2, die_state_vector=[[1,2,3,4,5,6],[1,2,3,4,5,6]])
    # classDieNovelty = novelty_generator.TypeClassNovelty()
    # # die_state_distribution_vector = ['uniform','uniform','biased','biased']
    # die_state_distribution_vector = ['biased', 'uniform']
    # die_type_vector = ['odd_only','even_only']
    # classDieNovelty.die_novelty(current_gameboard, die_state_distribution_vector, die_type_vector)

    # Card

    # Card -num
    # numberCardNovelty = novelty_generator.NumberClassNovelty()
    # community_chest_cards_num = {"go_to_jail":1}
    # chance_cards_num = {"go_to_jail":1}
    # numberCardNovelty.card_novelty(current_gameboard, community_chest_cards_num, chance_cards_num)

    # Card - destination
    # desCardNovelty = novelty_generator.InanimateAttributeNovelty()
    # community_chest_card_destinations, chance_card_destinations = dict(), dict()
    # community_chest_card_destinations['advance_to_go'] = location.ActionLocation("action", 'chance', 36, 37, "None", "pick_card_from_chance")
    # desCardNovelty.card_destination_novelty(current_gameboard, community_chest_card_destinations, chance_card_destinations)

    # Card - Amount
    # cardamountNovelty = novelty_generator.InanimateAttributeNovelty()
    # community_chest_card_amounts = dict()
    # key = "sale_of_stock"
    # community_chest_card_amounts[key] = 60
    # cardamountNovelty.card_amount_novelty(current_gameboard, community_chest_card_amounts=community_chest_card_amounts)

    # Type - Card
    # cardtypeNovelty = novelty_generator.TypeClassNovelty()
    # community_chest_cards_contingency = dict()
    # community_chest_cards_contingency["street_repairs"] = "alternate_contingency_function_1"
    # cardtypeNovelty.card_novelty(current_gameboard,
    #                              community_chest_cards_contingency=community_chest_cards_contingency,
    #                              chance_cards_contingency=dict())

    # Inanimate

    # Tax
    taxNovelty = novelty_generator.InanimateAttributeNovelty()
    tax_location = current_gameboard['location_objects']["Luxury Tax"]
    # tax_location = location.TaxLocation("tax", "Luxury Tax", 38, 39, "None", 100)
    taxNovelty.tax_novelty(tax_location, 200)

    # Mortgage
    morNovelty = novelty_generator.InanimateAttributeNovelty()
    mor_location = current_gameboard['location_objects']["Mediterranean Avenue"]
    morNovelty.mortgage_novelty(mor_location, 40)

    # Contingent
    # bank - percentage
    conNovelty = novelty_generator.ContingentAttributeNovelty()
    conNovelty.change_mortgage_percentage(current_gameboard, 0.2)

    # Granularity
    # granularityNovelty = novelty_generator.GranularityRepresentationNovelty()
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 6)
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['States Avenue'], 20)
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Tennessee Avenue'], 27)
    # spatialNovelty = novelty_generator.SpatialRepresentationNovelty()
    # spatialNovelty.color_reordering(current_gameboard, ['Boardwalk', 'Park Place'], 'Blue')
    # granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Park Place'], 52)




#All the tournaments get logged in seperate folders inside ../tournament_logs folder
try:
    os.makedirs("../tournament_logs/")
except:
    pass

#Specify the name of the folder in which the tournament games has to be logged in the following format: "/name_of_your_folder/"
# play_tournament_without_novelty('/tournament_without_novelty_4/', meta_seed=1, num_games=100)

play_tournament_with_novelty_1('/tournament_with_novelty/', meta_seed=1, num_games=12, novelty_index=5)
