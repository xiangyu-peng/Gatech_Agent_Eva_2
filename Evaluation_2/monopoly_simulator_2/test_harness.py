import numpy as np
import gameplay_socket
import novelty_generator
from server_agent_serial import ServerAgent
import location
from logging_info import log_file_create
import os
import shutil
import json


def play_tournament_without_novelty(tournament_log_folder=None, meta_seed=5, num_games=100):
    """
    Tournament logging is not currently supported, but will be soon.
    :param tournament_log_folder: String. The path to a folder.
    :param meta_seed: This is the seed we will use to generate a sequence of seeds, that will (in turn) spawn the games in gameplay_socket/simulate_game_instance
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
    # agent=None
    agent = ServerAgent()
    f_name = "meta_seed_" + str(meta_seed) + '_without_novelty'
    agent.start_tournament(f_name)
    ##############################
    for t in tournament_seeds:
        print('Logging gameplay for seed: ', str(t), ' ---> Game ' + str(count))
        filename = folder_name + "meta_seed_" + str(meta_seed) + '_num_games_' + str(count) + '.log'
        logger = log_file_create(filename)
        #####GT - add agent to parameter#####
        winners.append(gameplay_socket.play_game_in_tournament(t, agent=agent))
        #####################################
        # winners.append(gameplay_socket.play_game_in_tournament(t))
        handlers_copy = logger.handlers[:]
        for handler in handlers_copy:
            logger.removeHandler(handler)
            handler.close()
            handler.flush()
        count += 1

    print(winners)
    win_rate = 0
    for winner in winners:
        if 'player_1' == winner:
            win_rate += 1
    print('win rate is ', float(win_rate / num_games))

    agent.end_tournament()


def play_tournament_with_novelty_1(tournament_log_folder=None, meta_seed=0, num_games=100, novelty_index=23):
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
    agent = ServerAgent()
    f_name = "meta_seed_" + str(meta_seed) + '_with_novelty'
    agent.start_tournament(f_name)
    ##############################

    for t in range(0,novelty_index):
        print('Logging gameplay without novelty for seed: ', str(t), ' ---> Game ' + str(count))
        filename = folder_name + "meta_seed_" + str(meta_seed) + '_without_novelty' + '_num_games_' + str(count) + '.log'
        logger = log_file_create(filename)
        #####GT - add agent to parameter#####
        winners.append(gameplay_socket.play_game_in_tournament(tournament_seeds[t],agent=agent))
        #####################################
        # winners.append(gameplay_socket.play_game_in_tournament(tournament_seeds[t]))
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
        # new_winners.append(gameplay_socket.play_game_in_tournament(tournament_seeds[t], class_novelty_1))
        #####GT - add agent to parameter#####
        new_winners.append(gameplay_socket.play_game_in_tournament(tournament_seeds[t], class_novelty_1, agent=agent))
        #####################################
        handlers_copy = logger.handlers[:]
        for handler in handlers_copy:
            logger.removeHandler(handler)
            handler.close()
            handler.flush()
        count += 1

    print('pre_novelty winners', winners)
    print('post_novelty_winners', new_winners)
    agent.end_tournament()


def class_novelty_1(current_gameboard):


    #Mortgage
    # morNovelty = novelty_generator.InanimateAttributeNovelty()
    # mor_location = current_gameboard['location_objects']["Mediterranean Avenue"]
    # morNovelty.mortgage_novelty(mor_location, 40)

    # classCardNovelty = novelty_generator.TypeClassNovelty()
    # novel_cc = dict()
    # novel_cc["street_repairs"] = "alternate_contingency_function_1"
    # novel_chance = dict()
    # novel_chance["general_repairs"] = "alternate_contingency_function_1"
    # classCardNovelty.card_novelty(current_gameboard, novel_cc, novel_chance)

    # price
    inanimateNovelty = novelty_generator.InanimateAttributeNovelty()
    asset_lists = ["Mediterranean Avenue", "Baltic Avenue", "Reading Railroad", "Oriental Avenue", "Vermont Avenue",
                   "Connecticut Avenue", "St. Charles Place", "Electric Company", "States Avenue",
                   "Virginia Avenue", "Pennsylvania Railroad", "St. James Place", "Tennessee Avenue",
                   "New York Avenue", "Kentucky Avenue", "Indiana Avenue", "Illinois Avenue", "B&O Railroad",
                   "Atlantic Avenue", "Ventnor Avenue", "Water Works", "Marvin Gardens", "Pacific Avenue",
                   "North Carolina Avenue", "Pennsylvania Avenue", "Short Line", "Park Place", "Boardwalk"]
    num = 0
    for asset in asset_lists:
        num += 1
        if num >= 0 and num < 10:
            inanimateNovelty.price_novelty(current_gameboard['location_objects'][asset], 1499)

    # Dice
    # numberDieNovelty = novelty_generator.NumberClassNovelty()
    # numberDieNovelty.die_novelty(current_gameboard, 2, die_state_vector=[[1,2,3,4,5,6],[1,2,3,4,5,6]])
    # classDieNovelty = novelty_generator.TypeClassNovelty()
    # # die_state_distribution_vector = ['uniform','uniform','biased','biased']
    # die_state_distribution_vector = ['biased', 'uniform']
    # die_type_vector = ['odd_only','even_only']
    # classDieNovelty.die_novelty(current_gameboard, die_state_distribution_vector, die_type_vector)

#All the tournaments get logged in seperate folders inside ../tournament_logs folder
try:
    os.makedirs("../tournament_logs/")
except:
    pass

#Specify the name of the folder in which the tournament games has to be logged in the following format: "/name_of_your_folder/"
# play_tournament_without_novelty('/tournament_without_novelty_4/', meta_seed=10, num_games=100)
play_tournament_with_novelty_1('/tournament_with_novelty/',num_games=100, novelty_index=5, meta_seed=0)
