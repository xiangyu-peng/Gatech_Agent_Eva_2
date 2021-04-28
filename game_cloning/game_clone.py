import os
import json
import path
import numpy as np
import torch

class GameClone():
    def __init__(self, rules_manual_json=None):
        """
        Sets up the game cloning rules, where the main structure
        Args:
            rules_manual: json of rules
        """
        self.rules = {}
        self.rules_manual_json = rules_manual_json
        self.red_button = False
        if rules_manual_json:
            self._process_manual()

        # a record is a sequence that grows over time
        # a sequence is a set that is ordered
        # a set is a group of like objects
        self.gamestate_key_types = {
            'record': ['die_sequence', 'history', ],
            'sequence': ['location_sequence'],
            'set': ['locations'],
            'not_monitored': ['players']
        }

    def predict_next(self, state, action):
        relevant_rules = []
        for rule_key, rule_value in self.rules.items():
            if state.issuperset(rule_value[0]):
                relevant_rules.append(rule_key)

        # Constraint search
        for rules in relevant_rules:
            predicted_state = None
            ######## TODO
        return predicted_state

    def k_forward(self, state_action_history):
        for state, action in state_action_history:
            self.predict_next(state, action)

    def set_rules_from_json(self, rules_manual_json):
        self.rules_manual_json = rules_manual_json
        self._process_manual()

    def output_rules(self, rules_output_file='game_clone_rules.json', overwrite=False):
        if self.rules:
            if not overwrite:
                counter = 0
                if os.path.isfile(os.path.abspath(rules_output_file)):
                    rules_output_file = rules_output_file + str(counter)
                while os.path.isfile(os.path.abspath(rules_output_file)):
                    counter += 1
                    rules_output_file = rules_output_file - str(counter - 1) + str(counter)

            with open(rules_output_file, 'w') as f:
                self.rules = json.load(self.rules_manual_json)
            return rules_output_file
        else:
            return ''

    def _process_manual(self):
        with open(self.rules_manual_json, 'r') as f:
            self.rules = json.load(self.rules_manual_json)

    def update(self, samples):
        raise NotImplementedError

    def gc_detect_novelty(self, gameboard_message):
        data_dict_from_server = json.loads(gameboard_message)

        novelty_properties = {}

        ## TODO balloch: dummy
        if not self.red_button and np.random.rand() > 0.99:
            self.red_button = True
            novelty_properties['type'] = 'dummy'
            novelty_properties['trigger'] = 'random'

        return self.red_button, novelty_properties


    def update_novelty_detector(self, state_stats, gameboard):
        # tree_search
        for key in gameboard:
            if (not gameboard[key]) or (key in ('cards', 'history', 'players')):
                continue
            elif type(gameboard[key]) is dict:
                if key not in state_stats:
                    state_stats[key] = {}
                state_stats[key] = self.update_novelty_detector(state_stats[key],
                                                           gameboard[key])

            elif type(gameboard[key]) is list:
                if type(gameboard[key][-1]) is dict:  # history # TODO balloch: untested
                    if key not in state_stats:
                        state_stats[key] = []
                    for idx, subdict in enumerate(gameboard[key]):
                        if idx > (len(state_stats[key]) - 1):
                            state_stats[key].append(self.update_novelty_detector({}, subdict))
                        else:  # TODO: if subtree the same, add count, if not, insert? frankly list should become a tree
                            state_stats[key][idx] = self.update_novelty_detector(state_stats[key][idx],
                                                                            gameboard[key][idx])
                elif type(gameboard[key][-1]) is list:  # hardcoding this because die sequence only list of lists
                    if key not in state_stats:
                        state_stats[key] = {'length': 0, 'data': [{}, {}]}
                    if state_stats[key]['length'] == len(gameboard[key]):
                        pass
                        # print('No dice change updates')
                    else:
                        state_stats[key]['length'] += 1
                        for idx, item in enumerate(gameboard[key][-1]):
                            if item not in state_stats[key]['data'][idx]:
                                state_stats[key]['data'][idx][item] = 1
                            else:
                                state_stats[key]['data'][idx][item] += 1
                else:  # (str, bool int float)
                    if key not in state_stats:
                        state_stats[key] = {tuple(gameboard[key]): 1}
                    elif tuple(gameboard[key]) not in state_stats[key]:
                        state_stats[key][tuple(gameboard[key])] = 1  # TODO: I think we need this somewhere else too?
                    else:
                        state_stats[key][tuple(gameboard[key])] += 1

            else:  # (str, bool, int, float)
                if key not in state_stats:
                    state_stats[key] = {gameboard[key]: 1}
                elif gameboard[key] not in state_stats[key]:
                    state_stats[key][gameboard[key]] = 1  # TODO: I think we need this somewhere else too?
                else:
                    state_stats[key][gameboard[key]] += 1
        return state_stats


    def check_novelty(self, state_stats, parsed_json):
        novelty = False
        novelty_properties = {}
        for key in parsed_json:
            print(key)
            novelty_properties['trigger'] = [key]

            if (not parsed_json[key]) or (key in ('cards', 'history', 'players', 'locations', 'die_sequence')):
                continue
            elif type(parsed_json[key]) is dict:  # TODO balloch: untested
                if key not in state_stats:
                    novelty = True
                    novelty_properties['trigger'] = [key]
                    break
                novelty, novelty_properties = self.check_novelty(state_stats[key],
                                                                 parsed_json[key])
            elif type(parsed_json[key]) is list:
                if type(parsed_json[key][-1]) is dict:  # history # TODO balloch: untested
                    if key not in state_stats:
                        novelty = True
                    for idx, subdict in enumerate(parsed_json[key]):
                        if idx > (len(state_stats[key]) - 1):
                            novelty = True
                            novelty_properties['trigger'] = [key, subdict]
                        else:  # TODO: if subtree the same, add count, if not, insert? frankly list should become a tree/dict
                            novelty, novelty_properties = self.check_novelty(state_stats[key][idx],
                                                                             parsed_json[key][idx])
                            # novelty_properties.update(temp_novelty_properties)
                # elif type(gameboard[key][-1]) is list:  # hardcoding this because die sequence only list of lists
                #     if key not in state_stats:
                #         state_stats[key] = {'length': 0, 'data': [{},{}]}
                #     if state_stats[key]['length'] == len(gameboard[key]):
                #         pass
                #         # print('No dice change updates')
                #     else:
                #         state_stats[key]['length'] += 1
                #         for idx, item in enumerate(gameboard[key][-1]):
                #             if item not in state_stats[key]['data'][idx]:
                #                 state_stats[key]['data'][idx][item] = 1
                #             else:
                #                 state_stats[key]['data'][idx][item] += 1
                else:  # (str, bool int float) # TODO balloch: untested
                    if key not in state_stats:
                        novelty = True
                    elif tuple(parsed_json[key]) not in state_stats[key]:
                        novelty = True
                    else:
                        continue
                        # state_stats[key][tuple(gameboard[key])] += 1

            else:  # (str, bool, int, float) # TODO balloch: untested
                if parsed_json[key] not in state_stats[key]:
                    novelty = True
                    novelty_properties['trigger'] = [parsed_json[key]]
                    break
                else:
                    continue
                    # state_stats[key][parsed_json[key]] += 1

            if novelty == True:
                # novelty_properties.insert()
                break
        return novelty, novelty_properties


def main():
    game_clone = GameClone()
    state_stats = dict()
    for i in range(100):
        try:
            json_file = os.path.join('sample_json_data', 'data_' + '1' + '_' + str(i) + '.json')
        except:
            continue
        with open(json_file, 'r') as infile:
            data_dict_from_server = json.load(infile)
        state_stats = game_clone.update_novelty_detector(state_stats, data_dict_from_server['current_gameboard'])
    print(state_stats)

    bad_json_file = os.path.join('sample_json_data', 'bad_data_' + '1' + '_' + '10' + '.json')
    with open(bad_json_file, 'r') as badfile:
        bad_data_dict_from_server = json.load(badfile)
    print(game_clone.check_novelty(state_stats, bad_data_dict_from_server['current_gameboard']))


            #### ONLY FOR CURRENT GAMEBOARD
if __name__ == '__main__':
    main()