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
        self.rules = dict()
        self.rules_manual_json = rules_manual_json
        self.red_button = False
        if rules_manual_json:
            self._process_manual()

        self.state_stats = dict()
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

        if self.red_button:
            return True, {'message': 'novelty alredy detected!'}

        novelty, novelty_properties = self.check_novelty()
        if novelty:
            self.red_button = True

        return self.novelty, novelty_properties


    def update_novelty_detector(self, parsed_json, subtree=None):
        # tree_search
        if subtree is None:
            subtree = self.state_stats
        for key in parsed_json:
            if (not parsed_json[key]) or (key in ('cards', 'history', 'players')):
                continue
            elif type(parsed_json[key]) is dict:
                if key not in subtree:
                    subtree[key] = {}
                subtree[key] = self.update_novelty_detector(parsed_json[key], subtree[key])

            elif type(parsed_json[key]) is list:
                if type(parsed_json[key][-1]) is dict:  # history # TODO balloch: untested
                    if key not in subtree:
                        subtree[key] = []
                    for idx, subjson in enumerate(parsed_json[key]):
                        if idx > (len(subtree[key]) - 1):
                            subtree[key].append(self.update_novelty_detector(subjson, {}))
                        else:  # TODO: if subtree the same, add count, if not, insert? frankly list should become a tree
                            subtree[key][idx] = self.update_novelty_detector(parsed_json[key][idx], subtree[key][idx])
                elif type(parsed_json[key][-1]) is list:  # hardcoding this because die sequence only list of lists
                    if key not in subtree:
                        subtree[key] = {'length': 0, 'data': [{}, {}]}
                    if subtree[key]['length'] == len(parsed_json[key]):
                        pass  # do nothing
                        # print('No dice change updates')
                    else:
                        subtree[key]['length'] += 1
                        for idx, item in enumerate(parsed_json[key][-1]):
                            if item not in subtree[key]['data'][idx]:
                                subtree[key]['data'][idx][item] = 1
                            else:
                                subtree[key]['data'][idx][item] += 1
                else:  # (str, bool int float)
                    if key not in subtree:
                        subtree[key] = {tuple(parsed_json[key]): 1}
                    elif tuple(parsed_json[key]) not in subtree[key]:
                        subtree[key][tuple(parsed_json[key])] = 1  # TODO: I think we need this somewhere else too?
                    else:
                        subtree[key][tuple(parsed_json[key])] += 1

            else:  # (str, bool, int, float)
                if key not in subtree:
                    subtree[key] = {parsed_json[key]: 1}
                elif parsed_json[key] not in subtree[key]:
                    subtree[key][parsed_json[key]] = 1  # TODO: I think we need this somewhere else too?
                else:
                    try:
                        subtree[key][parsed_json[key]] += 1
                    except TypeError:
                        print('key: ', key, ' - parsed_json[key]: ', parsed_json[key])
        return subtree


    def check_novelty(self, parsed_json, subtree=None):
        if subtree is None:
            subtree = self.state_stats

        novelty = False
        novelty_properties = {}
        for key in parsed_json:
            print(key)
            novelty_properties['trigger'] = [key]

            if (not parsed_json[key]) or (key in ('cards', 'history', 'players', 'locations')):
                continue
            elif key not in subtree:
                novelty = True
                break
            elif type(parsed_json[key]) is dict:
                novelty, novelty_properties = self.check_novelty(subtree[key],
                                                                 parsed_json[key])
            elif type(parsed_json[key]) is list:
                if type(parsed_json[key][-1]) is dict:  # list of dicts - history # TODO balloch: untested
                    for idx, subjson in enumerate(parsed_json[key]):
                        if idx > (len(subtree[key]) - 1):  # WARNING: EXCEPT WITH HISTORY
                            novelty = True
                            novelty_properties['trigger'] = [key, parsed_json[key]]
                            break
                        else:  # TODO: if subtree the same, add count, if not, insert? frankly list should become a tree/dict
                            novelty, novelty_properties = self.check_novelty(subtree[key][idx],
                                                                             parsed_json[key][idx])
                            # novelty_properties.update(temp_novelty_properties)
                elif type(parsed_json[key][-1]) is list:  # list of lists - hardcoding this because die sequence only
                    if subtree[key]['length'] == len(parsed_json[key]):
                        pass  # TODO balloch: this logic works but probably should be more rigorous
                        # print('No dice change updates')
                    else:
                        for idx, item in enumerate(parsed_json[key][-1]):
                            if item not in subtree[key]['data'][idx]:
                                novelty = True
                                novelty_properties['trigger'] = [key, parsed_json[key]]
                                break

                            else:
                                subtree[key]['data'][idx][item] += 1
                else:  # list of (str, bool int float)
                    if tuple(parsed_json[key]) not in subtree[key]:
                        novelty = True
                        novelty_properties['trigger'] = [key, parsed_json[key]]
                        break
                    else:
                        continue
                        # subtree[key][tuple(parsed_json[key])] += 1

            else:  # (str, bool, int, float) # TODO balloch: untested
                if parsed_json[key] not in subtree[key]:
                    novelty = True
                    novelty_properties['trigger'] = [key, parsed_json[key]]
                    break
                else:
                    continue
                    # subtree[key][parsed_json[key]] += 1

            if novelty == True:
                # novelty_properties.insert()
                break
        return novelty, novelty_properties


def main():
    game_clone = GameClone()
    for i in range(100):
        try:
            json_file = os.path.join('sample_json_data', 'data_' + '1' + '_' + str(i) + '.json')
        except:
            continue
        with open(json_file, 'r') as infile:
            data_dict_from_server = json.load(infile)
        state_stats = game_clone.update_novelty_detector(data_dict_from_server['current_gameboard'])
    print(state_stats)

    bad_json_file = os.path.join('sample_json_data', 'bad_data_' + '1' + '_' + '10' + '.json')
    with open(bad_json_file, 'r') as badfile:
        bad_data_dict_from_server = json.load(badfile)
    print(game_clone.check_novelty(bad_data_dict_from_server['current_gameboard']))


            #### ONLY FOR CURRENT GAMEBOARD
if __name__ == '__main__':
    main()