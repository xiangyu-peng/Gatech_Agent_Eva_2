import os
import json
import numpy as np
# import torch

class GameClone():
    def __init__(self, state_stats_json_file='game_clone_state_stats.json', rules_manual_json=None):
        """
        Sets up the game cloning rules, where the main structure
        Args:
            rules_manual: json of rules
        """
        self.state_stats = dict()
        if os.path.isfile(state_stats_json_file):
            try:
                self.load_state_stats(state_stats_json_file)
            except:
                print('state_stats failed to load')
        else:
            print('state_stats failed to load')

        self.rules = dict()
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
        self.gamestate_avoided_keys = {'history', 'players', 'picked_chance_cards', 'picked_community_chest_cards'} # 'cards'

    def load_state_stats(self, state_stats_json_file='game_clone_state_stats.json', overwrite=True):
        if overwrite or not self.state_stats:
            with open(state_stats_json_file, 'r') as in_file:
                loaded_json = json.load(in_file)
                self.state_stats = loaded_json
                self._remap_unhashables(self.state_stats)

            return 'loaded'
        else:
            return 'NOT Loaded'

    def load_rules(self, rules_manual_json):
        self.rules_manual_json = rules_manual_json
        self._process_manual()

    def save_rules(self, rules_output_file='game_clone_rules.json', overwrite=False):
        if self.rules:
            return self._safe_save_json(self.rules, rules_output_file)
        else:
            return None

    def save_state_stats(self, state_stats_output_file='game_clone_state_stats.json', overwrite=False):
        if self.state_stats:
            return self._safe_save_json(self.state_stats, state_stats_output_file)
        else:
            return None

    def _safe_save_json(self, save_attr, output_file, overwrite=False):
        if (not overwrite) and os.path.isfile(os.path.abspath(output_file)):
            counter = 0
            output_file = output_file[:-5] + str(counter)
            while os.path.isfile(os.path.abspath(output_file+ '.json')):
                counter += 1
                output_file = output_file[:-len(str(counter - 1))] + str(counter)
            output_file = output_file + '.json'
        with open(output_file, 'w') as of:
            json.dump(self._map_unhashables(save_attr), of, indent=4)
        return os.path.abspath(output_file)

    def _map_unhashables(self, pre_dict, fixed_dict={}):
        for key in pre_dict:
            if type(key) not in (str, int, float, bool, None):
                fixed_key = 'strmap' + str(key)
            else:
                fixed_key = key
            if type(pre_dict[key]) is dict:
                fixed_dict[fixed_key] = self._map_unhashables(pre_dict[key], {})
            elif type(pre_dict[key]) is list:
                fixed_dict[fixed_key] = []
                for idx, elem in enumerate(pre_dict[key]):
                    if type(elem) is dict:
                        fixed_dict[fixed_key].append(self._map_unhashables(pre_dict[key][idx], {}))
                    elif type(elem) is list:  # TODO: untested
                        fixed_dict[fixed_key].append(self._map_unhashables(pre_dict[key][idx], []))
                    else: # type(key) in (str, int, float, bool, None):
                        fixed_dict[fixed_key].append(pre_dict[key][idx])
                pass ## DO ALL
            elif type(pre_dict[key]) is tuple:
                print('weve got a probelm')
            else: # type(key) in (str, int, float, bool, None):
                fixed_dict[fixed_key] = pre_dict[key]
        return fixed_dict

    def _remap_unhashables(self, pre_dict):
        for key in pre_dict:
            if type(key) is str:
                if key[:6] == 'strmap': # not in (str, int, float, bool, None):
                    fixed_key = eval(key[6:])
                    pre_dict[fixed_key] = pre_dict[key]
                    del pre_dict[key]
                else:
                    fixed_key = key

            if type(pre_dict[fixed_key]) is dict:
                pre_dict[fixed_key] = self._remap_unhashables(pre_dict[fixed_key])
            elif type(pre_dict[fixed_key]) is list:
                for idx, elem in enumerate(pre_dict[fixed_key]):
                    if type(elem) is dict:
                        pre_dict[fixed_key][idx] = self._remap_unhashables(pre_dict[fixed_key][idx])
                    elif type(elem) is list:  # TODO: untested
                        pre_dict[fixed_key][idx] = self._remap_unhashables(pre_dict[fixed_key][idx])
            elif type(pre_dict[fixed_key]) is tuple:
                print('weve got a probelm')
        return pre_dict

    # def pythonify_dict(self, json_data):
    #     correctedDict = {}
    #
    #     for key, value in json_data.items():
    #         if isinstance(value, list):
    #             value = [self.pythonify_dict(item) if isinstance(item, dict) else item for item in value]
    #         elif isinstance(value, dict):
    #             value = self.pythonify_dict(value)
    #         try:
    #             key = int(key)
    #         except Exception as ex:
    #             pass
    #         correctedDict[key] = value
    #
    #     return correctedDict

    def _process_manual(self):
        with open(self.rules_manual_json, 'r') as f:
            self.rules = json.load(self.rules_manual_json)

    def update_novelty_detector(self, parsed_json, subtree=None):
        # tree_search
        if subtree is None:
            subtree = self.state_stats
        for key in parsed_json:
            if (not parsed_json[key]) or (key in self.gamestate_avoided_keys):  # 'cards'
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
                                subtree[key]['data'][idx][str(item)] = 1
                            else:
                                subtree[key]['data'][idx][str(item)] += 1
                else:  # (str, bool int float)
                    if key not in subtree:
                        subtree[key] = {str(parsed_json[key]): 1}
                    elif str(parsed_json[key]) not in subtree[key]:
                        subtree[key][str(parsed_json[key])] = 1  # TODO: I think we need this somewhere else too?
                    else:
                        subtree[key][str(parsed_json[key])] += 1

            else:  # (str, bool, int, float)
                if key not in subtree:
                    subtree[key] = {str(parsed_json[key]): 1}
                elif str(parsed_json[key]) not in subtree[key]:
                    subtree[key][str(parsed_json[key])] = 1  # TODO: I think we need this somewhere else too?
                else:
                    try:
                        subtree[key][str(parsed_json[key])] += 1
                    except TypeError:
                        print('key: ', key, ' - parsed_json[key]: ', parsed_json[key])
        return subtree

    def update_rules(self, samples):
        raise NotImplementedError

    def gc_detect_novelty(self, gameboard_message, message_type='str'):
        if (message_type in ('str')) or (type(gameboard_message) in (bytes, bytearray)):  # JSON string
            data_dict_from_server = json.loads(gameboard_message)
        elif message_type == 'file':  # JSON file
            data_dict_from_server = json.load(gameboard_message)
        else:  # Already loaded
            data_dict_from_server=gameboard_message
        if self.red_button:
            return True, {'message': 'novelty already detected!'}
        if 'current_gameboard' in data_dict_from_server:
            novelty, novelty_properties = self.check_novelty(data_dict_from_server['current_gameboard'])
        if novelty:
            self.red_button = True

        return novelty, novelty_properties

    def check_novelty(self, parsed_json, subtree=None):
        if subtree is None:
            subtree = self.state_stats

        novelty = False
        novelty_properties = {}
        for key in parsed_json:
            if (not parsed_json[key]) or (key in self.gamestate_avoided_keys):
                continue
            elif key not in subtree:
                if key == 'perform_action':
                    continue
                else:
                    novelty = True
                    novelty_properties['trigger'] = [key]
                    break
            elif type(parsed_json[key]) is dict:
                novelty, novelty_properties_add = self.check_novelty(parsed_json[key],
                                                                     subtree[key])
                if novelty_properties_add:
                    novelty_properties = self._extend_properties(novelty_properties,
                                                                 novelty_properties_add)

            elif type(parsed_json[key]) is list:
                if type(parsed_json[key][-1]) is dict:  # list of dicts - history # TODO balloch: untested
                    for idx, subjson in enumerate(parsed_json[key]):
                        if idx > (len(subtree[key]) - 1):  # WARNING: EXCEPT WITH HISTORY
                            novelty = True
                            novelty_properties = self._extend_properties(novelty_properties,
                                                                         {'trigger': [key, parsed_json[key]]})
                            break
                        else:  # TODO: if subtree the same, add count, if not, insert? frankly list should become a tree/dict
                            novelty, novelty_properties_add = self.check_novelty(parsed_json[key][idx],
                                                                                 subtree[key][idx])
                            if novelty_properties_add:
                                novelty_properties = self._extend_properties(novelty_properties,
                                                                         novelty_properties_add)

                            # novelty_properties.update(temp_novelty_properties)
                elif type(parsed_json[key][-1]) is list:  # list of lists - hardcoding this because die sequence only
                    if subtree[key]['length'] == len(parsed_json[key]):
                        pass  # TODO balloch: this logic works but probably should be more rigorous
                        # print('No dice change updates')
                    else:
                        for idx, item in enumerate(parsed_json[key][-1]):
                            if str(item) not in subtree[key]['data'][idx]:
                                novelty = True
                                novelty_properties = self._extend_properties(novelty_properties,
                                                                             {'trigger': [key, parsed_json[key]]})
                                break
                            else:
                                continue
                                # subtree[key]['data'][idx][item] += 1
                else:  # list of (str, bool int float)
                    if str(parsed_json[key]) not in subtree[key]:
                        novelty = True
                        novelty_properties = self._extend_properties(novelty_properties,
                                                                     {'trigger': [key, parsed_json[key]]})
                        break
                    else:
                        continue
                        # subtree[key][tuple(parsed_json[key])] += 1

            else:  # (str, bool, int, float) # TODO balloch: untested
                if str(parsed_json[key]) not in subtree[key]:
                    novelty = True
                    novelty_properties = self._extend_properties(novelty_properties,
                                                                 {'trigger': [key, parsed_json[key]]})
                    break
                else:
                    continue
                    # subtree[key][parsed_json[key]] += 1

            if novelty:
                print("novelty!")
                # novelty_properties.insert()
                break
        return novelty, novelty_properties

    def _extend_properties(self, novelty_properties, novelty_properties_add):
        if 'trigger' in novelty_properties_add:
            if 'trigger' in novelty_properties:
                novelty_properties['trigger'].extend(novelty_properties_add['trigger'])
            else:
                novelty_properties['trigger'] = novelty_properties_add['trigger']
        else:
            print('weve got a problem: none-trigger content in dict')
            return None
        return novelty_properties

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



def main():
    game_clone = GameClone()
    for game in range(1,98): #first game is junk
        count = 0
        json_file = os.path.join(os.environ['HOME'], 'sample_json_data', 'data_' + str(game) + '_' + str(count) + '.json')

        while os.path.isfile(json_file):
            with open(json_file, 'r') as infile:
                data_dict_from_server = json.load(infile)
            if 'current_gameboard' in data_dict_from_server:
                state_stats = game_clone.update_novelty_detector(data_dict_from_server['current_gameboard'])
            else:
                print('bad sample: ', data_dict_from_server)
            count += 1
            json_file = os.path.join(os.environ['HOME'], 'sample_json_data', 'data_' + str(game) + '_' + str(count) + '.json')
    # print('orig_state stats: ', state_stats)

    bad_json_file = os.path.join('sample_json_data', 'bad_data_' + '1' + '_' + '10' + '.json')
    with open(bad_json_file, 'r') as badfile:
        bad_data_dict_from_server = json.load(badfile)
    # print('ID novelty: ', game_clone.gc_detect_novelty(bad_data_dict_from_server, 'loaded'))
    gcss_file = game_clone.save_state_stats()
    # print('file: ', gcss_file)

    game_clone.load_state_stats(gcss_file, True)
    # print('loaded stats: ',game_clone.state_stats)
    print('Novelty? ', game_clone.state_stats == state_stats)


            #### ONLY FOR CURRENT GAMEBOARD
if __name__ == '__main__':
    main()
