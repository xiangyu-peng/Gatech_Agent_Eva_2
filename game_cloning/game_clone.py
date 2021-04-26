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
        data_dict_from_server = json.load(gameboard_message)

        novelty_properties = {}

        ## TODO balloch: dummy
        if not self.red_button and np.random.rand() > 0.99:
            self.red_button = True
            novelty_properties['type'] = 'dummy'
            novelty_properties['trigger'] = 'random'

        return self.red_button, novelty_properties
