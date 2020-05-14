import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/GNOME-p3','')
upper_path_eva = upper_path + '/Evaluation/GNOME-p3'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation/GNOME-p3')
#####################################

from monopoly_simulator.agent import Agent
from multiprocessing.connection import Listener
import monopoly_simulator.action_choices as action_choices
import monopoly_simulator.background_agent_v3 as background_agent_v3

# All of the gameplay functions just send a request to the client to call the function with the given arguments and send
# back a reply which is then returned.

def recover(back, player, current_gameboard):
    if 'player' in back[1].keys():
        back[1]['player'] = player
    if 'current_gameboard' in back[1].keys():
        back[1]['current_gameboard'] = current_gameboard

def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    return background_agent_v3.make_pre_roll_move(player, current_gameboard, allowable_moves, code)


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    return background_agent_v3.make_out_of_turn_move(player, current_gameboard, allowable_moves, code)


def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    player.agent.conn.send(("make_post_roll_move", (player, current_gameboard, allowable_moves, code)))
    back = player.agent.conn.recv()
    recover(back, player, current_gameboard)
    return back


def make_buy_property_decision(player, current_gameboard, asset):
    return background_agent_v3.make_buy_property_decision(player, current_gameboard, asset)


def make_bid(player, current_gameboard, asset, current_bid):
    return background_agent_v3.make_bid(player, current_gameboard, asset, current_bid)


def handle_negative_cash_balance(player, current_gameboard):
    player.agent.conn.send(("handle_negative_cash_balance", (player, current_gameboard)))
    back = player.agent.conn.recv()
    return background_agent_v3.handle_negative_cash_balance(player, current_gameboard)


def _build_decision_agent_methods_dict():
    ans = dict()
    ans['handle_negative_cash_balance'] = handle_negative_cash_balance
    ans['make_pre_roll_move'] = make_pre_roll_move
    ans['make_out_of_turn_move'] = make_out_of_turn_move
    ans['make_post_roll_move'] = make_post_roll_move
    ans['make_buy_property_decision'] = make_buy_property_decision
    ans['make_bid'] = make_bid
    ans['type'] = "decision_agent_methods"
    return ans


class ServerAgent(Agent):
    """
    To play over TCP, start a game with at least one ServerAgent. The ServerAgent will wait for a connection from a
    ClientAgent, and then relay all game state information to the client. The client will decide what move to make
    and send the result back to the ServerAgent.
    """

    def __init__(self, address=('localhost', 6001), authkey=b"password"):
        """
        Create a new ServerAgent on a particular port. If you are playing a game with multiple server agents, make sure
        each is operating on a different port.
        @param address: Tuple, the address and port number. Defaults to localhost:6000
        @param authkey: Byte string, the password used to authenticate the client. Defaults to "password"
        """
        super().__init__(**_build_decision_agent_methods_dict())
        print("Waiting for connection...")
        self.listener = Listener(address, authkey=authkey)
        self.conn = self.listener.accept()
        print('Connection accepted from', self.listener.last_accepted)

    def __getstate__(self):
        """Make sure that the socket connection doesn't get pickled."""
        out = self.__dict__.copy()
        out['listener'] = None
        out['conn'] = None
        return out

    def startup(self, current_gameboard, indicator=None):
        """Performs normal Agent startup and signals for the client agent to do the same"""
        super().startup(current_gameboard, indicator)
        self.conn.send(("startup", (current_gameboard, indicator)))
        return self.conn.recv()

    def shutdown(self):
        """Performs normal Agent shutdown and signals for the client agent to do the same, then closes the connection"""
        self.conn.send(("shutdown", ()))
        # self.conn.close()
        # self.listener.close()
        # return super().shutdown()
        result = -1
        result = self.conn.recv()
        return result

    def end_tournament(self):
        self.conn.send(("end_tournament", ()))
        self.conn.close()
        self.listener.close()
        return super().shutdown()

    # def start_tournament(self):
    #     self.conn.send(("start_tournament", ()))
    #     return self.conn.recv()


