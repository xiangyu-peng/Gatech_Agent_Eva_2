import networkx as nx
state = nx.DiGraph()
state.add_ndoe('0')
state.add_ndoe('1')
state.add_ndoe('2')

state.add_edge('0', '1')
state.add_edge('0', '2')

print(state['0'])