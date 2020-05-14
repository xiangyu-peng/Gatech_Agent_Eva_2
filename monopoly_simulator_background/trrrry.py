import json
schema = json.load(open('/media/becky/GNOME-p3/monopoly_game_schema_v1-1.json', 'r'))
print(schema['players']['player_states']['starting_cash'])