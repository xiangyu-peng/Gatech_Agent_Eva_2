import os
import tempfile
from pathlib import Path
from subprocess import Popen
from sys import stderr
from zipfile import ZipFile

import wget


class KG_OpenIE():
    def __init__(self, core_nlp_version: str = '2018-10-05'):
        self.remote_url = 'https://nlp.stanford.edu/software/stanford-corenlp-full-{}.zip'.format(core_nlp_version)
        self.install_dir = Path('~/.stanfordnlp_resources/').expanduser()
        self.install_dir.mkdir(exist_ok=True)
        if not (self.install_dir / Path('stanford-corenlp-full-{}'.format(core_nlp_version))).exists():
            print('Downloading from %s.' % self.remote_url)
            output_filename = wget.download(self.remote_url, out=str(self.install_dir))
            print('\nExtracting to %s.' % self.install_dir)
            zf = ZipFile(output_filename)
            zf.extractall(path=self.install_dir)
            zf.close()

        os.environ['CORENLP_HOME'] = str(self.install_dir / 'stanford-corenlp-full-2018-10-05')
        from stanfordnlp.server import CoreNLPClient
        self.client = CoreNLPClient(annotators=['openie'], memory='8G')
        self.relations = ['priced', 'rented', 'located', 'colored', 'classified', 'away']
        self.kg_rel = dict()
        self.kg_sub = dict()
        self.board_state = ['Go', 'Mediterranean Avenue', 'Community Chest',
                            'Baltic Avenue', 'Income Tax', 'Reading Railroad', 'Oriental Avenue',
                            'Chance-One', 'Vermont Avenue', 'Connecticut Avenue', 'In Jail/Just Visiting',
                            'St. Charles Place', 'Electric Company', 'States Avenue', 'Virginia Avenue',
                            'Pennsylvania Railroad', 'St. James Place', 'Community Chest', 'Tennessee Avenue',
                            'New York Avenue', 'Free Parking', 'Kentucky Avenue', 'Chance-Two', 'Indiana Avenue',
                            'Illinois Avenue', 'B&O Railroad', 'Atlantic Avenue', 'Ventnor Avenue',
                            'Water Works', 'Marvin Gardens', 'Go to Jail', 'Pacific Avenue', 'North Carolina Avenue',
                            'Community Chest', 'Pennsylvania Avenue', 'Short Line', 'Chance-Three', 'Park Place',
                            'Luxury Tax', 'Boardwalk']

    def annotate(self, text: str, properties_key: str = None, properties: dict = None, simple_format: bool = True):
        """
        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (str) properties_key: key into properties cache for the client
        :param (dict) properties: additional request properties (written on top of defaults)
        :param (bool) simple_format: whether to return the full format of CoreNLP or a simple dict.
        :return: Depending on simple_format: full or simpler format of triples <subject, relation, object>.
        """
        # https://stanfordnlp.github.io/CoreNLP/openie.html
        core_nlp_output = self.client.annotate(text=text, annotators=['openie'], output_format='json',
                                               properties_key=properties_key, properties=properties)
        if simple_format:
            triples = []
            for sentence in core_nlp_output['sentences']:
                for triple in sentence['openie']:

                    for rel in self.relations:
                        if rel in triple['relation']:
                            triples.append({
                                'subject': triple['subject'],
                                'relation': triple['relation'],
                                'object': triple['object']
                            })
                    # triples.append({
                    #     'subject': triple['subject'],
                    #     'relation': triple['relation'],
                    #     'object': triple['object']
                    # })
            return triples
        else:
            return core_nlp_output

    def kg_update(self, triple, level='rel'):
        if level == 'sub':
            if triple['subject'] in self.kg_sub.keys():
                if triple['relation'] in self.kg_sub[triple['subject']].keys():
                    return True if self.kg_sub[triple['subject']][triple['relation']] != triple['object'] else False
                else:
                    self.kg_sub[triple['subject']][triple['relation']] = triple['object']

            else:
                self.kg_sub[triple['subject']] = dict()
                self.kg_sub[triple['subject']][triple['relation']] = triple['object']
            return False
        else:
            if triple['relation'] in self.kg_rel.keys():
                if triple['subject'] in self.kg_rel[triple['relation']].keys():
                    return True if self.kg_rel[triple['relation']][triple['subject']] != triple['object'] else False
                else:
                    self.kg_rel[triple['relation']][triple['subject']] = triple['object']

            else:
                self.kg_rel[triple['relation']] = dict()
                self.kg_rel[triple['relation']][triple['subject']] = triple['object']
            return False


    def build_kg(self, text, level='rel'):
        kg_change = False
        entity_relations = self.annotate(text, simple_format=True)
        for er in entity_relations:
            kg_change_once = self.kg_update(er,level=level)
            if kg_change_once:
                kg_change = True
        return kg_change

    #plot the knowledge graph
    def generate_graphviz_graph_(self, text: str = '', png_filename: str = './out/graph.png', level:str = 'acc', kg_level='rel'):
        """
       :param (str | unicode) text: raw text for the CoreNLPServer to parse
       :param (list | string) png_filename: list of annotators to use
       :param (str) level: control we plot the whole image all the local knowledge graph
       """
        entity_relations = self.annotate(text, simple_format=True)
        """digraph G {
        # a -> b [ label="a to b" ];
        # b -> c [ label="another label"];
        }"""
        if level == 'single':
            graph = list()
            graph.append('digraph {')
            for er in entity_relations:
                kg_change = self.kg_update(er)
                graph.append('"{}" -> "{}" [ label="{}" ];'.format(er['subject'], er['object'], er['relation']))
            graph.append('}')
        else:
            graph = list()
            graph.append('digraph {')
            if kg_level == 'rel':
                for rel in self.kg_rel.keys():
                    for sub in self.kg_rel[rel]:
                        graph.append('"{}" -> "{}" [ label="{}" ];'.format(sub, self.kg_rel[rel][sub], rel))
            else:
                for sub in self.kg_sub.keys():
                    for rel in self.kg_sub[sub]:
                        graph.append('"{}" -> "{}" [ label="{}" ];'.format(sub, self.kg_sub[sub][rel], rel))
            graph.append('}')

        output_dir = os.path.join('.', os.path.dirname(png_filename))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        out_dot = os.path.join(tempfile.gettempdir(), 'graph.dot')
        with open(out_dot, 'w') as output_file:
            output_file.writelines(graph)

        command = 'dot -Tpng {} -o {}'.format(out_dot, png_filename)
        dot_process = Popen(command, stdout=stderr, shell=True)
        dot_process.wait()
        assert not dot_process.returncode, 'ERROR: Call to dot exited with a non-zero code status.'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        self.client.stop()
        del os.environ['CORENLP_HOME']

file='/media/becky/GNOME-p3/monopoly_simulator/gameplay.log'
log_file = open(file,'r')
client = KG_OpenIE()
for line in log_file:
    kg_change = client.build_kg(line,level='rel')
client.generate_graphviz_graph_(png_filename='graph.png',kg_level='rel')

print(client.kg_rel.keys())
# line = 'Vermont Avenue is colored as SkyBlue'
# kg_change = client.build_kg(line,level='sub')
# print(client.kg_sub)