import os
import tempfile
from pathlib import Path
from subprocess import Popen
from sys import stderr
from zipfile import ZipFile

import wget


class KG_OpenIE():
    def __init__(self, core_nlp_version: str = '2018-10-05', jsonfile='/media/becky/GNOME-p3/KG-rule/json_kg.json',\
                 setfile='/media/becky/GNOME-p3/KG-rule/kg_set.json'):
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
        self.jsonfile = jsonfile
        self.setfile = setfile
        self.client = CoreNLPClient(annotators=['openie'], memory='8G')
        self.relations = ['priced', 'rented', 'located', 'colored', 'classified', 'away']
        self.kg_rel = dict()
        self.kg_sub = dict()
        self.kg_set = set()
        self.kg_rel_diff = dict()
        self.kg_sub_diff = dict()
        self.kg_introduced = False

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

    def kg_update(self, triple, level='sub'):
        if level == 'sub':
            if triple['subject'] not in self.kg_sub_diff.keys():
                self.kg_sub_diff[triple['subject']] = dict()
                self.kg_sub_diff[triple['subject']][triple['relation']] = [self.kg_sub[triple['subject']][triple['relation']]]
            self.kg_sub[triple['subject']][triple['relation']] = triple['object']
            self.kg_sub_diff[triple['subject']][triple['relation']].append(triple['object'])
            return (triple['subject'],triple['relation'],self.kg_sub_diff[triple['subject']][triple['relation']])
        else:
            if triple['relation'] not in self.kg_rel_diff.keys():
                self.kg_rel_diff[triple['relation']] = dict()
                self.kg_rel_diff[triple['relation']][triple['subject']] = [self.kg_rel[triple['relation']][triple['subject']]]
            self.kg_rel[triple['relation']][triple['subject']] = triple['object']
            self.kg_rel_diff[triple['relation']][triple['subject']].append(triple['object'])
            return (triple['subject'], triple['relation'], self.kg_rel_diff[triple['relation']][triple['subject']])

    def kg_add(self, triple, level='sub'):
        '''
        :param triple (dict): triple is a dict with three keys: subject, relation and object
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed, False means not changed or no existing rule, and add a rule
    '''
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
        else: #level = 'rel'
            if triple['relation'] in self.kg_rel.keys():
                if triple['subject'] in self.kg_rel[triple['relation']].keys():
                    return True if self.kg_rel[triple['relation']][triple['subject']] != triple['object'] else False
                else:
                    self.kg_rel[triple['relation']][triple['subject']] = triple['object']

            else:
                self.kg_rel[triple['relation']] = dict()
                self.kg_rel[triple['relation']][triple['subject']] = triple['object']
            return False


    def build_kg(self, text, level='sub',use_hash=False):
        '''
        :param text (string): One sentence from logging info
        :param level (string): level indicates the kg dict's keys. i.e. level='rel' means kg_rel.keys are relations
        :return: bool. True indicates the rule is changed
        '''
        diff = []
        if use_hash:
            triple_hash = hash(text)
            if triple_hash in self.kg_set:
                return diff
            else:
                self.kg_set.add(triple_hash)
        entity_relations = self.annotate(text, simple_format=True)
        for er in entity_relations:
            kg_change_once = self.kg_add(er,level=level)
            if kg_change_once:
                diff.append(self.kg_update(er, level=level))

        return diff

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
                kg_change = self.kg_add(er)
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

    def save_json(self, level='sub'):
        import json
        if level == 'sub':
            with open(self.jsonfile, 'w') as f:
                json.dump(self.kg_sub, f)
        else:
            with open(self.jsonfile, 'w') as f:
                json.dump(self.kg_rel, f)

    def read_json(self, level='sub'):
        import json
        with open(self.jsonfile, 'r') as f:
            if level == 'sub':
                self.kg_sub = json.load(f)
            else:
                self.kg_rel = json.load(f)

import time
start = time.time()
file='/media/becky/GNOME-p3/monopoly_simulator/gameplay.log'
log_file = open(file,'r')
client = KG_OpenIE()

# client.read_json(level='rel')
for line in log_file:
    kg_change = client.build_kg(line,level='rel',use_hash=True)
for line in log_file:
    kg_change = client.build_kg(line,level='rel',use_hash=True)
for line in log_file:
    kg_change = client.build_kg(line,level='rel',use_hash=True)
# client.save_json(level='rel')
# client.generate_graphviz_graph_(png_filename='graph.png',kg_level='rel')

# print(client.kg_rel.keys())
# line = 'Vermont Avenue is colored as SkyBlue'
# kg_change = client.build_kg(line,level='sub')
print(client.kg_rel_diff)
end = time.time()
print(str(end-start))