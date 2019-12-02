# This script join two SQuAD datasets
import argparse
import random
import logging
import json
import os

logging.basicConfig(level=logging.INFO)

def join(squad_file1, squad_file2):
    with open(squad_file1) as sf1:
        squad1 = json.load(sf1)
    
    with open(squad_file2) as sf2:
        squad2 = json.load(sf2)
    if squad1['version'] == squad2['version']:
        squad_joint = {"version": squad1['version'], "data":[]}
        data_squad1, data_squad2 = squad1['data'], squad2['data']
        squad_joint["data"] = data_squad1 + data_squad2

        # Shuffle the data with fixed seed
        random.seed(10)
        random.shuffle(squad_joint["data"])
        squad_joint_filename = os.path.join(os.path.dirname(squad_file2),
                                            'joint_{}_{}.json'.format(os.path.basename(squad_file1),
                                                                 os.path.basename(squad_file2)))
        with open(squad_joint_filename, 'w') as sjf:
            json.dump(squad_joint, sjf)
    else:
        raise ValueError('The two SQUAD file must have the same version!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('squad_file1', type=str, help='SQUAD file number 1')
    parser.add_argument('squad_file2', type=str, help='SQUAD file number 2')
    args = parser.parse_args()
    logging.info('Join {} and {}'.format(args.squad_file1, args.squad_file2))
    join(args.squad_file1, args.squad_file2)