from entities import EntityTracker
from bow import BoW_encoder
from lstm_net import LSTM_net
from embed import UtteranceEmbed
from actions import ActionTracker
from data_utils import Data
import util as util

import numpy as np
import sys


class Trainer():

        def __init__(self):

            et = EntityTracker()
            self.bow_enc = BoW_encoder()
            self.emb = UtteranceEmbed()
            at = ActionTracker(et)

            self.dataset, dialog_indices = Data(et, at).trainset
            self.predata_test = Data(et, at).trainset_test
            self.dialog_indices_tr = dialog_indices[:200]
            self.dialog_indices_dev = dialog_indices[200:250]
            
            obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
            self.action_templates = at.get_action_templates()
            action_size = at.action_size
            nb_hidden = 128

            self.net = LSTM_net(obs_size=obs_size,action_size=action_size,nb_hidden=nb_hidden)
             

if __name__ == '__main__':
        # setup trainer
            trainer = Trainer()
                # start training
            #print trainer.dialog_indices
            print trainer.dataset[0:10]
            #print trainer.predata_test
            #print trainer.dialog_indices_tr
        
