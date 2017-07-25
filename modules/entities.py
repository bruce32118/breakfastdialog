# coding=UTF-8
from enum import Enum
import numpy as np


class EntityTracker():


    def __init__(self):
                self.entities = {
                    '<cuisine>' : None,
                    '<drink>' : None,
                    }                                                                                 
                self.num_features = 2 # tracking 4 entities
                self.rating = None

                                # constants
                self.cuisines = ['麵', '三明治', '蛋餅', '雞塊', '漢堡','薯條']
                self.drink = ['紅茶','奶茶','可樂']
                self.EntType = Enum('Entity Type', '<drink> <cuisine> <non_ent>')

    def ent_type(self, ent):
        #if ent in self.party_sizes:
        #    return self.EntType['<party_size>'].name
        #elif ent in self.locations:
        #    return self.EntType['<location>'].name
        if ent in self.cuisines:
            return self.EntType['<cuisine>'].name
        elif ent in self.drink:
            return self.EntType['<drink>'].name
        else:
            return ent


    def extract_entities(self, utterance, update=True):
        tokenized = []
        for word in utterance.split(' '):
            entity = self.ent_type(word)
            if word != entity and update:
                self.entities[entity] = word

            tokenized.append(entity)

        return ' '.join(tokenized)


    def context_features(self):
       keys = list(set(self.entities.keys()))
       self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys], 
                                   dtype=np.float32 )
       return self.ctxt_features


    def action_mask(self):
        print('Not yet implemented. Need a list of action templates!')
