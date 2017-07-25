# coding=UTF-8
import util as util
import numpy as np


'''
    0. '請問今天想要吃什麼呢'
    1. '請問有想吃的主餐嘛'
    2. '請問有想喝的飲料嘛'
    3. '這是您要的主餐和飲料'

    [1] : cuisine
    [2] : drink

'''
class ActionTracker():

    def __init__(self, ent_tracker):
        # maintain an instance of EntityTracker
        self.et = ent_tracker
        # get a list of action templates
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)
        # action mask
        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        '''
        self.am_dict = {
                '0000' : [ 4,8,1,14,7,15],
                '0001' : [ 4,8,1,14,7],
                '0010' : [ 4,8,1,14,15],
                '0011' : [ 4,8,1,14],
                '0100' : [ 4,8,1,7,15],
                '0101' : [ 4,8,1,7],
                '0110' : [ 4,8,1,15],
                '0111' : [ 4,8,1],
                '1000' : [ 4,8,14,7,15],
                '1001' : [ 4,8,14,7],
                '1010' : [ 4,8,14,15],
                '1011' : [ 4,8,14],
                '1100' : [ 4,8,7,15],
                '1101' : [ 4,8,7],
                '1110' : [ 4,8,15],
                '1111' : [ 2,3,5,6,8,9,10,11,12,13,16 ]
                }
                '''
        self.am_dict = {
                '00' : [ 1 ],
                '01' : [ 2 ],
                '10' : [ 3 ],
                '11' : [ 4 ]
                }

    def action_mask(self):
        # get context features as string of ints (0/1)
        ctxt_f = ''.join([ str(flag) for flag in self.et.context_features().astype(np.int32) ])

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index-1] = 1.
            return self.am
    
        return construct_mask(ctxt_f)

    def get_action_templates(self):
        responses = list(set([ self.et.extract_entities(response, update=False) 
            for response in util.get_responses() ]))

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'resto_' in word: 
                    if 'phone' in word:
                        template.append('<info_phone>')
                    elif 'address' in word:
                        template.append('<info_address>')
                    else:
                        template.append('<restaurant>')
                else:
                    template.append(word)
            return ' '.join(template)

        # extract restaurant entities
        return sorted(set([ extract_(response) for response in responses ]))
