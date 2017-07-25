from gensim.models import word2vec
import numpy as np


class UtteranceEmbed():

    def __init__(self, fname='/home/bruce/med250.model.bin', dim=250):
        self.dim = dim
        try:
            # load saved model
            self.model = word2vec.Word2Vec.load(fname)
        except:
            print(':: creating new word2vec model')
            self.create_model()
            self.model = word2vec.Word2Vec.load(fname)

    def encode(self, utterance):
        embs = [ self.model[word] for word in utterance.split(' ') if word and word in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim],np.float32)

    def create_model(self, fname='text8'):
        sentences = word2vec.Text8Corpus('/home/bruce/work/hcn/data/text8')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        model.save('/home/bruce/work/hcn/data/text8.model')
        print(':: model saved to data/text8.model')
