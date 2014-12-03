import random
from context_creator import ContextCreator


# returns a random TARGET context for the wordset and parameters supplied to the constructor
class RandomContextCreator(ContextCreator):
    def __init__(self, word_list, num_contexts=5000, length_bounds=[6,12], tagset=set([0])):
        self.word_list = set(word_list)
        self.num_contexts = num_contexts
        self.length_bounds = length_bounds
        self.tagset = set(tagset)

    def get_contexts(self, token):
        return [self.build_context_obj(token) for i in range(self.num_contexts)]

    def random_context(self):
        rand_length = random.randint(self.length_bounds[0],self.length_bounds[1])
        # casting the set to a tuple makes this faster apparently
        rand_words = [random.choice(tuple(self.word_list)) for i in range(rand_length)]
        return rand_words

    def build_context_obj(self, token):
        rand_context = self.random_context()
        # get the index of the token after we know the length of the random context
        rand_idx = random.randint(0, len(rand_context)-1)
        # substitute that index for our token
        rand_context[rand_idx] = token
        # casting the set to a tuple makes this faster apparently
        random_tag = random.choice(tuple(self.tagset))
        # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
        new_obj = { 'token': token, 'index': rand_idx, 'target': rand_context, 'tag': random_tag }
        return new_obj



