# a corpus_context_creator gets its contexts from a corpus of instances
# the list of contexts presumably come from the parser for this particular corpus

from collections import defaultdict
from context_creator import ContextCreator

class CorpusContextCreator(ContextCreator):

    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    def __init__(self, all_contexts, max_instances=10000):
        self.context_map = defaultdict(list)
        for context in all_contexts:
            if max_instances is not None:
                if len(self.context_map[context['token']]) <= max_instances:
                    self.context_map[context['token']].append(context)
            else:
                self.context_map[context['token']].append(context)


    def get_contexts(self, token, max_size=None):
        if max_size is not None:
            return self.context_map[token][:max_size]
        else:
            return self.context_map[token]




