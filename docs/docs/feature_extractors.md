### Feature Extractors

Feature extractors take a context object as input, and output a list of values. For NLP tasks, feature values are typically scalars or strings. If your feature values are not scalars, it's likely that you'll need to do something like binarization to convert them to numbers, in order to learn models with them.  

The features that are extracted for a particular experiment should be configured in the Marmot experiment config.

#### Interface

All feature extractors have two methods:

* get_features(self, context_obj) -- extracts one or more features for a given context_obj. Throws an error if the context object does not have any of the required fields.

* get_feature_names(self) -- returns names of features returned by get_features.

The methods have the same parameters for all feature extractors. So they can be called uniformly for a list of feature extractors, none of them requires special handling:

	features = []
	for f_e in feature_extractors:
		features.extend(f_e.get_features(context_obj))

#### Available feature extractors

* FeatureExtractor -- the abstract feature extractor class. All feature extractors should inherit from it.

* AlignmentFeatureExtractor -- extracts the alignments. Can either extract the alignments from context object or align source and target sentence on the fly. For the alignment it can use a pre-trained model or train a model itself if the training data is provided.

Features extracted:

	* source word aligned to the target token. If the target token is aligned to multiple words, they are concatenated.
	* the left context of the source word aligned to the target token. In case of multiple alignments the context of the leftmost aligned source word is returned.
	* the right context of the source word aligned to the target token. In case of multiple alignments the context of the rightmost aligned source word is returned.

* DictionaryFeatureExtractor -- extracts a set of binary features indicating if a token belongs (1) or does not belong (0) to some closed class.

Features extracted:

	* token is a stopword (1 - is a stopword, 0 - is not as stopword)
	* token is punctuation
	* token is a proper noun
	* token is a number

* GoogleTranslateFeatureExtractor -- extracts the automatic translation by Google Translate. Looks for the representation "pseudo-reference" in the context object, if it is missing, performs the translation on the fly. In this case the "source" field has to exist in the context object.

Features extracted:

	* binary feature indicating whether the token occurs or does not occur in the pseudo-reference

* LMFeatureExtractor -- extracts the LM features. 

Features extracted:

	* length of the longest sequence of left context of the token that occurs in the LM.
	* length of the longest sequence of right context of the token that occurs in the LM.
	* backoff behaviour features as described in [Raybaud et al. 2011] (http://perso.crans.org/raybaud/cm-springer-utf8.pdf):
		* for ngram w<sub></sub> w<sub></sub> w<sub></sub>
	

* POSFeatureExtractor

* SourceLMFeatureExtractor

* TokenCountFeatureExtractor

* WordnetFeatureExtractor
