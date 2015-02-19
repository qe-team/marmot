### Feature Extractors

Feature extractors take a context object as input, and output a list of values. For NLP tasks, feature values are typically scalars or strings. If your feature values are not scalars, it's likely that you'll need to do something like binarization to convert them to numbers, in order to learn models with them.  

The features that are extracted for a particular experiment should be configured in the Marmot experiment config.

