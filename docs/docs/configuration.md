### Configuration

We provide an example pipeline which performs the data preparation, feature extraction, model training, tagging of the test data and evaluation of result. User does not have to use this pipeline or the configuration file format described below, the functions and classes provided in Marmot can be used any other way. However, the tool has a set of utilities designed to organise the experiment and a config file format that helps to use them efficiently.

The config file is written in [YAML format] (http://www.yaml.org/spec/1.2/spec.html). A YAML document can be parsed by the Python yaml module into a dictionary object.

The config file not only contains the paths to data files or external tools, it defines the whole pipeline. It has several blocks that correspond to experiment stages:

* __datasets__ block defines the data files to load and classes that should be used for that.

* __representations__ block defines the set of objects that create the additional representations of the loaded data (POS tags, alignments).

* __feature_extractors__ block defines the set of feature extractors.

All objects in these blocks are defined with the keyword __module__.

## Declaring a module

The declaration of a module means that the script will load a Python class and create an instance of this class with the provided parameters.
The keyword __module__ is the path to the Python class. It can belong to any Python library and does not need to be in the Marmot directory.
The keyword __args__ is the list of arguments that are needed for this class initialisation. 
	
	  -- module: marmot.representations.pos_representation_generator.POSRepresentationGenerator
         args:
           -- tiny_test/tree-tagger
           -- tiny_test/english-utf8.par
           -- 'source'
           -- tiny_test/tmp_dir

Code in the example above will generate the instance of POSRepresentationGenerator which is located in /MARMOT_HOME/marmot/representations.pos_representation_generator.py. This declaration is equivalent to the following Python code:

	from marmot.representations.pos_representation_generator import POSRepresentationGenerator
	r = POSRepresentationGenerator('tiny_test/tree-tagger', 'tiny_test/english-utf8.par', 'source', 'tiny_test/tmp_dir')

The __args__ format does not support the keyword arguments, so all the arguments should be declared in the order they appear in the __init__ function of the class. However, the keyword arguments can be omitted. For example, in order to create the LogisticRegression classifier object with the penalty 'l1' you need to write the following code in the config:

	-- module: sklearn.linear_model.LogisticRegression
	   args:
		 - 'l1'

Compare this notation with the full list of parameters:

	sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

If the initialisation has no parameters, no __args__ keyword is needed:

	module: sklearn.linear_model.LogisticRegression

## Declaring a function

An argument of a module can be the output of a function. A function is defined analogously to a class. If an argument has the field __type__ with the value __function_output__, the parser will look for the fields with keys __func__ and __args__. __func__ is the path to a Python function and __args__ is a set of arguments this function requires.

The declaration of a function can be recursive, i.e. its arguments can be outputs of a function as well. In this case the parser will go through the whole function tree calling the functions and returning their values where appropriate.

## Datasets

This block contains the datasets that should be used for the experiment and the objects that .
The block has to contain sublists each of which stands for a specific part of data (training, test, development, etc.). The "training" and "test" sections are compulsory in the example experiment run provided. Each section needs to have the following format:

	training:
		- module: marmot.representations.wmt_representation_generator.WMTRepresentationGenerator
          args:
            - tiny_test/EN_ES.tgt_ann.train
            - tiny_test/EN_ES.source.train
            - tiny_test/tmp_dir
            - False

Several modules can be declared in each section. There can be any number of sections.

If there is no representation generator for the data format you want to use, you should create a new appropriate representation generator.

## Representation Generators

In this section the additional representation generators are declared. The declaration uses the same __module__ notation.

While the generators listed in the __datasets__ block should take data files as inputs and return the internal representation of the data, the additional generators take the internal representations and extend them.

These representation generators are applied to all datasets declared in __datasets__ block. 

All generators declared in this section have to use the data generated earlier by the generators from __datasets__ block. So all additional generators have pre-requisites: they need a particular data field to generate a representation (e.g. AlignmentDataGenerator needs both source and target exist). If there is no needed datafield, the generator will throw an error.

## Feature extractors

The feature extractors are declared using the __module__ notation. Every extractor takes a context object (an object that contains a training instance and all information about it) and extracts one or more features from it. Like representation generators, they look for particular data fields in the context object and throw an error if they are not able to find it. Some data fields can be generated on the fly -- e.g. if there is no part-of-speech tagging for a given example, but the POSFeatureExtractor is given a POS-tagger and parameters for it, it will generate the tagging and extract the POS features. However, this is usually inefficient. First of all, the features are extracted for each word, whereas the additional representations are usually generated for the whole sentence. Therefore, if a sentence contains 20 words, it will be tagged 20 times, if the POSFeatureExtractor is declared and no POS representation was generated in advance. Secondly, when calling the external tools much time is spent on the call operation itself, so calling Tree-Tagger once for 100 sentences (as it is done in a representation generator) is much faster than calling it 100 times for one sentence (which a feature extractor will do).

So acquiring all the needed representations in advance (by declaring relevant representation generators) will save much time.

## List of Variables

There is a set of variables that should be declared in the config file:

* workers -- the number of workers that Marmot can use, defaults to 1.

* tmp_dir -- the directory to store temporary files produced by the script, default is /script_dir/tmp_dir.

* datasets -- the list of datasets used in the experiment.

* contexts -- the type of organisation of data, possible values are __plain__, __token__, and __sequential__. If set to __plain__, the examples are organised into a flat list for the training of one classifier. If set to __token__, the data is stored in a list of lists, where each list is a set of examples for one specific word (i.e. it will contain a set of training examples for the word "he", another set for the word "it", etc.). In this case a separate classifier for every token will be trained. __sequential__ means that the data will be stored in a list of sequences (list of lists of examples) to train a sequence labelling model.

* filters -- settings to filter out some training examples. 3 filters currently exist (all active only in __token__ mode):

	* min_count -- minimum number of training instances per token

	* min_label_count -- minimum number of training instances of each label per token

	* proportion -- maximum possible ratio of counts of "BAD" and "GOOD" labels in the list of training instances.

* feature_extractors -- the list of feature extractors.

* learning -- the learning model to use. Has to contain either a field __classifier__ or __sequence_labeller__. Both need to be defined as modules.
