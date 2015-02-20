### Configuration

We provide an example pipeline which performs the data preparation, feature extraction, model training, tagging of the test data and evaluation of result. User does not have to use this pipeline or the configuration file format described below, the functions and classes provided in Marmot can be used any other way. However, the tool has a set of utilities designed to organise the experiment and a config file format that helps to use them efficiently.

The config file is written in [YAML format] (http://www.yaml.org/spec/1.2/spec.html). A YAML document can be parsed by the Python yaml module into a dictionary object.

The config file not only contains the paths to data files or external tools, it defines the whole pipeline. It has several blocks that correspond to experiment stages.

The keyword __module__ used in the config is designed for loading and creating Python objects.


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

An argument can 

## Variables

There is a set of variables that should be declared in the config file:

* workers -- the number of workers that Marmot can use, defaults to 1.

* tmp_dir -- the directory to store temporary files produced by the script, default is /script_dir/tmp_dir.

* datasets -- the list of datasets used in the experiment. The list has to contain sublists each of which stands for a specific part of data (training, test, development, etc.). The "training" and "test" sections are compulsory.

* contexts -- the type of organisation of data, possible values are "plain", "token", and "sequential". If
