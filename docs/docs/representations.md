### Representations

A representation is exactly what it sounds like -- a way of representing your data. For many NLP tasks, a model must be learned before the data can be mapped into a new representation. This applies to parsing, POS tagging, chunking, alignment, etc...     
       
It often makes more sense to compute representations offline -- i.e. not when we are trying to learn our models, because computing the representation at feature extraction time can make things much slower. For this reason, we introduce the concept of _representation generators_. A representation generator adds some data to your context objects, data that will be used to compute some feature values later in the experiment pipeline. 

#### Internal representation

Representation generators work with the internal representation of the data. Internal representation is a Python dictionary where keys are names of representations ("target", "source", "tags", etc.) and values are lists of representations for each sentences. In the case of target and source representations are lists of target and source words, respectively, but other formats of representations can exist.

All representation generators return internal representations. Some of generators take representations as input, too, and return their extended versions. Others take data files as input.

#### Interface

The only method all representation generators have is _generate_. It takes an internal representation or nothing, and returns the same internal representation with one additional field which is a new representation of the data.

#### List of the available representation generators:

* RepresentationGenerator -- abstract class, all representation generators should inherit from them in order to have the same interface.

* WMTRepresentationGenerator -- the representation generator that converts the data in the format used for [WMT word-level QE task] (http://statmt.org/wmt14/quality-estimation-task.html) into the internal representation. It requires two files: the target file in one-word-per-line format annotated with errors and the source file with source sentences and sentence IDs. It returns an internal representation {'target': [target sentences], 'source': [source sentences], 'tags': [tags for all sentences]}.

* WordQERepresentationGenerator -- the representation generator that converts source, target and tags represented in plain-text files into the internal representation.
	__init__(self, source_file, target_file, tags_file):
	* source_file -- source
	* target_file -- target
	* tags_file -- tags
All files need to be sentence-aligned and tokenized.

* POSRepresentationGenerator -- the representation generator that runs Tree-Tagger for the chosen language ('target' or 'source').
	__init__(self, tagger, parameters, data_label, tmp_dir=None)
	Parameters:
	* path to Tree-Tagger tagging script 
	* path to Tree-Tagger parameters file 
	* label of the representation to be tagged (it will most probably be 'target' or 'source', but other representations can be tagged as well, if they are natural language sentences).
	* tmp_dir (optional) -- directory for storing temporary files. If not defined, a directory 'tmp_dir' will be created in the working directory.

	generate(self, data_obj)
	Parameters:
	* data_obj -- an internal representation to POS-tag.

* AlignmentRepresentationGenerator -- generates the alignments of the source and the target sentences.
	__init__(self, align_model=None, src_file=None, tg_file=None, tmp_dir=None)
	* align_model (optional) -- an alignment model trained with [fast-align] (http://www.cdec-decoder.org/guide/fast_align.html). <align_model> should be a string such that files <align_model>.fwd_err, <align_model>.rev_err, <align_model>.fwd_params, <align_model>.rev_params exist.
	* src_file (optional) -- source file to train the alignment model.
	* tg_file (optional) -- target file to train the alignment model. The source and the target files should be sentence-aligned. If the pre-trained alignment model is not defined, both source and target training data have to be defined. Training of an alignment model can take several hours, so it is preferable to train a model in advance.
	* tmp_dir (optional) -- a temporary directory where a trained alignment model is stored.

        generate(self, data_obj)
        Parameters:
        * data_obj -- an internal representation. Fields data_obj['target'] and data_obj['source'] should exist and contain the same number of sentences.

* GoogleTranslateRepresentationGenerator -- generates the automatic translations of the source sentences using Google Translate tool. This generator requires the Internet connection. The generation may fail due to network errors.
	__init__(self, lang='en')
	* lang (optional) -- target language

        generate(self, data_obj)
        * data_obj -- an internal representation. Field data_obj['source'] should exist.

