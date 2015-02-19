### Representations

A representation is exactly what it sounds like -- a way of representing your data. For many NLP tasks, a model must be learned before the data can be mapped into a new representation. This applies to parsing, POS tagging, chunking, alignment, etc...     
       
It often makes more sense to compute representations offline -- i.e. not when we are trying to learn our models, because computing the representation at feature extraction time can make things much slower. For this reason, we introduce the concept of _representation generators_. A representation generator adds some data to your context objects, data that will be used to compute some feature values later in the experiment pipeline. 
