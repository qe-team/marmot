<img src="http://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Marmot-edit1.jpg/1024px-Marmot-edit1.jpg" alt="Marmot Feature Extraction and Machine Learning" style="width: 200px;"/>
<center> <h1>Welcome to Marmot</h1> </center>
<!-- ![Marmot Picture](http://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Marmot-edit1.jpg/1024px-Marmot-edit1.jpg) -->

## Components and Design Philosophy

### QEObj and NLPObj - The QE and NLP Data JSON Exchange Format:
* We propose the following object representation for exchanging QE data (and for the general exchange of NLP data of all kinds).
 `{ 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}`
Any of the fields in a QEObj may be missing (i.e. may be null), so the components that use them should ensure that the objects they consume have the neccessary fields populated. Of course, you are free to add fields, but it may limit the extensibility of your code.

We allow all fields to be nullable to account for the reality of missing or unnecessary data, and to give maximum flexibility to implementations. However, this design decision means that feature extractors must ensure that a QEObj has the fields that the feature extractor requires.

Feature Extractors should throw an error when the data they need is missing.
<!-- insert an example of how a feature extractor should fail if it gets called with a contextObj that has missing fields -->

#### Testing Philosophy

We test everything. Please do not submit a pull request without including tests that show that the code works. You can see some example tests [here]().


### Positive Contexts ('Good' Instances):
<options for creating positive contexts>

### Negative Contexts ('Good' Instances):
<current options for creating negative contexts>

### Parsing data

### Evaluation
<Evaluation options>


### test link
[another page](another_page)


## Commands

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
