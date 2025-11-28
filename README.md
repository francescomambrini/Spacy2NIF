# Spacy2NIF

Convert spaCy Doc objects to [RDF/NIF 2.1](https://github.com/NLP2RDF/ontologies/blob/master/nif-core/nif-core.ttl).

## Installation

You can install it using `pip`, but keep in mind that it is still **very** 
experimental!

```bash
pip install git+https://github.com/francescomambrini/Spacy2NIF.git
```

Please report any issue/suggestion [here](https://github.com/francescomambrini/Spacy2NIF/issues).

## Examples

Minimal usage example:

```python
from spacy2nif.exporter import NIFExporter
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Barack Obama visited Paris.")

exporter = NIFExporter()
g = exporter.export_doc(doc)

print(g.serialize(format="turtle"))
```

For other examples, see the Jupyter notebook in the [example](https://github.com/francescomambrini/Spacy2NIF/tree/main/examples) folder.
