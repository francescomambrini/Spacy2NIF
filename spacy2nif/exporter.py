from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS
from spacy.tokens import Doc, Token, Span


class NIFExporter:
    """
    Export a spaCy `Doc` object to an RDF representation following the
    NLP Interchange Format (NIF) 2.1 specification.

    This class is designed to be modular and adaptive: it automatically
    infers which linguistic layers (tokens, POS tags, lemmas, morphology,
    sentences, dependencies, named entities, etc.) are present in a given
    `Doc` and exports only the annotations that are actually available.
    Users can also explicitly control which layers should be included.

    Parameters
    ----------
    base_uri : str, optional
        Base IRI used to generate subject URIs in the output RDF graph.
        Default is "http://example.org/doc#".
    layers : dict or None, optional
        Specification of which linguistic layers to include in the export.
        Keys should be strings such as "tokens", "pos", "lemma", "deps",
        "sents", "ner", etc. Values are booleans. If `None`, the exporter
        will infer the available layers from the `Doc` automatically.
    export_full_text : bool, optional
        if `True` the full text of the Doc will be exported as the 
        `nif:isString` of the context object. If your document is very long, 
        this may end up creating a very big RDF file. 
        Default: `True`.

    Notes
    -----
    - The exporter uses `rdflib` internally to construct a graph.
    - If `layers` is provided, the class will not attempt to infer the
      annotation levels from the `Doc`.
    - This class does not assume a specific spaCy pipeline and works with
      both full and minimal configurations.
    """

    NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
    ITSRDF = Namespace("http://www.w3.org/2005/11/its/rdf#")
    CONLL = Namespace("http://ufal.mff.cuni.cz/conll2009-st/task-description.html#")

    def __init__(self, base_uri: str = "http://example.org/doc#", 
                 layers: dict | None = None,
                 export_full_text: bool = True):
        self.base_uri = base_uri
        self.layers = layers
        self.export_full_text = export_full_text

    def _build_uri(self, el: Doc | Token | Span):
        if isinstance(el, Doc): 
            start = 0
            end = start + len(el.text)
        elif isinstance(el, Token):
            start = el.idx
            end = start + len(el)
        else:
            start = el.start_char
            end = el.end_char
        # uri = URIRef(f"{self.base_uri}char={el.idx},{el.idx + len(el)}")
        uri = URIRef(f"{self.base_uri}char={start},{end}")
        return uri

    def _ensure_layers(self, doc: Doc) -> None:
        """
        Determine the annotation layers available in the given `Doc`.

        This method populates the `self.layers` dictionary either by
        inferring available annotations from the `Doc` using
        `Doc.has_annotation(...)`, or by leaving the dictionary unchanged
        if the user explicitly provided `layers` in the constructor.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            The spaCy document from which to infer available annotation layers.

        Returns
        -------
        None
            The method updates the object's `layers` attribute in place.

        Notes
        -----
        This inference step allows the exporter to operate robustly and
        transparently across pipelines of different complexity.
        """
        if self.layers is not None:
            return

        self.layers = {
            "tokens": True,
            "pos": doc.has_annotation("TAG"),
            "lemma": doc.has_annotation("LEMMA"),
            "morph": doc.has_annotation("MORPH"),
            "deps": doc.has_annotation("DEP"),
            "sents": doc.has_annotation("SENT_START") or doc.has_annotation("DEP"),
            "ner": doc.has_annotation("ENT_IOB"),
        }

    
    def export_doc(self, doc: Doc) -> Graph:
        """
        Export a spaCy `Doc` to an RDF graph in NIF format.

        This function constructs a new `rdflib.Graph` containing all
        triples corresponding to the linguistic layers indicated in
        `self.layers`. If no layer specification was provided, the
        exporter infers the available annotations automatically.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            The spaCy document to export.

        Returns
        -------
        rdflib.Graph
            An RDF graph containing NIF representations of the document
            context, tokens, and any additional annotated layers
            (sentences, POS tags, lemmas, dependencies, named entities,
            etc.) supported by the exporter.

        Example
        -------
        >>> exporter = NIFExporter()
        >>> g = exporter.export_doc(nlp("This is a test."))
        >>> print(g.serialize(format="turtle"))
        """
        self._ensure_layers(doc)
        g = Graph()

        # Export base context
        context_uri = URIRef(f"{self.base_uri}context")
        g.add((context_uri, self.NIF.beginIndex, Literal(0)))
        g.add((context_uri, self.NIF.endIndex, Literal(len(doc.text))))
        
        if self.export_full_text:
            g.add((context_uri, self.NIF.isString, Literal(doc.text)))

        # Export layers
        if self.layers.get("sents"):
            self._export_sentences(doc, g, context_uri)

        if self.layers.get("tokens"):
            self._export_tokens(doc, g, context_uri)

        # if any(self.layers.get(k) for k in ("pos", "lemma", "morph")):
        #     self._export_token_annotations(doc, g)

        # if self.layers.get("deps"):
        #     self._export_dependencies(doc, g)

        if self.layers.get("ner"):
            self._export_entities(doc, g)

        g.bind('nif', str(self.NIF))
        g.bind('conll', str(self.CONLL))

        return g

    
    def _export_tokens(self, doc: Doc, graph: Graph, context_uri: URIRef) -> None:
        """
        Add token-level NIF triples to the RDF graph.

        This includes the basic `nif:Word` units, their character offsets
        within the document (`nif:beginIndex`, `nif:endIndex`), and their
        surface form (`nif:anchorOf`).

        Parameters
        ----------
        doc : spacy.tokens.Doc
            The spaCy document whose tokens will be exported.
        graph : rdflib.Graph
            The RDF graph into which the triples will be added.
        context_uri : rdflib.URIRef
            The URI of the enclosing `nif:Context` resource.

        Returns
        -------
        None
        """
        prev_tok_uri = None
        for token in doc:
            if token.is_space:
                continue

            uri = URIRef(f"{self.base_uri}char={token.idx},{token.idx + len(token)}")
            
            if prev_tok_uri:
                graph.add((prev_tok_uri, self.NIF.nextWord, uri))
            
            graph.add((uri, self.NIF.anchorOf, Literal(token.text)))
            graph.add((uri, self.NIF.beginIndex, Literal(token.idx)))
            graph.add((uri, self.NIF.endIndex, Literal(token.idx + len(token))))
            graph.add((uri, self.NIF.referenceContext, context_uri))
            graph.add((uri, RDF.type, self.NIF.Word))
            prev_tok_uri = uri

            # Token annotation
            if 'lemma' in self.layers:
                graph.add((uri, self.NIF.lemma, Literal(token.lemma_)))
            if 'pos' in self.layers:
                graph.add((uri, self.NIF.posTag, Literal(token.pos_)))
            if 'morph' in self.layers and token.morph:
                # TO DO: how do we express morph feats?
                # one option is to use conll:FEATS as in `conll-rdf`
                graph.add((uri, self.CONLL.FEATS, Literal(token.morph)))

            # Dependency Syntax
            # TO DO! Check
            if 'deps' in self.layers:
                htok = token.head 
                huri = self._build_uri(htok)
                graph.add((uri, self.CONLL.HEAD, huri))
                graph.add((uri, self.NIF.dependencyRelationType, Literal(token.dep_)))


    def _export_sentences(self, doc: Doc, graph: Graph, context_uri: URIRef) -> None:
        """
        Export sentence-level annotations as `nif:Sentence` units.

        Each sentence span is represented using character offsets and is
        linked to the document context.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            The spaCy document whose sentences will be exported.
        graph : rdflib.Graph
            The RDF graph being constructed.
        context_uri : rdflib.URIRef
            The URI of the document's NIF context.

        Returns
        -------
        None
        """
        prev_sent_uri = None
        for sent in doc.sents:
            # ignore white-space only sentences
            if all(t.is_space for t in sent):
                continue
            uri = URIRef(f"{self.base_uri}char={sent.start_char},{sent.end_char}")
            if prev_sent_uri:
                graph.add((prev_sent_uri, self.NIF.nextSentence, uri))
            graph.add((uri, self.NIF.beginIndex, Literal(sent.start_char)))
            graph.add((uri, self.NIF.endIndex, Literal(sent.end_char)))
            graph.add((uri, self.NIF.anchorOf, Literal(sent.text)))
            graph.add((uri, self.NIF.referenceContext, context_uri))
            graph.add((uri, RDF.type, self.NIF.Sentence))

            # Unfortunately, we can't use sent.is_sent_end and is_sent_start reliably
            # In many sentences, whitespace tokens are sent_end and this won't do!
            toks = [t for t in sent if not t.is_space]

            # double-check that the list is ordered correctly; unnecessary, but you know...
            toks.sort(key=lambda t: t.i)
            
            # Let's add first and last word in the sentence
            first_tok_uri = self._build_uri(toks[0])
            last_tok_uri = self._build_uri(toks[-1])
            graph.add((uri, self.NIF.firstWord, first_tok_uri))
            graph.add((uri, self.NIF.lastWord, last_tok_uri))

            for tok in toks:
                tok_uri = URIRef(f"{self.base_uri}char={tok.idx},{tok.idx + len(tok)}")
                graph.add((tok_uri, self.NIF.sentence, uri))
                if tok.is_sent_start:
                    graph.add((uri, self.NIF.firstWord, tok_uri))
                if tok.is_sent_end:
                    graph.add((uri, self.NIF.lastWord, tok_uri))
            prev_sent_uri = uri


    def _export_dependencies(self, doc: Doc, graph: Graph) -> None:
        """
        Export syntactic dependencies as RDF triples.

        This method represents dependency relations between tokens,
        including dependency labels and head–child links. The exact RDF
        properties used may depend on the chosen NIF extensions.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            The parsed spaCy document.
        graph : rdflib.Graph
            The RDF graph to populate with dependency triples.

        Returns
        -------
        None
        """
        # TO DO: Implementation left for the future
        # at the present it is all done at the token level with nif:dependency
        pass

    # ---------------------------------------------------------------------
    # Entity export
    # ---------------------------------------------------------------------
    def _export_entities(self, doc: Doc, graph: Graph) -> None:
        """
        Export named entities according to the NIF representation.

        Entities are encoded as `nif:Phrase` (or an alternative class,
        depending on the ontology extensions you adopt). Each entity is
        represented with offsets, its surface form, and its entity label.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            The spaCy document containing named entities.
        graph : rdflib.Graph
            The RDF graph to receive the entity triples.

        Returns
        -------
        None
        """
        for e in doc.ents:
            uri = self._build_uri(e)
            graph.add((uri, RDF.type, self.NIF.Span))
            graph.add((uri, RDF.type, self.NIF.EntityOccurrence))
            graph.add((uri, self.NIF.literalAnnotation, Literal(e.label_)))
            graph.add((uri, self.NIF.beginIndex, Literal(e.start_char)))
            graph.add((uri, self.NIF.endIndex, Literal(e.end_char)))
            graph.add((uri, self.NIF.anchorOf, Literal(e.text)))

            if len(e) > 1:
                for t in e:
                    turi = self._build_uri(t)
                    graph.add((turi, self.NIF.subString, uri))


