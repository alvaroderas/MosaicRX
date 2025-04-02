from sklearn.feature_extraction.text import TfidfVectorizer

class Embeddings(TfidfVectorizer):
    """
    Embeddings class that extends TfidfVectorizer for processing text data.
    """
    def __init__(self, documents, max_df=0.80, min_df=5, max_features=5000, stop_words='english'):
        super().__init__(max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df)
        self.documents = documents  # Store documents for later use if needed
        self.term_document_matrix = self.generate_term_doc_matrix(documents)

    def get_term_document_matrix(self):
        return self.term_document_matrix

    def generate_term_doc_matrix(self, documents):
        if 'uses' not in documents:
            raise KeyError("The 'uses' column is missing from the input documents.")
        ## naive approach, need to "learn" a better tokenization rep that has higher weights to
        # key sickness words i.e cold
        return self.fit_transform(documents['uses'].fillna(""))




    print()
