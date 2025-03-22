from sklearn.feature_extraction.text import TfidfVectorizer


class Embeddings(TfidfVectorizer):
    """
    Embeddings class that extends TfidfVectorizer for processing text data.
    """
    def __init__(self, data_loader, max_df=1.0, min_df=1, max_features=5000, stop_words='english', norm='l2'):
        super().__init__(max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df, norm=norm)
        self.data_loader = data_loader
        self.term_document_matrix = self.__generate_term_doc_matrix__()
   
    def __generate_term_doc_matrix__(self):
        fields = self.data_loader.dataset_headers()
        
        td_mat = []
        data_instances = self.data_loader.dataset()
        for data_instance in data_instances:
            current_doc = []
            for field in fields:
                current_doc.append(data_instance.get(field))
            current_doc = "".join(current_doc)
            td_mat.append(current_doc)
        return self.fit_transform(td_mat)
    
    def svd_embeds(self):
    
        pass  # Implement SVD logic here
    