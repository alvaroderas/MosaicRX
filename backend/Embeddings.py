from sklearn.feature_extraction.text import TfidfVectorizer
from DataLoader import DataLoader

class Embeddings(TfidfVectorizer):
    """
    Embeddings class that extends TfidfVectorizer for processing text data.
    """
    def __init__(self, dataloader, max_df=1.0, min_df=1, max_features=5000, stop_words='english', norm='l2'):
        super().__init__(max_features=max_features, stop_words=stop_words, max_df=max_df, min_df=min_df, norm=norm)
        self.dataloader = dataloader
        self.term_document_matrix = self.__generate_term_doc_matrix__()
   
    def __generate_term_doc_matrix__(self):
        fields = self.dataloader.dataset_headers
        isvalid_field = lambda field: field != "Image URL" and field != "Excellent Review %" and field != "Average Review %" and field != "Poor Review %"
        
        dataset = []
        for data_instance in self.dataloader.dataset:
            current_doc = []
            for field in fields:
                if isvalid_field(field):
                    current_doc.append(data_instance.get(field))
            current_doc = "".join(current_doc)
            dataset.append(current_doc)
        return self.fit_transform(dataset)
    
    def svd_embeds(self):
    
        pass  # Implement SVD logic here
    