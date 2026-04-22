from FlagEmbedding import BGEM3FlagModel

class BGEEmbedder:
    def __init__(self, model_path):
        self.model = BGEM3FlagModel(model_path, use_fp16=True, show_progress_bar=False)

    def encode(self, text):
        query_encoded_result = self.model.encode(
            [text],
        )

        # query_embedding = np.array(query_embedding)
        query_embedding = query_encoded_result['dense_vecs']

        return query_embedding[0]