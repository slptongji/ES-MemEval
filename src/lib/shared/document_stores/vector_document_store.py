from typing import Iterable, Literal, Sequence
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma, VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from lib.shared.document_stores.document_store import DocumentStore


class VectorDocumentStore(DocumentStore):
    def __init__(self, candidate_size: int, store: Literal["FAISS", "Chroma"], embeddings: Embeddings):
        super().__init__(candidate_size)

        self.embeddings: Embeddings = embeddings
        self.store: VectorStore | None = None
        self.store_type: Literal["FAISS", "Chroma"] = store
        self.ids = []

    def extend(self, documents: Iterable[Document]):
        if not isinstance(documents, list):
            documents = list(documents)

        if self.store is None:
            if self.store_type == "FAISS":
                self.store = FAISS.from_documents(documents, self.embeddings)
                return
            elif self.store_type == "Chroma":
                self.store = Chroma.from_documents(documents, self.embeddings)
                return
            else:
                raise ValueError(f"Unsupported vector store type '{self.store_type}'.")
            
        self.ids.extend(self.store.add_documents(documents))

    def _retrieve(self, request: str):
        if self.store is None:
            return []
        retriever: VectorStoreRetriever = self.store.as_retriever(search_kwargs={
            "k": self.candidate_size()
        })
        return retriever.invoke(request)
    
    def all_documents(self):
        if self.store is None:
            return []
        return self.store.get_by_ids(self.ids)


def _test():
    store: VectorDocumentStore = VectorDocumentStore(2, "FAISS", 
                                                     HuggingFaceEmbeddings(model_name="BAAI/bge-m3", 
                                                                           model_kwargs={"device": "cpu"}))
    store.extend([
        Document("Hi, AI companion. Today I went for a walk in the park and saw some beautiful scenery. I'd love to share it with you."),
        Document("Great! Which park did you go to and what interesting scenery did you see?"),
        Document("I went to Green Meadow Park and saw a particularly beautiful blooming cherry blossom and a super cute squirrel!"),
        Document("Awesome! Hearing you describe it makes me want to go for a walk in the park too!"),
    ])
    store.extend([
        Document("Hello AI companion. Today is a beautiful day. I went to see a movie with my friend and had a wonderful time."),
        Document("That's great! What movie did you see and did you like it?"),
        Document("TWe saw an art film called Taxi Driver. I found it very interesting. The visuals were beautiful and the theme was very thought-provoking."),
        Document("Sounds like it was a good film. What was the main theme of the movie?"),
    ])
    print(store.retrieve("What film did she see?"))


if __name__ == "__main__":
    _test()

