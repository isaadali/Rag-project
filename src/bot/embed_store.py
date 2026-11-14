from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import os


class Embedding_Model:
    def __init__(self,model_name='all-MiniLM-L6-v2'):
        self.model_name=model_name
        self.model=None
        self._load_model()
    def _load_model(self):
        print(f"Loading model {self.model_name}")
        try:
            self.model=SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Exception {e}")
            raise        

    def generate_embeddings(self,chunks):
        if not self.model:
            raise ValueError("No embedding model")

        encodings=self.model.encode(chunks,show_progress_bar=True)
        return encodings

class Vector_store:
    def __init__(self,collection_name='pdf_documents',persist_directory='./data/vector_store'):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._initialize_vector_store()
    def  _initialize_vector_store(self):
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"{self.collection_name} embeddings for Rag ", "hnsw:space": "cosine"}
            )
        except Exception as e :
            print("Can't initialize vector store")
            raise

    def add_documents(self,documents,embeddings):
        if len(documents)!= len(embeddings):
            raise ValueError("len of documents not equal to length of embeddings")
        ids=[]
        metadatas=[]
        document_text=[]
        embeddings_list=[]
        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            #doc_id = f"{doc.metadata.get('source_file')}_{doc.metadata.get('page')}_{i}"
            source = doc.metadata.get('source_file') or doc.metadata.get('file_name')
            doc_id = f"{source}_{doc.metadata.get('page', 0)}_{i}"
            ids.append(doc_id)

            #metadata
            metadata=dict(doc.metadata)
            metadata["index"]=i
            metadata["context_length"]=len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=document_text
                                    )       
            print("Docs added successfully")
        except Exception as e:
            raise ValueError("Docs not added successfully")


        
def generate_plus_store(text,docs):
    model=Embedding_Model()
    embeddings=model.generate_embeddings(text)
    store=Vector_store()
    store.add_documents(docs,embeddings)
            

