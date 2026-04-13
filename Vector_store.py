
import os
import uuid
import chromadb
import numpy as np
from typing import List, Any


class Vector_Store:
    def __init__(self, collection_name: str = 'TruthGate', persist_directory:str = "TruthGate_db/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store() 
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={'description':'Embeddings of text collected  By Api while LLM Observation and Toughts'}
            )
        except Exception as e:
            print(f"Error initializing vector store:{e}")
            raise
    def add_document(self, documents:List[Any], embeddings:np.ndarray):
        if len(documents)!= len(embeddings):
            raise ValueError("Number of documents and embeddings must be same")
        print(f"Adding {len(documents)} documents to vector store")

        ids = [] 
        metadatas = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f'doc_{uuid.uuid4().hex[:8]}_{i}'
            ids.append(doc_id)
            metadata = dict(doc.metadata) 
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_content.append(doc.page_content)
            
            embeddings_list.append(embedding.tolist())
        try:
            self.collection.add(
                ids = ids,
                metadatas = metadatas,
                embeddings = embeddings_list,
                documents = documents_content
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f'Total documents in collection:{self.collection.count()}')

        except Exception as e:
            print(f"Error adding documents to vector store:{e}")
            raise
vector_db  = Vector_Store()