class Retriever:
    def __init__(self,vector_store,embedding_manager):
        self.vector_store=vector_store
        self.embedding_manager=embedding_manager

    def retrieve(self,query,top_k: int=3,score_threshold:float=0.5):
        query_embedding=self.embedding_manager.generate_embeddings(query)
        try:
            results=self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()] ,
                n_results=top_k
            )
            retreived_docs=[]
            if results['documents'] and results['documents'][0]:
                documents=results['documents'][0]
                distances=results['distances'][0]
                metadatas=results['metadatas'][0]
                ids=results['ids'][0]
                
                for i ,(doc_id, doc,metadata,dist) in enumerate(zip(ids,documents,metadatas,distances)):
                    similarity_score=1-dist
                    if    similarity_score>score_threshold:
                        retreived_docs.append({'id':doc_id,'doc':doc,'metadata':metadata,'dist':dist})
            else:
                print('no doc')
            return retreived_docs    

        except Exception as e:
            print(f"{e}")    
            return []


