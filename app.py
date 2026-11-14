from src.bot.ingest import chunking
from src.bot.embed_store import Embedding_Model,Vector_store
from src.bot.retriever import Retriever
if __name__=="__main__":
    embedding_model=Embedding_Model()
    vector_store=Vector_store() 
    chunks=chunking("data")
    print(chunks[0].page_content)   
    text=[doc.page_content for doc in chunks]
    embeddings= embedding_model.generate_embeddings(text)
   
    vector_store.add_documents(chunks,embeddings)
    rag_retriever=Retriever(vector_store,embedding_model)
    while True:
        query =input("Write query")
        if query.lower()=='quit':
            break
        f= rag_retriever.retrieve(query)
        print("Number of results:", len(f))
        for i in f:
                print(i['id'])
                print(i['doc'])
    print('allah hafiz')
                
