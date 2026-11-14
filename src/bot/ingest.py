from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,WebBaseLoader,YoutubeLoader
from pathlib import Path
#from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
def _load_documents(data_dir): 
    data_path=Path(data_dir).resolve()
    print(f"[decoded] {data_path}")
    documents=[]

    ## pdf files
    pdf_files=list(data_path.glob("**/*.pdf"))
    if pdf_files:
     print("pdf files detected")
     for file in pdf_files:
        print(f"loading {file.name}")
        try:
            loader=PyPDFLoader(file)
            docs =loader.load()
            for doc in docs:
                doc.metadata["source_file"]=file.name
                doc.metadata["file_type"]='pdf'
            documents.extend(docs)
        except Exception as e:
            print(f"Exception {e}") 

    #word files
    word_files=list(data_path.glob("**/*.docx"))
    if word_files:
     print("word files detected")
     for file in word_files:
        try:
            loader=Docx2txtLoader(file)
            docs=loader.load()
            for doc in docs:
                doc.metadata['file_name']=file.name
                doc.metadata["file_type"]='word'
            documents.extend(docs)
        except Exception as e:
                print(f"Exception {e}") 
                                    
    return documents

def _load_web(url):
    loader=WebBaseLoader(url)
    docs=loader.load()
    return docs

def chunking(data_dir,chunking_size=1000,chunking_overlap=200):
    docs=_load_documents(data_dir)
    splitter=RecursiveCharacterTextSplitter(
        chunk_overlap=chunking_overlap,
        chunk_size=chunking_size,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs=splitter.split_documents(docs)
    return split_docs
def web_chunking(url,chunking_size=1000,chunking_overlap=200):
        docs=_load_web(url)
        splitter=RecursiveCharacterTextSplitter(
            chunk_overlap=chunking_overlap,
            chunk_size=chunking_size,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs=splitter.split_documents(docs)
        return split_docs


