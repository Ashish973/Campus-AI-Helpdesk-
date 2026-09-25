from importlib.resources import path
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")


def load_documents(directory_path=KB_PATH):
    dir_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False
    )
    return dir_loader.load()

## split documents and create chunks..

def split_documents(documents , chunk_size=1000,chunk_overlap=200):
    """ split documents into smaller chunks for better RAg performance""" 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Total Chunks created : {len(split_docs)}")

    

    return split_docs

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks.")



