from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import getpass
import os 

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
load_dotenv()
#load pdf form the file path
file_path = r"C:\Users\ahmad\OneDrive\Desktop\story_.pdf"
loader = PyPDFLoader(file_path)
docs=loader.load()
print(f"there is a  {len(docs)} documents.")
 
#split documents vai RecursiveCharacterTextSplitter function 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   
    chunk_overlap=200,  
    add_start_index=True,   
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

#call the embeding vai api 

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#intaite vectore store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./first_Aid_embeddings",  
)
#Add the splited documents to the vector store
id=vector_store.add_documents(all_splits)
