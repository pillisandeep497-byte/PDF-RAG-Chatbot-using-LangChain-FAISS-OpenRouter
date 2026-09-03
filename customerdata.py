from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os 


load_dotenv()

loader = PyPDFLoader("RAG/sample2.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunk = splitter.split_documents(documents)

embedding=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


vectore_store=FAISS.from_documents(
    documents=chunk,
    embedding=embedding
)
retriever=vectore_store.as_retriever(
    search_kwargs={"k":15}
)


prompt=ChatPromptTemplate.from_template(
    """You are a helpful assistant.

Answer only from the provided context.

If the answer is not present in the context, say:
'I could not find that information in the retrieved documents.'

Context:
{context}

Question:
{input}"""
)

llm = ChatOpenRouter(
    model="gpt-oss-20b",
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

document_chain=create_stuff_documents_chain(
    llm,
    prompt
)
retriever_chain=create_retrieval_chain(
    retriever,
    document_chain

)

while True:
    user_input=input("enter: ")
    if user_input.lower()=="exit":
        break
    response=retriever_chain.invoke(
        {"input":user_input}
    )

    print("answer")
    print(response["answer"])

    print("-"*100)