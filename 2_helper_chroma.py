from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
import os
from langchain.llms import GooglePalm

load_dotenv()

google_api_key = os.environ["GOOGLE_API_KEY"]

print(os.environ["GOOGLE_API_KEY"])

# create model instance
llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

# create embedding
google_palm_embeddings = GooglePalmEmbeddings(google_api_key="api_key")

# we will create vector database and store it locally and use again (to avoid databasing multiple times)
vector_db_filepath = "faiss_index"


def create_vector():
    loader = CSVLoader(file_path="data.csv", source_column="prompt")
    data = loader.load()


    # create FAISS instance for data embedding
    # vectordb = FAISS.from_documents(documents=data,
    #                                 embedding=google_palm_embeddings)


    # Will try chroma
    vectordb = Chroma.from_documents(data, google_palm_embeddings, persist_directory='./chromadb')

    # save vector database locally
    vectordb.persist()


# Create RetrievalQA chain along with prompt template

def get_qa_chain(query):

    # laod the vector database from local folder
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=google_palm_embeddings)

    # print(vectordb)

    # create retriever for performing the query
    retriever = vectordb.similarity_search(query)

    # print(retriever)
    # prompt template to prevent model hallucination
    # prompt_template = """ Given the following context and a question, generate an answer based on this
    #     context only. In the answer try to provide as much text as possible from "response" section in the source
    #     document context without making much changes. If the answer is not found in the context, kindly state "I dont know".
    #     Dont try to make up an answer.
    #
    #     CONTEXT : {context}
    #     QUESTION : {question} """
    #
    # PROMPT = PromptTemplate(
    #     template=prompt_template,
    #     input_variables=["context", "question"]
    # )

    # chain = RetrievalQA(llm=llm,
    #                     chain_type="stuff",
    #                     retriever=retriever,
    #                     chain_type_kwargs={"prompt": PROMPT})

    return retriever



if __name__ == "__main__":
    # create_vector()
    query = "power bi"
    chain = get_qa_chain(query)
    print(chain)



