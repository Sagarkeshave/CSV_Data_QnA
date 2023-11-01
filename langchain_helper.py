from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os

load_dotenv()

google_api_key = os.environ["GOOGLE_API_KEY"]

print(os.environ["GOOGLE_API_KEY"])

# create model instance
llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

# create embedding
instructor_embeddings = HuggingFaceInstructEmbeddings()  # model is default

# we will create vector database and store it locally and use again (to avoid databasing multiple times)
vectordb_file_path = "faiss_index_main"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='data.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():

    # laod the vector database from local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # prompt template to prevent model hallucination
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))