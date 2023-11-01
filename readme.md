
# CSV Q&A: Question and Answer System Based on Google Palm LLM and Langchain for E-learning company.

This is an end to end LLM project based on Google Palm and Langchain. Building a Q&A system for an e-learning company.


<br>

## Demo
![Demo](https://github.com/Sagarkeshave/CSV_Data_QnA/blob/master/demo/ezgif.com-video-to-gif%20(4).gif)

<br>

## Project Highlights

- Use a  CSV file of FAQs aout data in that company.
- We will build an LLM based question and answer system.
- User should be able to use this system to ask questions directly and get answers within seconds.

## Techs used
  - Langchain + Google Palm: LLM based Q&A
  - Streamlit: UI
  - Huggingface instructor embeddings: Text embeddings
  - FAISS: Vector databse

## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/Sagarkeshave/CSV_Data_QnA.git
```
2.Navigate to the project directory:

```bash
  cd CSV_Data_QnA
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```

## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter

## Sample Questions
  - Do you guys provide internship and also do you offer EMI payments?
  - Do you have javascript course?
  - Should I learn power bi or tableau?
  - I've a MAC computer. Can I use powerbi on it?
  - I don't see power pivot. how can I enable it?

## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your Google API key.
