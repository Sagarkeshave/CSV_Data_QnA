from langchain.llms import GooglePalm
import os

google_api_key = os.environ.get("google_api_key")

llm = GooglePalm(google_api_key=google_api_key, temperature= 0.9)

poem = llm("Write a 4 line poem of my love for samosa")
print(poem)
