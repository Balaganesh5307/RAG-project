from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are DocLens AI.

    Explain the following question clearly and simply.

    Question:
    {question}
    """
)

chain = prompt | model

question = "What is RAG?"

response = chain.invoke({
    "question": question
})

print(response.content)