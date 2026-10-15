from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# Step 1: Preprocess
preprocess = RunnableLambda(
    lambda log: {"log": log.strip()}
)

# Step 2: Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cybersecurity log analyst."),
    ("human", "Analyze this HTTP log:\n{log}")
])

# Step 3: Model
model = ChatOpenAI(model="gpt-4")

# Step 4: Parser
parser = StrOutputParser()

# Chain 
#chain = preprocess | prompt | model | parser

chain = RunnableSequence(
    first = preprocess,
    middle = [prompt, model],
    last = parser)


# Execution
result = chain.invoke(
    "  GET /login?user=admin HTTP/1.1 200  "
)

print(result)