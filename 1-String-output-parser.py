from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

#We will use Prompt Template to make our prompts dynamic
#1st prompt -> detailed report
template1= PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
    )
#2nd prompt -> summary
template2 = PromptTemplate(
    template=" write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

parser = StrOutputParser()  

#Now we will implement out complex flow with the help of chain
chain = template1 | model | parser | template2 | model | parser

#Lets execute the chain
result = chain.invoke({'topic': 'black_hole'})

print(result)


