from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

import os
api = os.getenv('HF_TOKEN')

llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=api,
    repo_id= "google/gemma-4-31B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.format()

print("\n========================\n")
print(prompt)
print("\n========================\n")

result = model.invoke(prompt)

print(result)

print("\n========================\n")

parser= JsonOutputParser()

final_result = parser.parse(result.content)

print(final_result)


#now  we can skip this passing inputs manually  and create chains 
chain = template | model | parser

#Here even though we dont have any input variables, we need pass an empty dictionary otherwise we will get an error.
result = chain.invoke({})

print(result) 