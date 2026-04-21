from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

import os
api = os.getenv('HF_TOKEN')

llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=api,   
    repo_id= "google/gemma-4-26B-A4B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name : str = Field(description="name of the person")
    age : int = Field(gt=18, description="age of the person")
    city : str = Field(description="name of the city that the person belongs to ")

 
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'place': 'indian'})

#lets print the prompt and see how it actually looks like
print("\n=======================\n")
print(prompt)
print("\n=======================\n")

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
print("\n=======================\n")