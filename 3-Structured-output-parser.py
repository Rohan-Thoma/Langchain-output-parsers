from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

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

#here suppose we want 3 facts about any topic so we need to design the schema in such a manner
#We need to pass list of schema objects to get our desired result
schema =[
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

#now we design the parser
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'black_hole'})

result = model.invoke(prompt)

final_result =  parser.parse(result.content)

print(final_result)
