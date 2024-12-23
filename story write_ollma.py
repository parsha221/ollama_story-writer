import warnings
warnings.filterwarnings('ignore')

import numpy as np
from crewai import Agent, Task, Crew
from crewai import LLM

llm=LLM(
 model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
)

# Define the writer agent
writer = Agent(
role="Story Writer",
goal="To write a small story",
    backstory=(
 "You are a story writer who writes small stories for children's books. "
 "Your story takes shape with words given by the words writer. "
 "The main objective is to follow the words given by the words writer "
    "and make a meaningful, small, funny story from it."
),
allow_delegation=False,
    verbose=True,
    llm = llm
)
# Define the task
Story_writer = Task(
description=(
"Write a meaningful story for children about a Panda. "
"Include simple words in it, things that children will love, "
"and give a moral message in it."
),
expected_output="A small story for children with a meaningful message",
agent=writer
)
print("task_complete") 
# Create the crew
crew = Crew(
agents=[writer],
tasks=[Story_writer],
verbose=0
)
print("crew created")
 
# Run the crew
result = crew.kickoff()
print(result)
print('finish')
