from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


api_key = os.getenv("OPENAI_API_KEY")
set_default_openai_key(api_key)

# --- Agent: Search Agent ---
search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def upload_file(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)   # just the name, like "notes.pdf"
    try:
        # 1) Safely open the file
        with open(file_path, "rb") as f:
            # 2) Send the file to OpenAI's server
            file_response = client.files.create(file=f, purpose="assistants")
            # file_response has a unique ID, e.g. "file_abc123"

        # 3) Attach the uploaded file to your vector store (your searchable “library”)
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id,         # use the ID we got back
        )

        # 4) Return useful info to the caller (so other code can reuse it)
        return {"file": file_name, "status": "success", "file_id": file_response.id}

    except Exception as e:
        # If anything fails (file not found, network hiccup, etc.), say so
        return {"file": file_name, "status": "failed", "error": str(e)}


def create_vector_store(store_name: str) -> dict:
    try:
        vs = client.vector_stores.create(name=store_name)
        return {
            "id": vs.id,
            "name": vs.name,
            "created_at": vs.created_at,
            "file_count": getattr(vs.file_counts, "completed", 0),
        }
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}
    
vector_store = create_vector_store("Interview_KB_PM_Intern")
assert vector_store and "id" in vector_store, "Vector store creation failed"
vector_store_id = vector_store["id"]

sources = [
    "rubrics/competencies.txt",
    "roles/pm_intern/job_description.md",
    "company/values.md",
]
results = [upload_file(p, vector_store_id) for p in sources]
print(results)

# --- Agent: Knowledge Agent ---
knowledge_agent = Agent(
    name="KnowledgeAgent",
    instructions=(
        "You are an AI behavioral interviewer for engineering candidates. "
        "Use the FileSearchTool to access competency rubrics, job descriptions, and company values. "
        "Ask STAR-method questions (Situation, Task, Action, Result) to evaluate behavioral competencies. "
        "Evaluate responses using the 1-5 scoring rubric provided in the competencies file. "
        "Focus on: problem-solving, teamwork, communication, leadership, adaptability, "
        "time management, resilience, and ethics. Be conversational and encouraging."
    ),
    tools=[FileSearchTool(
            max_num_results=3,
            vector_store_ids=[vector_store_id],
        ),],
)
