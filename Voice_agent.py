from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


api_key = os.getenv("OPENAI_API_KEY")
set_default_openai_key(api_key)

# Common system prompt for voice output best practices:
voice_system_prompt = """
[Output Structure]
Your output will be delivered in an audio voice response, please ensure that every response meets these guidelines:
1. Use a friendly, human tone that will sound natural when spoken aloud.
2. Keep responses short and segmented—ideally one to two concise sentences per step.
3. Avoid technical jargon; use plain language so that instructions are easy to understand.
4. Provide only essential details so as not to overwhelm the listener.
"""

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
    instructions=voice_system_prompt + (
        "You are an AI behavioral interviewer for engineering candidates. "
        "Use the FileSearchTool to access competency rubrics, job descriptions, and company values. "
        "Ask behavioral questions naturally and conversationally. Focus on: problem-solving, teamwork, "
        "communication, leadership, adaptability, time management, resilience, and ethics. "
        "When evaluating candidate responses, internally assess them using the STAR method "
        "(Situation, Task, Action, Result) and the 1-5 scoring rubric from the competencies file. "
        "Do not mention STAR or scoring to the candidate. Be encouraging and supportive."
    ),
    tools=[FileSearchTool(
            max_num_results=2,
            vector_store_ids=[vector_store_id],
        ),],
)
# --- Agent: Triage Agent ---
triage_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Vectorly's behavioral interview practice platform.
Welcome engineering candidates and help them practice for behavioral interviews.

Based on the user's intent, route to:
- KnowledgeAgent to conduct behavioral interview practice using STAR method
- SearchAgent for current information about interview best practices or companies
"""),
    handoffs=[knowledge_agent, search_agent],
)

from agents import Runner, trace

def test_queries():
    import asyncio

    async def run_tests():
        examples = [
            "Hi, I want to practice for a behavioral interview for a mechanical engineering internship", # Knowledge Agent test
            "How do I answer a Tell me about yourself question?", # Knowledge Agent test
            "What are the behavioural questions that are asked in the tesla interview", # Search Agent test
        ]
        with trace("Vectorly Interview Assistant"):
            for query in examples:
                result = await Runner.run(triage_agent, query)
                print(f"User: {query}")
                print(result.final_output)
                print("---")

    asyncio.run(run_tests())

# Run the tests
test_queries()

# %%
import numpy as np
import sounddevice as sd
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline, TTSModelSettings, VoicePipelineConfig

def voice_assistant():
    import asyncio

    # Voice settings for behavioral interviewer
    interviewer_voice_settings = TTSModelSettings(
        instructions=(
            "Voice Affect: Calm, composed, and reassuring; project quiet authority and confidence. "
            "Tone: Sincere, empathetic, and gently authoritative—express genuine apology while conveying competence. "
            "Pacing: Steady and moderate; unhurried enough to communicate care, yet efficient enough to demonstrate professionalism."
        )
    )

    async def run_voice():
        input_samplerate = sd.query_devices(kind='input')['default_samplerate']
        output_samplerate = 24000  # Standard OpenAI TTS sample rate
        voice_config = VoicePipelineConfig(tts_settings=interviewer_voice_settings)

        while True:
            pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(knowledge_agent), config=voice_config)

            # Check for input to either provide voice or exit
            cmd = input("Press Enter to speak your query (or type 'esc' to exit): ")
            if cmd.lower() == "esc":
                print("Exiting...")
                break
            print("Listening...")
            recorded_chunks = []

             # Start streaming from microphone until Enter is pressed
            with sd.InputStream(samplerate=input_samplerate, channels=1, dtype='int16', callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())):
                input()

            # Concatenate chunks into single buffer
            recording = np.concatenate(recorded_chunks, axis=0)

            # Input the buffer and await the result
            audio_input = AudioInput(buffer=recording)

            with trace("Vectorly Interview Voice Assistant"):
                result = await pipeline.run(audio_input)

             # Transfer the streamed result into chunks of audio
            response_chunks = []
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    response_chunks.append(event.data)

            response_audio = np.concatenate(response_chunks, axis=0)

            # Play response
            print("Assistant is responding...")
            sd.play(response_audio, samplerate=output_samplerate)
            sd.wait()
            print("---")

    asyncio.run(run_voice())

# Run the voice assistant
voice_assistant()

