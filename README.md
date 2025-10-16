# Voice-Agent-Behavioral-Interview

An AI-powered voice assistant designed to conduct realistic behavioral interview practice sessions for engineering candidates. Built with OpenAI's Agent SDK and real-time voice capabilities.

## Overview

This application provides non-CS engineering students (mechanical, electrical, civil, etc.) with an interactive platform to practice behavioral interviews using natural voice interaction. The system employs a structured interview methodology, evaluating candidates against industry-standard competency frameworks while maintaining a conversational, supportive tone.

## Architecture

### Core Components

**1. Agent System**
- **Knowledge Agent (`Mo Ganda`)**: Primary behavioral interviewer utilizing RAG (Retrieval-Augmented Generation) with vector search capabilities
- **Search Agent**: Provides real-time web search for current interview trends and company-specific information
- **Triage Agent**: Routes user queries to appropriate specialized agents based on intent classification

**2. Vector Store Knowledge Base**
The system leverages OpenAI's vector store with semantic search across three key documents:
- `rubrics/competencies.txt`: 8 core behavioral competencies with STAR evaluation criteria and 1-5 scoring rubrics
- `roles/pm_intern/job_description.md`: Templated job description with competency mappings
- `company/values.md`: Company values framework with behavioral indicators

**3. Voice Pipeline**
- Real-time audio I/O using `sounddevice` library
- Configurable TTS (Text-to-Speech) model settings for voice affect, tone, and pacing
- Separate input/output sample rate handling (24kHz standard)
- Streaming audio response with chunked playback

### Technology Stack

```
├── OpenAI Agents SDK (v0.3.3+)
│   ├── Agent orchestration
│   ├── FileSearchTool (vector RAG)
│   ├── WebSearchTool
│   └── Voice pipeline
├── OpenAI API
│   ├── GPT-4 class models
│   ├── Vector store (embeddings)
│   └── TTS-1 model
├── Audio Processing
│   ├── sounddevice (PortAudio wrapper)
│   └── numpy (audio buffer manipulation)
└── Python 3.9+
```

## Key Features

### Structured Interview Flow

The Knowledge Agent follows a deterministic three-phase interview protocol:

**Phase 1: Initialization**
- Warm greeting and interviewer introduction (Mo Ganda)
- Readiness confirmation check
- Context setting

**Phase 2: Question Administration**
- Sequentially asks exactly 5 behavioral questions
- Covers distinct competency domains:
  - Problem Solving & Analytical Thinking
  - Teamwork & Collaboration
  - Communication Skills
  - Leadership & Initiative
  - Adaptability/Time Management/Resilience/Ethics (rotates)
- Provides brief encouraging feedback after each response
- Questions sourced dynamically from vector store competency rubrics

**Phase 3: Conclusion**
- Session summary and appreciation
- Optional feedback provision upon request
- Closing remarks

### STAR Method Evaluation

Internal evaluation engine assesses responses using the STAR framework:
- **S**ituation: Context clarity and relevance
- **T**ask: Objective definition and challenge articulation
- **A**ction: Specific steps taken and decision rationale
- **R**esult: Quantifiable outcomes and lessons learned

Scoring system: 1-5 scale per competency (evaluation kept internal unless explicitly requested)

### Voice Optimization

Custom TTS model configuration:
```python
TTSModelSettings(
    instructions=(
        "Voice Affect: Calm, composed, and reassuring; "
        "project quiet authority and confidence. "
        "Tone: Sincere, empathetic, and gently authoritative. "
        "Pacing: Steady and moderate; unhurried yet professional."
    )
)
```

Output characteristics:
- Short, segmented responses (1-2 sentences per thought)
- Plain language, minimal jargon
- Natural conversational flow
- Essential information only (cognitive load management)

## Installation

### Prerequisites

- Python 3.9+
- PortAudio (for `sounddevice`)
  ```bash
  # macOS
  brew install portaudio

  # Ubuntu/Debian
  sudo apt-get install portaudio19-dev
  ```
- OpenAI API key with GPT-4 access and vector store capabilities

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ranoobaba/Voice-Agent-Behavioral-Interview.git
   cd Voice-Agent-Behavioral-Interview
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirments.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=sk-your-key-here
   ```

## Usage

### Running the Voice Assistant

```bash
python Voice_agent.py
```

The script performs the following initialization:
1. Creates vector store: `Interview_KB_PM_Intern`
2. Uploads knowledge base documents
3. Initializes agent pipeline
4. Starts voice interface

### Voice Interaction Flow

```
1. Press Enter → Start recording
2. Speak your response
3. Press Enter → Stop recording & process
4. Listen to AI response
5. Repeat or type 'esc' to exit
```

### Testing (Text-based)

The application includes a `test_queries()` function for non-voice testing:

```python
# Uncomment in Voice_agent.py:
test_queries()
```

Test queries exercise different agent routing:
- Knowledge Agent: "I want to practice for a behavioral interview"
- Search Agent: "What are behavioral questions asked at Tesla?"

## Project Structure

```
Voice_agent_Vectorly/
├── Voice_agent.py              # Main application entry point
├── requirments.txt             # Python dependencies
├── .env                        # Environment configuration (not tracked)
├── .gitignore                  # Git ignore rules
├── rubrics/
│   └── competencies.txt        # Competency framework & scoring rubrics
├── roles/
│   └── pm_intern/
│       └── job_description.md  # Job description template
├── company/
│   └── values.md               # Company values & behavioral indicators
└── venv/                       # Virtual environment (not tracked)
```

## Configuration

### Agent Customization

**Knowledge Agent Instructions** (`Voice_agent.py:86-121`)
- Modify interview structure, question count, or competency focus
- Adjust voice output guidelines
- Configure FileSearchTool parameters (`max_num_results`, `vector_store_ids`)

**TTS Voice Settings** (`Voice_agent.py:148-154`)
- Customize voice affect, tone, and pacing
- Adjust for different use cases (formal interview vs. casual practice)

**Sample Rate Tuning** (`Voice_agent.py:157-158`)
- Input: Auto-detected from microphone
- Output: 24000 Hz (standard OpenAI TTS)
- Adjust output rate to slow down (lower) or speed up (higher) speech

### Knowledge Base Updates

To update competencies or add new interview frameworks:

1. Edit files in `rubrics/`, `roles/`, or `company/`
2. Run the upload script (embedded in `Voice_agent.py` startup)
3. Vector store automatically re-indexes on next run

Alternatively, manage vector stores manually:
```python
from Voice_agent import create_vector_store, upload_file

# Create new store
vs = create_vector_store("My_Custom_KB")

# Upload files
upload_file("path/to/file.md", vs["id"])
```

## Performance Considerations

### Latency Optimization

**Current bottlenecks:**
1. Vector search retrieval (2 documents @ ~200ms)
2. LLM inference (GPT-4 class @ ~1-3s)
3. TTS generation (streaming, ~500ms first chunk)

**Optimization strategies implemented:**
- Direct Knowledge Agent routing (bypass Triage Agent for voice interface)
- Reduced `max_num_results=2` for vector search
- Streaming audio playback (plays while generating)

### Cost Management

**Estimated costs per interview session:**
- Vector search: ~$0.001 per query
- GPT-4 tokens: ~$0.05-0.10 (5 Q&A rounds)
- TTS: ~$0.015 per 1000 characters

Total: ~$0.07-0.12 per 5-question interview

## Development

### Running in Development Mode

```bash
# Enable verbose logging
export OPENAI_LOG=debug

# Run with auto-reload (using tools like nodemon)
pip install nodemon
nodemon -e py --exec python Voice_agent.py
```

### Adding New Competencies

1. Edit `rubrics/competencies.txt`
2. Follow YAML structure:
   ```yaml
   - name: "New Competency"
     description: "..."
     key_behaviors: [...]
     star_evaluation_criteria: {...}
     scoring_guide: {...}
   ```
3. Update Knowledge Agent instructions to include new competency in rotation

### Extending Agent Capabilities

Example: Add a feedback agent
```python
feedback_agent = Agent(
    name="FeedbackAgent",
    instructions="Provide detailed STAR-based feedback...",
    tools=[FileSearchTool(...)]
)

# Add to triage handoffs
triage_agent = Agent(..., handoffs=[knowledge_agent, search_agent, feedback_agent])
```

## Troubleshooting

### Common Issues

**1. Audio not recording**
```bash
# Test microphone
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check permissions (macOS)
# System Preferences → Security & Privacy → Microphone
```

**2. Vector store upload fails**
- Ensure `.yaml` files are saved as `.txt` (OpenAI limitation)
- Check file paths are relative to script location
- Verify API key has vector store permissions

**3. TTS voice sounds robotic**
- Increase `output_samplerate` (try 32000 or 44100)
- Adjust TTS instructions for more natural pacing
- Check audio driver compatibility

**4. Slow response times**
- Reduce `max_num_results` to 1
- Use GPT-3.5 for testing (faster, cheaper)
- Check network latency to OpenAI API

## Roadmap

- [ ] Multi-turn conversation memory (maintain context across sessions)
- [ ] Interview session recording and playback
- [ ] Automated feedback generation (STAR scoring breakdown)
- [ ] Multi-language support
- [ ] Web-based UI (remove terminal dependency)
- [ ] Integration with calendar APIs for scheduled practice
- [ ] Company-specific interview prep (Google, Amazon, Tesla, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built with [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- Voice processing powered by [PortAudio](http://www.portaudio.com/)
- Inspired by behavioral interview frameworks from top tech companies

## Contact

Project Link: [https://github.com/Ranoobaba/Voice-Agent-Behavioral-Interview](https://github.com/Ranoobaba/Voice-Agent-Behavioral-Interview)

---

**Note**: This is an educational tool for interview practice. Actual interview performance depends on genuine experience, preparation, and self-reflection. Use this tool to build confidence and communication skills, not to memorize scripted answers.
