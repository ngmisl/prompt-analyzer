# Prompt Analyzer

A tool for analyzing and refining prompts using various LLM models through OpenRouter API.

## Features

- Analyze prompt structure and effectiveness
- Rewrite prompts for clarity
- Generate potential improvements
- Refine prompts based on suggestions
- Create optimized final versions
- Customizable model pipeline
- Support for multiple LLM models (Llama 3, Nova Lite, LearnLM Pro)

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenRouter API key

## Installation

1. Install uv if you haven't already:

    ```bash
    pip install uv
    ```

2. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/prompt-analyzer.git
    cd prompt-analyzer
    ```

3. Create and activate virtual environment:

    ```bash
    uv venv
    source .venv/bin/activate  # On Unix/macOS
    # or
    .venv\Scripts\activate  # On Windows
    ```

4. Install the package with development dependencies:

    ```bash
    uv pip install -e ".[dev]"
    ```

## Development

The project uses:

- Black for code formatting (line length: 100)
- Ruff for linting with rules E, F, B (line length: 100)

To format and lint code:

```bash
black .
ruff check .
```

## Running the Code

1. Make sure your virtual environment is activated:

    ```bash
    source .venv/bin/activate  # On Unix/macOS
    # or
    .venv\Scripts\activate  # On Windows
    ```

2. Run the analyzer:

    ```bash
    uv run src/prompt_analyzer/main.py
    ```

3. On first run:
   - Enter your OpenRouter API key when prompted
   - Choose whether to customize the model pipeline
   - Enter your prompt to analyze

4. The tool will display:
   - Initial analysis of your prompt
   - Rewritten version for clarity
   - List of potential improvements
   - Refined version
   - Final optimized prompt
   - Models used in each stage

Example usage:

```bash
$ python main.py

Prompt Analysis and Refinement Tool
===================================

Available Models:
- Llama 3 (llama): High performance instruct model
- Nova Lite (nova): Efficient general purpose model
- LearnLM Pro (learnlm): Specialized learning model

Customize model pipeline? (y/n): n

Enter your prompt to analyze: Write a blog post about AI safety

Analyzing prompt...

=== Results ===
...
```

## Project Structure

``` bash
prompt-analyzer/
├── .env               # Created automatically with your API key
├── .venv/             # Virtual environment directory
├── pyproject.toml     # Project configuration and dependencies
├── main.py           # Main application code
└── tests/            # Test directory (when adding tests)
```

## Dependencies

Core:

- langchain >= 0.1.0
- langchain-openai >= 0.0.5
- python-dotenv >= 1.0.0
- pydantic >= 2.0.0
- typing-extensions >= 4.0.0

Development:

- pytest >= 8.0.0
- black >= 24.0.0
- ruff >= 0.1.0

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (stored in .env file)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

Make sure to:

- Format code with Black
- Run Ruff linter
- Add tests for new features
- Update documentation as needed

## License

MIT License

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenRouter](https://openrouter.ai/)
- [uv](https://github.com/astral-sh/uv)
