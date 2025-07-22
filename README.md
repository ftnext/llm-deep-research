# llm-deep-research

[![PyPI](https://img.shields.io/pypi/v/llm-deep-research.svg)](https://pypi.org/project/llm-deep-research/)
[![Changelog](https://img.shields.io/github/v/release/ftnext/llm-deep-research?include_prereleases&label=changelog)](https://github.com/ftnext/llm-deep-research/releases)
[![Tests](https://github.com/ftnext/llm-deep-research/actions/workflows/test.yml/badge.svg)](https://github.com/ftnext/llm-deep-research/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ftnext/llm-deep-research/blob/main/LICENSE)



## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-deep-research
```
## Usage

### GenAI Processors' research

https://github.com/google-gemini/genai-processors/tree/main/examples/research

```
export LLM_GEMINI_KEY=your_api_key_here

llm -m genai-processors-research 'Research the best things about owning dalmatians!'
```

Also supports `--async`.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-deep-research
```
Now create a new virtual environment and install the dependencies and test dependencies:
```bash
uv sync --extra test
```
To run the tests:
```bash
uv run pytest
```
