# llm-ollama

[![PyPI](https://img.shields.io/pypi/v/llm-ollama.svg)](https://pypi.org/project/llm-ollama/)
[![Changelog](https://img.shields.io/github/v/release/taketwo/llm-ollama?include_prereleases&label=changelog)](https://github.com/taketwo/llm-ollama/releases)
[![Tests](https://github.com/taketwo/llm-ollama/actions/workflows/test.yml/badge.svg)](https://github.com/taketwo/llm-ollama/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/taketwo/llm-ollama/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin providing access to models running on local [Ollama](https://ollama.ai) server.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-ollama
```

## Usage

First, ensure that your Ollama server is running and that you have pulled some models. You can use `ollama list` to check what is locally available.

The plugin will query the Ollama server for the list of models. You can use `llm ollama list-models` to see the list; it should be the same as output by `ollama list`. All these models will be automatically registered with LLM and made available for prompting and chatting.

Assuming you have `llama2:latest` available, you can run a prompt using:

```bash
llm -m llama2:latest 'How much is 2+2?'
```

To start an interactive chat session:

```bash
llm chat -m llama2:latest
```
```
Chatting with llama2:latest
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
>
```

## Model options

All models accept the following options, using `-o name value` syntax:

- `-o temperature 0.8`: The temperature of the model. Increasing the temperature will make the model answer more creatively.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-ollama
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:
```bash
pytest
```
