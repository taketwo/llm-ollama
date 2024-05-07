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

The plugin automatically creates a short alias for models that have `:latest` in the name, so the previous command is equivalent to running:

```bash
llm -m llama2 'How much is 2+2?'
```

To start an interactive chat session:

```bash
llm chat -m llama2
```
```
Chatting with llama2:latest
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
>
```

## Model aliases

The same Ollama model may be referred by several names with different tags. For example, in the following list, there is a single unique model with three different names:

```bash
ollama list
NAME                    ID              SIZE    MODIFIED
stable-code:3b          aa5ab8afb862    1.6 GB  9 hours ago
stable-code:code        aa5ab8afb862    1.6 GB  9 seconds ago
stable-code:latest      aa5ab8afb862    1.6 GB  14 seconds ago
```

In such cases, the plugin will register a single model and create additional aliases. Continuing the previous example, this is what LLM will have:

```bash
llm models
...

Ollama: stable-code:3b (aliases: stable-code:code, stable-code:latest, stable-code)
```

## Model options

All models accept [Ollama modelfile parameters](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter) as options. Use the `-o name value` syntax to specify them, for example:

- `-o temperature 0.8`: set the temperature of the model
- `-o num_ctx 256000`: set the size of the context window used to generate the next token

See the referenced page for the complete list with descriptions and default values.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-ollama
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

To run the tests:
```bash
pytest
```
