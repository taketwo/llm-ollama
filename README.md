# llm-ollama

[![PyPI](https://img.shields.io/pypi/v/llm-ollama.svg)](https://pypi.org/project/llm-ollama/)
[![Changelog](https://img.shields.io/github/v/release/taketwo/llm-ollama?include_prereleases&label=changelog)](https://github.com/taketwo/llm-ollama/releases)
[![Tests](https://github.com/taketwo/llm-ollama/actions/workflows/test.yml/badge.svg)](https://github.com/taketwo/llm-ollama/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/taketwo/llm-ollama/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin providing access to models running on an [Ollama](https://ollama.ai) server.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-ollama
```

## Usage

First, ensure that the Ollama server is running and that you have pulled some models. You can use `ollama list` to check what is locally available.

The plugin will query the Ollama server for the list of models. You can use `llm ollama list-models` to see the list; it should be the same as output by `ollama list`. All these models will be automatically registered with LLM and made available for prompting, chatting, and embedding.

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

### Image attachments

Multi-modal Ollama models can accept image attachments using the [LLM attachments](https://llm.datasette.io/en/stable/usage.html#attachments) option:

```bash
llm -m llava "Describe this image" -a https://static.simonwillison.net/static/2024/pelicans.jpg
```

### Tools

Some models such as [Qwen 3](https://ollama.com/library/qwen3) support [tools](https://llm.datasette.io/en/stable/tools.html). Ollama have a [list of tool supporting models](https://ollama.com/search?c=tools).
```bash
ollama pull qwen3:4b

llm -m qwen3:4b -T llm_time 'What is the time?' --td
```

### Embeddings

The plugin supports [LLM embeddings](https://llm.datasette.io/en/stable/embeddings/cli.html). Both regular and specialized embedding models (such as `mxbai-embed-large`) can be used:

```bash
llm embed -m mxbai-embed-large -i README.md
```

By default, the input will be truncated from the end to fit within the context length. This behavior can be changed by setting `OLLAMA_EMBED_TRUNCATE=no` environment variable. In such cases, embedding operation will fail if the context length is exceeded.

### JSON schemas

Ollama's built-in support for [structured outputs](https://ollama.com/blog/structured-outputs) can be accessed through [LLM schemas](https://llm.datasette.io/en/stable/schemas.html), for example:

```bash
llm -m llama3.2 --schema "name, age int, one_sentence_bio" "invent a cool dog"
```

### Async models

The plugin registers [async LLM models](https://llm.datasette.io/en/stable/python-api.html#async-models) suitable for use with Python [asyncio](https://docs.python.org/3/library/asyncio.html).

To utilize an async model, retrieve it using `llm.get_async_model()` function instead of `llm.get_model()` and then await the response:

```python
import asyncio, llm

async def run():
    model = llm.get_async_model("llama3.2:latest")
    response = model.prompt("A short poem about tea")
    print(await response.text())

asyncio.run(run())
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

Additionally, the `-o json_object 1` option can be used to force the model to reply with a valid JSON object. Note that your prompt must mention JSON for this to work.

## Ollama server address

`llm-ollama` will try to connect to a server at the default `localhost:11434` address. If your Ollama server is remote or runs on a non-default port, you can use `OLLAMA_HOST` environment variable to point the plugin to it, e.g.:

```bash
export OLLAMA_HOST=https://192.168.1.13:11434
```

### Authentication

If your Ollama server is protected with Basic Authentication, you can include the credentials directly in the `OLLAMA_HOST` environment variable:

```bash
export OLLAMA_HOST=https://username:password@192.168.1.13:11434
```

The plugin will parse the credentials and use them for authentication. Special characters in usernames or passwords should be URL-encoded:

```bash
# For username "user@domain" and password "p@ssw0rd"
export OLLAMA_HOST=https://user%40domain:p%40ssw0rd@192.168.1.13:11434
```

## Development

### Setup

To set up this plugin locally, first checkout the code. Then create a new virtual environment and install the dependencies. If you are using `uv`:

```bash
cd llm-ollama
uv venv
uv pip install -e '.[test,lint]'
```

Otherwise, if you prefer using standard tools:

```bash
cd llm-ollama
python3 -m venv .venv
pip install -e '.[test,lint]'
```

### Testing and linting

To test or lint the code, first activate the environment:

```bash
source .venv/bin/activate
```

To run unit and integration tests:

```bash
python -m pytest
```

Integration tests require a running Ollama server and will be:
- Enabled automatically if an Ollama server is available;
- Skipped if Ollama server is unavailable;
- Force-enabled with `--integration` (but fail if Ollama server is unavailable);
- Force-disabled with `--no-integration`.

To format the code:

```bash
python -m black .
```
