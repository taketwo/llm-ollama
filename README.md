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

## Quickstart

By default, the plugin connects to a local Ollama server. Ensure the server is running and has some models available. Alternatively, the plugin can connect to a remote or managed Ollama cloud server; see [Connecting to Ollama server](#connecting-to-ollama-server) for configuration instructions.

The plugin automatically discovers all models available on the server and registers them with LLM. To see the list of models and their capabilities, run:

```bash
llm ollama models
```
```bash
model                             digest          capabilities
gemma3:latest                     c0494fe00251    completion, vision
gpt-oss:120b-cloud                569662207105    completion, tools, thinking
mxbai-embed-large:latest          468836162de7    embedding
qwen3:4b                          2bfd38a7daaf    completion, tools, thinking
```

Once registered, models are available for prompting, chatting, and embedding. Assuming you have `gemma3:latest` available, you can run a prompt using:

```bash
llm -m gemma3:latest 'How much is 2+2?'
```

The plugin automatically creates shorter aliases for models that have `:latest` in the name, so the previous command is equivalent to running:

```bash
llm -m gemma3 'How much is 2+2?'
```

To start an interactive chat session instead of a one-shot prompt, run:

```bash
llm chat -m gemma3
```
```bash
Chatting with gemma3:latest
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
Type '!edit' to open your default editor and modify the prompt
Type '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments
>
```

## Features

### Image attachments

Multi-modal Ollama models can accept image attachments using the [LLM attachments](https://llm.datasette.io/en/stable/usage.html#attachments) option:

```bash
llm -m llava "Describe this image" -a https://static.simonwillison.net/static/2024/pelicans.jpg
```

### Tools

Ollama models with [tools support](https://ollama.com/search?c=tools) can make use of [LLM tools](https://llm.datasette.io/en/stable/tools.html) passed to them:

```bash
llm -m llama3.2 -T llm_time 'What is the time?' --td
```

The plugin also registers `ollama_web_search` and `ollama_web_fetch` tools that wrap the [web search API](https://docs.ollama.com/web-search) provided by `ollama.com`. These tools augment models with the latest information to reduce hallucinations and improve accuracy. To use these tools, an API key must be configured (see [Ollama cloud](#ollama-cloud)).

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

The same Ollama model may be referenced by several names with different tags. For example, in the following list, there is a single unique model with three different names:

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

All models accept [Ollama modelfile parameters](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx#parameter) as options. Use the `-o name value` syntax to specify them, for example:

- `-o temperature 0.8`: set the temperature of the model
- `-o num_ctx 256000`: set the size of the context window used to generate the next token

See the referenced page for the complete list with descriptions and default values.

Additionally, the -o flag supports plugin-specific options:

- `-o json_object 1` forces the model to reply with a valid JSON object. Note that your prompt must mention JSON for this to work;
- `-o think false` disables the intermediate reasoning step for thinking-capable models.

## Connecting to Ollama server

The plugin connects to an Ollama server to list and run models. Three deployment scenarios are supported: a local server, a self-hosted remote server, and Ollama's hosted cloud service.

### Local server

By default, the plugin connects to a local Ollama server at `localhost:11434`. If your local server runs on a non-default port, set `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://localhost:8080
```

### Remote server

To connect to a self-hosted Ollama server on your network or in the cloud, set `OLLAMA_HOST` to its address:

```bash
export OLLAMA_HOST=https://192.168.1.13:11434
```

#### Authentication

If the server is protected with Basic Authentication, include the credentials in the URL:

```bash
export OLLAMA_HOST=https://username:password@192.168.1.13:11434
```

Special characters in usernames or passwords must be URL-encoded:

```bash
# For username "user@domain" and password "p@ssw0rd"
export OLLAMA_HOST=https://user%40domain:p%40ssw0rd@192.168.1.13:11434
```

If the server is behind a reverse proxy that requires custom headers, use the `OLLAMA_HEADERS` environment variable with a comma-separated list of `key=value` pairs:

```bash
# JWT token auth (e.g. Open-WebUI's Ollama endpoint)
export OLLAMA_HEADERS='Authorization=Bearer mytoken,User-Agent=custom-client'

# Cloudflare Tunnel with a Service Token
export OLLAMA_HEADERS='CF-Access-Client-Id=abcdef.access,CF-Access-Client-Secret=123456789'
```

### Ollama cloud

[Ollama cloud](https://ollama.com) is a hosted service that lets you run models without installing or operating a local server. It offers a range of open models and does not require any local setup beyond configuring the plugin.

To use it, point the plugin at the cloud endpoint and provide your API key:

```bash
export OLLAMA_HOST=https://ollama.com
llm keys set ollama # paste your API key when prompted
```

The API key is stored securely by `llm` and used automatically. Alternatively, you can set it as an environment variable:

```bash
export OLLAMA_API_KEY=your-api-key
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

The environment includes `llm`; it will pick up the local version of the plugin, which is useful for manual testing.

To run automated unit and integration tests:

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
python -m ruff format .
```
