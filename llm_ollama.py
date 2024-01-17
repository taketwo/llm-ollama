import llm


@llm.hookimpl
def register_models(register):
    register(Ollama())


class Ollama(llm.Model):
    model_id = "ollama"
    can_stream: bool = False  # TODO: Implement streaming support

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]
