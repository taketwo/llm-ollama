import llm


@llm.hookimpl
def register_models(register):
    register(Ollama())


class Ollama(llm.Model):
    model_id = "ollama"
    can_stream: bool = False  # TODO: Implement streaming support

    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.model_id = model_id

    def __str__(self) -> str:
        return f"Ollama: {self.model_id}"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]
