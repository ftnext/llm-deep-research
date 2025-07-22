import asyncio
import sys

import llm
from genai_processors import content_api, streams
from genai_processors.examples.research import ResearchAgent
from genai_processors.processor import ProcessorPart


def render_part(part: ProcessorPart) -> None:
    if part.substream_name == "status":
        print(f"--- \n *Status*: {part.text}")
        sys.stdout.flush()
    else:
        try:
            print(part.text)
        except Exception:
            print(f" {part.text} ")
        sys.stdout.flush()


class GenAIProcessorsResearch(llm.KeyModel):
    model_id = "genai-processors-research"
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"

    def execute(self, prompt, stream, response, conversation, key):
        asyncio.run(self._execute(prompt.prompt, key))
        return ""

    async def _execute(self, query, key):
        input_stream = streams.stream_content([ProcessorPart(query)])

        output_parts = content_api.ProcessorContent()
        async for content_part in ResearchAgent(api_key=key)(input_stream):
            if content_part.substream_name == "status":
                render_part(content_part)
            output_parts += content_part
        render_part(
            ProcessorPart(f"""# Final synthesized research

  {content_api.as_text(output_parts, substream_name="")}""")
        )


class GenAIProcessorsAsyncResearch(llm.AsyncKeyModel):
    model_id = "genai-processors-research"
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True

    async def execute(self, prompt, stream, response, conversation, key):
        async for part in self._execute(prompt.prompt, key):
            if part.substream_name == "status":
                yield f"--- \n *Status*: {part.text}"
            else:
                try:
                    yield part.text
                except Exception:
                    yield f" {part.text} "

    async def _execute(self, query, key):
        input_stream = streams.stream_content([ProcessorPart(query)])

        output_parts = content_api.ProcessorContent()
        async for content_part in ResearchAgent(api_key=key)(input_stream):
            if content_part.substream_name == "status":
                yield content_part
            output_parts += content_part
        yield ProcessorPart(f"""# Final synthesized research

  {content_api.as_text(output_parts, substream_name="")}""")


@llm.hookimpl
def register_models(register):
    register(GenAIProcessorsResearch(), GenAIProcessorsAsyncResearch())
