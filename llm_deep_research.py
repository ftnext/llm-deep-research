import asyncio
import sys

import llm
from genai_processors import content_api, streams
from genai_processors.examples.research import ResearchAgent
from genai_processors.processor import ProcessorPart
from opentelemetry._events import set_event_logger_provider
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

set_tracer_provider(TracerProvider())
get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

set_logger_provider(LoggerProvider())
get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(ConsoleLogExporter())
)
set_event_logger_provider(EventLoggerProvider())

GoogleGenAiSdkInstrumentor().instrument()


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

        researcher = ResearchAgent(api_key=key)
        output_parts = content_api.ProcessorContent()
        async for content_part in researcher(input_stream):
            if content_part.substream_name == "status":
                yield content_part
            output_parts += content_part
        yield ProcessorPart(f"""# Final synthesized research

  {content_api.as_text(output_parts, substream_name="")}""")


@llm.hookimpl
def register_models(register):
    register(GenAIProcessorsResearch(), GenAIProcessorsAsyncResearch())
