import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llm.plugins import load_plugins, pm
from genai_processors.processor import ProcessorPart

from llm_deep_research import GenAIProcessorsAsyncResearch


def test_plugin_is_installed():
    load_plugins()

    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_deep_research" in names


@pytest.mark.asyncio
@patch("llm_deep_research.ResearchAgent")
async def test_GenAIProcessorsAsyncResearch_execute(ResearchAgent):
    researcher = ResearchAgent.return_value

    async def generator(self):
        yield ProcessorPart("doing...", substream_name="status")
        yield ProcessorPart("DONE!", substream_name="")

    researcher.side_effect = generator

    sut = GenAIProcessorsAsyncResearch()
    prompt = MagicMock()
    prompt.prompt = "Do some research"

    actual = []
    async for part in sut.execute(
        prompt,
        stream=False,
        response=MagicMock(),
        conversation=MagicMock(),
        key="test-key",
    ):
        actual.append(part)

    assert actual == [
        "--- \n *Status*: doing...",
        "# Final synthesized research\n\n  DONE!"
    ]
    ResearchAgent.assert_called_once_with(api_key="test-key")
