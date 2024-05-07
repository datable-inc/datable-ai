import os

import pytest
from langchain_core.prompts import ChatPromptTemplate

from datable_ai.core.llm import LLM_TYPE
from datable_ai.output import Output


@pytest.fixture
def output():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_API_MODEL"] = "gpt-4"
    return Output(LLM_TYPE.OPENAI, "テストプロンプトテンプレート")


def test_output_initialization(output):
    assert output.llm_type == LLM_TYPE.OPENAI
    assert output.prompt_template == "テストプロンプトテンプレート"
    assert isinstance(output.prompt, ChatPromptTemplate)
    assert output.llm is not None


def test_token_count_from_string(output):
    text = "これはテストテキストです。"
    token_count = output._num_tokens_from_string(text)
    assert isinstance(token_count, int)
    assert token_count > 0
