import pytest
from langchain_core.prompts import ChatPromptTemplate

from datable_ai.core.llm import LLM_TYPE
from datable_ai.output import Output


@pytest.fixture
def output():
    return Output(LLM_TYPE.OPENAI, "テストプロンプトテンプレート")


def test_init(output):
    assert output.llm_type == LLM_TYPE.OPENAI
    assert output.prompt_template == "テストプロンプトテンプレート"
    assert isinstance(output.prompt, ChatPromptTemplate)
    assert output.llm is not None


def test_num_tokens_from_string(output):
    text = "これはテストテキストです。"
    num_tokens = output._num_tokens_from_string(text)
    assert isinstance(num_tokens, int)
    assert num_tokens > 0
