from unittest.mock import MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from datable_ai.core.llm import LLM_TYPE, create_llm


# Mock environment variables if needed
@patch.dict(
    "os.environ",
    {
        "OPENAI_API_MODEL": "test-model-openai",
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.azure.com/",
        "AZURE_OPENAI_API_VERSION": "test-version",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "ANTHROPIC_API_MODEL": "test-anthropic-model",
    },
)
class TestCreateLLM:
    @patch("datable_ai.core.llm._create_chat_openai")
    def test_create_llm_openai(self, mock_create):
        mock_create.return_value = MagicMock(spec=ChatOpenAI)
        llm = create_llm(LLM_TYPE.OPENAI)
        assert isinstance(llm, BaseChatModel)
        mock_create.assert_called_once()

    @patch("datable_ai.core.llm._create_azure_chat_openai")
    def test_create_llm_azure_openai(self, mock_create):
        mock_create.return_value = MagicMock(spec=AzureChatOpenAI)
        llm = create_llm(LLM_TYPE.AZURE_OPENAI)
        assert isinstance(llm, BaseChatModel)
        mock_create.assert_called_once()

    @patch("datable_ai.core.llm._create_chat_anthropic")
    def test_create_llm_anthropic(self, mock_create):
        mock_create.return_value = MagicMock(spec=ChatAnthropic)
        llm = create_llm(LLM_TYPE.ANTHROPIC)
        assert isinstance(llm, BaseChatModel)
        mock_create.assert_called_once()

    def test_create_llm_invalid(self):
        with pytest.raises(ValueError):
            create_llm("INVALID_TYPE")
