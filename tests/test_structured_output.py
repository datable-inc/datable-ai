import os
from typing import Dict

import pytest
from langchain_core.pydantic_v1 import BaseModel

from datable_ai.core.llm import LLM_TYPE
from datable_ai.structured_output import StructuredOutput


@pytest.fixture
def structured_output_data() -> Dict:
    return {
        "llm_type": LLM_TYPE.OPENAI,
        "prompt_template": "This is a {test} prompt.",
        "output_fields": [
            {"name": "field1", "type": str, "description": "Field 1"},
            {"name": "field2", "type": int, "description": "Field 2"},
        ],
    }


@pytest.fixture
def structured_output(structured_output_data: Dict) -> StructuredOutput:
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_API_MODEL"] = "gpt-4"
    return StructuredOutput(**structured_output_data)


def test_init(structured_output_data: Dict, structured_output: StructuredOutput):
    assert structured_output.llm_type == structured_output_data["llm_type"]
    assert (
        structured_output.prompt_template == structured_output_data["prompt_template"]
    )
    assert structured_output.output_fields == structured_output_data["output_fields"]
    assert isinstance(structured_output.output_model, type(BaseModel))


def test_create_dynamic_model(structured_output: StructuredOutput):
    model = structured_output._create_dynamic_model()

    assert isinstance(model, type(BaseModel))
    assert model.__name__ == "Output"
    assert model.__module__ == "datable_ai.structured_output"
    assert model.__doc__ == "A model representing the output of the LLM"

    for field in structured_output.output_fields:
        assert field["name"] in model.__fields__
        assert model.__fields__[field["name"]].type_ == field["type"]
        assert model.__fields__[field["name"]].field_info.description == field.get(
            "description", ""
        )
