import pytest
import requests
from llm import LLM
from typing import Generator


@pytest.fixture
def llm():
    """Fixture to create an LLM instance for testing."""
    return LLM(model="llama3.1:latest", max_tokens=100, temperature=0.7)


@pytest.fixture(autouse=True)
def check_ollama():
    """Check if Ollama is running before running integration tests.
    Fails the test if Ollama is not running."""
    try:
        response = requests.get("http://localhost:11434/api/version")
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        pytest.fail("Ollama is not running. Please start it with 'ollama serve'")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Failed to connect to Ollama: {str(e)}")


def test_ollama_stream_response(llm):
    """Test that Ollama streams a response to a prompt."""
    prompt = "What is the capital of France? Answer in one word."
    response = llm.generate_response_stream(prompt)

    # Validate the complete response
    assert response is not None
    assert isinstance(response, Generator)

    # Collect the full response from the stream
    full_response = ""
    for chunk in response:
        assert isinstance(chunk, str)
        full_response += chunk

    assert full_response is not None
    assert len(full_response) > 0
