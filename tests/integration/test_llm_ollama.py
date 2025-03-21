import pytest
import requests
from llm import LLM


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


def test_ollama_responds_to_prompt(llm):
    """Test that Ollama responds to a prompt."""
    result = llm.generate_response("What is the capital of France?")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
