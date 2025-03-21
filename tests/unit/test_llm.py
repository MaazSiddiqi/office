import pytest
from unittest.mock import patch, MagicMock
from llm import LLM


@pytest.fixture
def llm():
    """Fixture to create an LLM instance for testing."""
    return LLM(model="llama3.1:latest", max_tokens=1000, temperature=0.7)


def test_initialization():
    """Test proper initialization of LLM class."""
    llm = LLM(model="mistral", max_tokens=500, temperature=0.5)

    assert llm.model == "mistral"
    assert llm.max_tokens == 500
    assert llm.temperature == 0.5


@patch("requests.post")
def test_successful_response(mock_post, llm):
    """Test successful response generation."""
    # Mock the response
    mock_response = MagicMock()
    mock_response.text = '{"response": "Paris is the capital of France.", "done": true}'
    mock_response.json.return_value = {
        "response": "Paris is the capital of France.",
        "done": True,
    }
    mock_post.return_value = mock_response

    # Test the response
    result = llm.generate_response("What is the capital of France?")

    # Assertions
    assert result == "Paris is the capital of France."
    mock_post.assert_called_once()


@patch("requests.post")
def test_empty_response(mock_post, llm):
    """Test handling of empty response."""
    # Mock the response
    mock_response = MagicMock()
    mock_response.text = '{"response": "", "done": true}'
    mock_response.json.return_value = {"response": "", "done": True}
    mock_post.return_value = mock_response

    # Test the response
    result = llm.generate_response("")

    # Assertions
    assert result == ""
    mock_post.assert_called_once()


@patch("requests.post")
def test_error_handling(mock_post, llm):
    """Test error handling when API call fails."""
    # Mock the response to raise an exception
    mock_post.side_effect = Exception("API Error")

    # Test the response
    result = llm.generate_response("Test prompt")

    # Assertions
    assert result == ""
    mock_post.assert_called_once()


@patch("requests.post")
def test_multiple_lines_response(mock_post, llm):
    """Test handling of multi-line response."""
    # Mock the response with multiple lines
    mock_response = MagicMock()
    mock_response.text = (
        '{"response": "Line 1", "done": false}\n{"response": " Line 2", "done": true}'
    )
    mock_response.json.return_value = {"response": "Line 1 Line 2", "done": True}
    mock_post.return_value = mock_response

    # Test the response
    result = llm.generate_response("Generate multiple lines")

    # Assertions
    assert result == "Line 1 Line 2"
    mock_post.assert_called_once()
