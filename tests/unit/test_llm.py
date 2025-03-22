import pytest
from unittest.mock import patch, MagicMock
from llm import LLM
from typing import Generator
import json


@pytest.fixture
def llm():
    """Fixture to create an LLM instance for testing."""
    return LLM(
        model="llama3.1:latest",
        system_prompt="You are a helpful assistant.",
        max_tokens=1000,
        temperature=0.7,
    )


@patch("requests.post")
def test_successful_response(mock_post, llm):
    """Test successful response generation."""
    # Create a mock response for streaming
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        json.dumps({"response": "Paris"}).encode(),
        json.dumps({"response": " is"}).encode(),
        json.dumps({"response": " the"}).encode(),
        json.dumps({"response": " capital"}).encode(),
        json.dumps({"response": " of"}).encode(),
        json.dumps({"response": " France."}).encode(),
    ]
    mock_post.return_value = mock_response

    # Test the response
    result = llm.generate_response_stream("What is the capital of France?")
    response = "".join(chunk for chunk in result)

    # Assertions
    assert response == "Paris is the capital of France."
    mock_post.assert_called_once()


@patch("requests.post")
def test_empty_response(mock_post, llm):
    """Test handling of empty response."""
    # Create a mock response with empty content
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [json.dumps({"response": ""}).encode()]
    mock_post.return_value = mock_response

    # Test the response
    result = llm.generate_response_stream("")
    response = "".join(chunk for chunk in result)

    # Assertions
    assert response == ""
    mock_post.assert_called_once()


@patch("requests.post")
def test_error_handling(mock_post, llm):
    """Test error handling when API call fails.

    The generate_response_stream method returns a generator,
    but the exception is raised when we try to iterate through it.
    """
    mock_post.side_effect = Exception("API Error")

    result = llm.generate_response_stream("Test prompt")
    assert isinstance(result, Generator)

    with pytest.raises(Exception) as e:
        next(result)

    assert "API Error" in str(e.value)
    mock_post.assert_called_once()


@patch("requests.post")
def test_multiple_lines_response(mock_post, llm):
    """Test handling of multi-line response."""
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        json.dumps({"response": "Line 1"}).encode(),
        json.dumps({"response": " Line 2"}).encode(),
    ]
    mock_post.return_value = mock_response

    result = llm.generate_response_stream("Generate multiple lines")
    response = "".join(chunk for chunk in result)

    assert response == "Line 1 Line 2"
    mock_post.assert_called_once()
