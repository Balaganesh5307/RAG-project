import time
import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


# Default Groq model configuration

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TEMPERATURE = 0

# Retry configuration

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 5


def create_model(model=None, temperature=None):
    """Create a ChatGroq model instance."""

    return ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE
    )


def invoke_with_retry(chain, inputs, max_retries=None, initial_backoff=None):
    """
    Invoke a LangChain chain with automatic retry on rate-limit errors.

    Retries up to max_retries times with exponential backoff
    when a 429 rate-limit error is encountered.
    """

    retries = max_retries if max_retries is not None else MAX_RETRIES
    backoff = initial_backoff if initial_backoff is not None else INITIAL_BACKOFF_SECONDS

    for attempt in range(retries + 1):

        try:
            return chain.invoke(inputs)

        except Exception as error:

            error_message = str(error).lower()

            is_rate_limit = (
                "429" in error_message
                or "rate" in error_message
                or "too many" in error_message
            )

            if is_rate_limit and attempt < retries:

                wait_time = backoff * (2 ** attempt)

                logger.warning(
                    "Rate limit hit (attempt %d/%d). "
                    "Retrying in %d seconds...",
                    attempt + 1,
                    retries + 1,
                    wait_time
                )

                time.sleep(wait_time)
                continue

            # Not a rate-limit error, or all retries exhausted
            raise
