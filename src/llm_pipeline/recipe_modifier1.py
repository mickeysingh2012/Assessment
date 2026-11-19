
"""
Step 1: Tweak Extraction & Parsing

Extract structured recipe modifications from community reviews.

Improvements:
- Use gpt-4o-mini instead of the deprecated default model.
- Reject duplicate JSON keys to prevent silent loss of edits.
- Validate the extracted modification with Pydantic.
- Reject empty edit lists and incomplete edit operations.
- Retry malformed AI responses with corrective feedback.
- Stop immediately on permanent API errors.
- Preserve existing pipeline-compatible public methods.
"""

import json
import os
import random
from typing import Optional

from loguru import logger
from openai import (
    OpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
from pydantic import ValidationError

from .models import ModificationObject, Recipe, Review
from .prompts import build_simple_prompt


def reject_duplicate_json_keys(pairs):
    """
    Parse a JSON object without silently overwriting duplicate keys.

    The standard json.loads() function keeps the last occurrence
    of a duplicate key. That can cause recipe edits to disappear.
    """
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate JSON key detected: '{key}'"
            )

        result[key] = value

    return result


def validate_extracted_modification(modification):
    """
    Perform additional validation beyond the Pydantic schema.

    Pydantic checks field types and allowed operation names.
    This function checks that edits are usable.
    """
    if not modification.edits:
        raise ValueError(
            "AI returned an empty edits list"
        )

    for index, edit in enumerate(modification.edits, start=1):
        if not edit.find or not edit.find.strip():
            raise ValueError(
                f"Edit {index} has an empty find field"
            )

        if edit.operation == "replace":
            if edit.replace is None or not edit.replace.strip():
                raise ValueError(
                    f"Edit {index}: replace operation "
                    "requires replacement text"
                )

        elif edit.operation == "add_after":
            if edit.add is None or not edit.add.strip():
                raise ValueError(
                    f"Edit {index}: add_after operation "
                    "requires addition text"
                )

        elif edit.operation == "remove":
            # A remove operation needs only the find field.
            pass

    return modification


class TweakExtractor:
    """Extract structured modifications from recipe reviews."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the extractor.

        Args:
            api_key: OpenAI API key, or OPENAI_API_KEY from environment.
            model: OpenAI model used for extraction.
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=0,
        )

        self.model = model

        logger.info(
            f"Initialized TweakExtractor with model: {model}"
        )

    def extract_modification(
        self,
        review: Review,
        recipe: Recipe,
        max_retries: int = 2,
    ) -> Optional[ModificationObject]:
        """
        Extract one ModificationObject containing all relevant edits.

        max_retries controls the number of additional attempts after
        the initial request. For example, max_retries=2 allows up
        to three requests.

        Returns None when extraction cannot be completed safely.
        """
        if not review.has_modification:
            logger.warning(
                "Review has no modification flag set"
            )
            return None

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative"
            )

        prompt = build_simple_prompt(
            review.text,
            recipe.title,
            recipe.ingredients,
            recipe.instructions,
        )

        # Additional guidance to reduce malformed output.
        extraction_instructions = """
Return exactly one valid JSON object.

Requirements:
1. Include exactly one "modification_type" key.
2. Include exactly one "reasoning" key.
3. Include exactly one "edits" key.
4. Put ALL atomic recipe changes into the SAME "edits" array.
5. Do not create multiple "edits" arrays.
6. Include every concrete change the reviewer actually made.
7. Do not include changes merely suggested for a future attempt.
8. Do not create a replacement if the old and new text are identical.
9. Copy the "find" text exactly from the original recipe.
10. For replace operations, provide "replace".
11. For add_after operations, provide "add".
12. For remove operations, provide "find".
13. Do not invent changes not supported by the review.
14. Do not include Markdown or explanatory text outside JSON.

Use the existing schema:
{
  "modification_type": "quantity_adjustment",
  "reasoning": "Brief explanation of the changes",
  "edits": [
    {
      "target": "ingredients",
      "operation": "replace",
      "find": "original ingredient text",
      "replace": "updated ingredient text"
    }
  ]
}

If the review contains multiple types of changes, put all
of them into the same edits array and select the category
that best represents the overall modification.
"""

        messages = [
            {
                "role": "system",
                "content": extraction_instructions,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        logger.debug(
            "Extracting modification from review: {}..."
            .format(review.text[:100])
        )

        total_attempts = max_retries + 1

        for attempt in range(total_attempts):
            attempt_number = attempt + 1

            try:
                logger.info(
                    f"Extraction attempt {attempt_number}/"
                    f"{total_attempts}"
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_object"
                    },
                    temperature=0.1,
                    max_tokens=1500,
                )

                if not response.choices:
                    raise ValueError(
                        "AI response contains no choices"
                    )

                choice = response.choices[0]
                raw_output = choice.message.content

                if choice.finish_reason == "length":
                    raise ValueError(
                        "AI response was truncated by the token limit"
                    )

                if not raw_output or not raw_output.strip():
                    raise ValueError(
                        "AI returned an empty response"
                    )

                logger.debug(
                    f"LLM raw output: {raw_output}"
                )

                # Reject duplicate keys instead of losing edits.
                modification_data = json.loads(
                    raw_output,
                    object_pairs_hook=reject_duplicate_json_keys,
                )

                if not isinstance(modification_data, dict):
                    raise ValueError(
                        "AI response must be a JSON object"
                    )

                required_keys = {
                    "modification_type",
                    "reasoning",
                    "edits",
                }

                missing_keys = (
                    required_keys - modification_data.keys()
                )

                if missing_keys:
                    raise ValueError(
                        "Missing required JSON keys: "
                        + ", ".join(sorted(missing_keys))
                    )

                # Validate against the existing Pydantic model.
                modification = ModificationObject.model_validate(
                    modification_data
                )

                validate_extracted_modification(
                    modification
                )

                # Remove exact no-op replacements.
                valid_edits = []

                for edit in modification.edits:
                    if (
                        edit.operation == "replace"
                        and edit.find == edit.replace
                    ):
                        logger.warning(
                            "Skipping no-op replacement: "
                            f"'{edit.find}'"
                        )
                        continue

                    valid_edits.append(edit)

                if not valid_edits:
                    raise ValueError(
                        "No actionable edits remained after validation"
                    )

                modification.edits = valid_edits

                logger.info(
                    "Successfully extracted "
                    f"{modification.modification_type} "
                    "modification with "
                    f"{len(modification.edits)} edits"
                )

                return modification

            except (
                AuthenticationError,
                PermissionDeniedError,
                NotFoundError,
                BadRequestError,
            ) as error:
                # Retrying invalid credentials, unavailable models,
                # or invalid requests will not fix the problem.
                logger.error(
                    "Permanent OpenAI API error. "
                    f"Stopping extraction: {error}"
                )
                return None

            except RateLimitError as error:
                error_code = None

                try:
                    error_body = getattr(
                        error, "body", None
                    )

                    if isinstance(error_body, dict):
                        error_code = error_body.get("code")

                        if not error_code:
                            nested_error = error_body.get(
                                "error", {}
                            )

                            if isinstance(nested_error, dict):
                                error_code = nested_error.get(
                                    "code"
                                )
                except (TypeError, AttributeError):
                    pass

                if error_code in {
                    "insufficient_quota",
                    "credit_balance_exhausted",
                }:
                    logger.error(
                        "OpenAI API credits are exhausted. "
                        "Stopping without retrying."
                    )
                    return None

                logger.warning(
                    f"Attempt {attempt_number}: "
                    f"Rate limit encountered: {error}"
                )

            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            ) as error:
                logger.warning(
                    f"Attempt {attempt_number}: "
                    f"Temporary API failure: {error}"
                )

            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
            ) as error:
                logger.warning(
                    f"Attempt {attempt_number}: "
                    f"Invalid AI output: {error}"
                )

                # Give the model explicit correction instructions
                # for the next attempt.
                if attempt < max_retries:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response could not "
                                "be accepted because: "
                                f"{error}. "
                                "Regenerate the complete JSON object. "
                                "Use exactly one edits array containing "
                                "all valid changes. Do not repeat keys. "
                                "Do not include unchanged replacements."
                            ),
                        }
                    )

            except Exception as error:
                logger.exception(
                    "Unexpected extraction error: "
                    f"{error}"
                )
                return None

            if attempt < max_retries:
                logger.info(
                    "Retrying extraction..."
                )

        logger.error(
            "Extraction failed after "
            f"{total_attempts} attempts"
        )

        return None

    def extract_single_modification(
        self,
        reviews: list[Review],
        recipe: Recipe,
    ) -> (
        tuple[ModificationObject, Review]
        | tuple[None, None]
    ):
        """
        Extract from one randomly selected review that has
        its modification flag set.

        This preserves the original pipeline behavior.
        """
        modification_reviews = [
            review
            for review in reviews
            if review.has_modification
        ]

        if not modification_reviews:
            logger.warning(
                "No reviews with modifications found"
            )
            return None, None

        selected_review = random.choice(
            modification_reviews
        )

        logger.info(
            f"Selected review: "
            f"{selected_review.text[:100]}..."
        )

        modification = self.extract_modification(
            selected_review,
            recipe,
        )

        if modification is not None:
            logger.info(
                "Successfully extracted modification "
                "from selected review"
            )
            return modification, selected_review

        logger.warning(
            "Failed to extract modification "
            "from selected review"
        )

        return None, None

    def test_extraction(
        self,
        review_text: str,
        recipe_data: dict,
    ) -> Optional[ModificationObject]:
        """
        Test extraction using raw review text and recipe data.

        Preserved for compatibility with existing tests.
        """
        review = Review(
            text=review_text,
            has_modification=True,
        )

        recipe = Recipe(
            recipe_id=recipe_data.get(
                "recipe_id", "test"
            ),
            title=recipe_data.get(
                "title", "Test Recipe"
            ),
            ingredients=recipe_data.get(
                "ingredients", []
            ),
            instructions=recipe_data.get(
                "instructions", []
            ),
        )

        return self.extract_modification(
            review,
            recipe,
        )
