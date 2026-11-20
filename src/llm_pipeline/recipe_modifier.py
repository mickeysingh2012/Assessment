

"""
Step 2: Recipe Modification

Apply structured recipe edits safely, preserve the original recipe,
and record only changes that actually occurred.

Ingredient removals are checked against the final instructions to
avoid leaving explicit references to removed ingredients.
"""

import copy
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from loguru import logger

from .models import (
    ChangeRecord,
    ModificationEdit,
    ModificationObject,
    Recipe,
)


class RecipeModifier:
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        logger.info(
            "Initialized RecipeModifier with similarity threshold: "
            f"{similarity_threshold}"
        )

    def find_best_match(
        self,
        target: str,
        candidates: List[str],
    ) -> Tuple[Optional[str], Optional[int], float]:
        """Find the closest candidate for diagnostic purposes."""
        if not target or not candidates:
            return None, None, 0.0

        best_match = None
        best_index = None
        best_score = 0.0

        for index, candidate in enumerate(candidates):
            score = SequenceMatcher(
                None,
                target.lower(),
                candidate.lower(),
            ).ratio()

            if score > best_score:
                best_match = candidate
                best_index = index
                best_score = score

        if best_score >= self.similarity_threshold:
            return best_match, best_index, best_score

        return None, None, best_score

    def _find_exact_target(
        self,
        target: str,
        candidates: List[str],
    ) -> Tuple[Optional[int], Optional[str]]:
        """Find one exact entry or one unambiguous substring match."""
        if not target or not target.strip():
            return None, None

        full_matches = [
            index
            for index, candidate in enumerate(candidates)
            if candidate == target
        ]

        if len(full_matches) == 1:
            index = full_matches[0]
            return index, candidates[index]

        if len(full_matches) > 1:
            logger.warning(
                f"Ambiguous exact target '{target}': "
                f"{len(full_matches)} matches"
            )
            return None, None

        substring_matches = [
            index
            for index, candidate in enumerate(candidates)
            if target in candidate
        ]

        if len(substring_matches) == 1:
            index = substring_matches[0]
            return index, candidates[index]

        if len(substring_matches) > 1:
            logger.warning(
                f"Ambiguous substring target '{target}': "
                f"{len(substring_matches)} matches"
            )

        return None, None

    def _ingredient_name(self, ingredient: str) -> str:
        """
        Remove a leading quantity and measurement unit.

        Example:
            '2 teaspoons hot water' -> 'hot water'

        This is a heuristic, not full ingredient parsing.
        """
        name = ingredient.lower().strip()

        name = re.sub(
            r"^\s*[\d./¼½¾⅓⅔⅛⅜⅝⅞\s-]+\s*"
            r"(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|"
            r"ounces?|oz|grams?|g|ml|liters?|l)?\s*",
            "",
            name,
        ).strip()

        return name

    def _ingredient_referenced_in_instructions(
        self,
        ingredient: str,
        instructions: List[str],
    ) -> bool:
        """
        Detect explicit ingredient references in instructions.

        Supports complete ingredient names and names with common
        leading preparation descriptors removed.

        This remains a conservative heuristic. It does not attempt
        semantic matching or synonym detection.
        """
        ingredient_name = self._ingredient_name(ingredient)

        if not ingredient_name:
            return False

        preparation_descriptors = {
            "chopped",
            "diced",
            "minced",
            "sliced",
            "shredded",
            "grated",
            "crushed",
            "ground",
            "finely",
            "roughly",
            "freshly",
        }

        words = ingredient_name.split()

        candidate_names = {ingredient_name}

        # Remove leading preparation descriptors while preserving
        # the original complete ingredient name.
        while words and words[0] in preparation_descriptors:
            words = words[1:]

            if words:
                candidate_names.add(" ".join(words))

        for candidate in candidate_names:
            pattern = (
                r"(?<!\w)"
                + re.escape(candidate)
                + r"(?!\w)"
            )

            if any(
                re.search(pattern, instruction.lower())
                is not None
                for instruction in instructions
            ):
                return True

        return False

    def apply_edit(
        self,
        edit: ModificationEdit,
        recipe_content: List[str],
    ) -> Tuple[List[str], List[ChangeRecord]]:
        """Apply one edit to a copy of an ingredient or instruction list."""
        modified_content = copy.deepcopy(recipe_content)
        change_records: List[ChangeRecord] = []

        logger.debug(
            f"Applying {edit.operation} edit: find='{edit.find}'"
        )

        index, matched_text = self._find_exact_target(
            edit.find,
            modified_content,
        )

        if index is None or matched_text is None:
            logger.warning(
                f"Skipping {edit.operation}: exact target "
                f"'{edit.find}' not found uniquely in {edit.target}"
            )
            return modified_content, change_records

        change_type = (
            "ingredient"
            if edit.target == "ingredients"
            else "instruction"
        )

        if edit.operation == "replace":
            if edit.replace is None or not edit.replace.strip():
                logger.warning(
                    f"Skipping replace: missing replacement "
                    f"for '{edit.find}'"
                )
                return modified_content, change_records

            original_text = modified_content[index]
            new_text = original_text.replace(
                edit.find,
                edit.replace,
            )

            if new_text == original_text:
                logger.info(
                    f"Skipping no-op replacement: '{edit.find}'"
                )
                return modified_content, change_records

            modified_content[index] = new_text

            change_records.append(
                ChangeRecord(
                    type=change_type,
                    from_text=original_text,
                    to_text=new_text,
                    operation="replace",
                )
            )

            logger.info(
                f"Replaced '{edit.find}' with '{edit.replace}'"
            )

        elif edit.operation == "add_after":
            if edit.add is None or not edit.add.strip():
                logger.warning(
                    f"Skipping add_after: no content to add "
                    f"after '{edit.find}'"
                )
                return modified_content, change_records

            next_index = index + 1

            if (
                next_index < len(modified_content)
                and modified_content[next_index] == edit.add
            ):
                logger.info(
                    f"Skipping duplicate addition: '{edit.add}'"
                )
                return modified_content, change_records

            modified_content.insert(next_index, edit.add)

            change_records.append(
                ChangeRecord(
                    type=change_type,
                    from_text="",
                    to_text=edit.add,
                    operation="add",
                )
            )

            logger.info(
                f"Added '{edit.add}' after '{edit.find}'"
            )

        elif edit.operation == "remove":
            if matched_text != edit.find:
                logger.warning(
                    f"Skipping unsafe removal: '{edit.find}' "
                    f"is not a complete {change_type} entry"
                )
                return modified_content, change_records

            removed_text = modified_content.pop(index)

            change_records.append(
                ChangeRecord(
                    type=change_type,
                    from_text=removed_text,
                    to_text="",
                    operation="remove",
                )
            )

            logger.info(f"Removed '{removed_text}'")

        else:
            logger.warning(
                f"Unsupported edit operation: {edit.operation}"
            )

        return modified_content, change_records

    def apply_modification(
        self,
        recipe: Recipe,
        modification: ModificationObject,
    ) -> Tuple[Recipe, List[ChangeRecord]]:
        """
        Apply all edits to a recipe copy.

        After applying the edits, reject ingredient removals that
        leave explicit references in the final instructions.

        Valid unrelated edits are preserved.
        """
        logger.info(
            f"Applying {modification.modification_type} "
            f"with {len(modification.edits)} edits"
        )

        modified_recipe = copy.deepcopy(recipe)
        modified_recipe.recipe_id = (
            f"{recipe.recipe_id}_modified"
        )

        all_change_records: List[ChangeRecord] = []

        # Track successful removals, including original positions.
        removed_ingredients = []

        for edit in modification.edits:
            if edit.target == "ingredients":
                before_ingredients = (
                    modified_recipe.ingredients.copy()
                )

                (
                    modified_recipe.ingredients,
                    change_records,
                ) = self.apply_edit(
                    edit,
                    modified_recipe.ingredients,
                )

                if (
                    edit.operation == "remove"
                    and change_records
                ):
                    removed_text = change_records[0].from_text

                    removed_ingredients.append(
                        {
                            "text": removed_text,
                            "original_index": (
                                before_ingredients.index(
                                    removed_text
                                )
                            ),
                            "records": change_records,
                        }
                    )

            elif edit.target == "instructions":
                (
                    modified_recipe.instructions,
                    change_records,
                ) = self.apply_edit(
                    edit,
                    modified_recipe.instructions,
                )

            else:
                logger.warning(
                    f"Unknown edit target: {edit.target}"
                )
                continue

            all_change_records.extend(change_records)

        # Evaluate ingredient removals after instruction edits.
        for removal in reversed(removed_ingredients):
            ingredient = removal["text"]

            # If another edit reintroduced the ingredient,
            # the removal no longer leaves it absent.
            if ingredient in modified_recipe.ingredients:
                continue

            if self._ingredient_referenced_in_instructions(
                ingredient,
                modified_recipe.instructions,
            ):
                logger.warning(
                    f"Rejecting removal of '{ingredient}': "
                    "still referenced in cooking instructions"
                )

                insertion_index = min(
                    removal["original_index"],
                    len(modified_recipe.ingredients),
                )

                modified_recipe.ingredients.insert(
                    insertion_index,
                    ingredient,
                )

                for record in removal["records"]:
                    if record in all_change_records:
                        all_change_records.remove(record)

        logger.info(
            f"Applied modification: "
            f"{len(all_change_records)} actual changes made"
        )

        return modified_recipe, all_change_records

    def apply_modifications_batch(
        self,
        recipe: Recipe,
        modifications: List[ModificationObject],
    ) -> Tuple[Recipe, List[List[ChangeRecord]]]:
        """Apply multiple modifications in sequence."""
        current_recipe = recipe
        all_change_records: List[List[ChangeRecord]] = []

        logger.info(
            f"Applying {len(modifications)} modifications"
        )

        for index, modification in enumerate(modifications):
            logger.info(
                f"Applying modification "
                f"{index + 1}/{len(modifications)}"
            )

            current_recipe, changes = (
                self.apply_modification(
                    current_recipe,
                    modification,
                )
            )

            all_change_records.append(changes)

        return current_recipe, all_change_records

    def validate_modification_safety(
        self,
        modification: ModificationObject,
        recipe: Recipe,
    ) -> Tuple[bool, List[str]]:
        """Validate edit targets and basic edit safety."""
        warnings: List[str] = []

        for edit in modification.edits:
            if edit.target == "ingredients":
                target_content = recipe.ingredients
            elif edit.target == "instructions":
                target_content = recipe.instructions
            else:
                warnings.append(
                    f"Unknown edit target: '{edit.target}'"
                )
                continue

            index, matched_text = self._find_exact_target(
                edit.find,
                target_content,
            )

            if index is None:
                warnings.append(
                    f"Cannot uniquely find exact target "
                    f"'{edit.find}' in {edit.target}"
                )
                continue

            if edit.operation == "replace":
                if (
                    edit.replace is None
                    or not edit.replace.strip()
                ):
                    warnings.append(
                        f"Missing replacement for '{edit.find}'"
                    )

                elif (
                    matched_text.replace(
                        edit.find,
                        edit.replace,
                    )
                    == matched_text
                ):
                    warnings.append(
                        f"No-op replacement for '{edit.find}'"
                    )

            elif edit.operation == "add_after":
                if edit.add is None or not edit.add.strip():
                    warnings.append(
                        f"Missing addition after '{edit.find}'"
                    )

            elif edit.operation == "remove":
                if matched_text != edit.find:
                    warnings.append(
                        f"Unsafe partial removal: '{edit.find}'"
                    )

                elif (
                    edit.target == "ingredients"
                    and self._ingredient_referenced_in_instructions(
                        matched_text,
                        recipe.instructions,
                    )
                ):
                    # Preliminary diagnostic only.
                    # The final check happens in apply_modification
                    # after all instruction edits have been applied.
                    logger.debug(
                        f"Ingredient '{matched_text}' is "
                        "referenced in the original instructions"
                    )

            else:
                warnings.append(
                    f"Unsupported operation: '{edit.operation}'"
                )

        return len(warnings) == 0, warnings
