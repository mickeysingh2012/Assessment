
"""
Step 2: Recipe Modification

Safely applies structured modifications to recipe ingredients and
instructions while preserving the original recipe and recording
only actual changes.
"""

import copy
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from loguru import logger

from .models import (
    ModificationObject,
    ModificationEdit,
    Recipe,
    ChangeRecord,
)


class RecipeModifier:
    """Apply validated recipe edits without unsafe fuzzy modifications."""

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        logger.info(
            f"Initialized RecipeModifier with similarity threshold: "
            f"{similarity_threshold}"
        )

    def find_best_match(
        self,
        target: str,
        candidates: List[str],
    ) -> Tuple[Optional[str], Optional[int], float]:
        """
        Find the closest candidate for diagnostics.

        This method preserves the original public interface.
        Fuzzy matches are NOT automatically accepted for edits.
        """
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
        """
        Locate a target safely.

        First prefer an exact list-entry match.
        Then allow a literal substring match within one entry.

        Reject ambiguous matches to avoid editing the wrong item.
        """
        if not target or not target.strip():
            return None, None

        # Prefer a full entry match.
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

        # Permit a literal substring in a longer instruction
        # or ingredient entry, but never a fuzzy-only match.
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

    def apply_edit(
        self,
        edit: ModificationEdit,
        recipe_content: List[str],
    ) -> Tuple[List[str], List[ChangeRecord]]:
        """
        Apply one edit safely.

        Returns:
            (modified_content, change_records)

        No-op, invalid, ambiguous, and fuzzy-only edits return
        unchanged content and an empty list of change records.
        """
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

            # Replace only the literal text requested.
            new_text = original_text.replace(
                edit.find,
                edit.replace,
            )

            # Never count unchanged content as a modification.
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

            # Avoid adding an identical entry twice immediately
            # after the same anchor.
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
            # Removal requires the complete entry to match.
            # Never delete an ingredient based on a partial
            # phrase or a fuzzy match.
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

            logger.info(
                f"Removed '{removed_text}'"
            )

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
        """Apply all edits from one modification to a recipe copy."""
        logger.info(
            f"Applying {modification.modification_type} "
            f"with {len(modification.edits)} edits"
        )

        modified_recipe = copy.deepcopy(recipe)
        modified_recipe.recipe_id = f"{recipe.recipe_id}_modified"

        all_change_records: List[ChangeRecord] = []

        for edit in modification.edits:
            if edit.target == "ingredients":
                (
                    modified_recipe.ingredients,
                    change_records,
                ) = self.apply_edit(
                    edit,
                    modified_recipe.ingredients,
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
        """Apply multiple modifications sequentially."""
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

            current_recipe, changes = self.apply_modification(
                current_recipe,
                modification,
            )

            all_change_records.append(changes)

        return current_recipe, all_change_records

    def validate_modification_safety(
        self,
        modification: ModificationObject,
        recipe: Recipe,
    ) -> Tuple[bool, List[str]]:
        """
        Validate whether edits can be safely applied.

        Uses the same matching rules as apply_edit().
        """
        warnings: List[str] = []

        for edit in modification.edits:
            target_content = (
                recipe.ingredients
                if edit.target == "ingredients"
                else recipe.instructions
            )

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
                if edit.replace is None or not edit.replace.strip():
                    warnings.append(
                        f"Missing replacement for '{edit.find}'"
                    )
                elif (
                    matched_text.replace(
                        edit.find,
                        edit.replace,
                    ) == matched_text
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

        return len(warnings) == 0, warnings
