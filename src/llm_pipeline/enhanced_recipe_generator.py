
"""
Step 3: Enhanced Recipe Generation with Attribution.

Generate enhanced recipes with source-review citations and
attribution based on successfully applied recipe changes.
"""

from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from .models import (
    ChangeRecord,
    EnhancedRecipe,
    EnhancementSummary,
    ModificationApplied,
    ModificationEdit,
    ModificationObject,
    Recipe,
    Review,
    SourceReview,
)


class EnhancedRecipeGenerator:
    """Generate enhanced recipes with accurate attribution."""

    def __init__(self, pipeline_version: str = "1.0.0"):
        self.pipeline_version = pipeline_version
        logger.info(
            f"Initialized EnhancedRecipeGenerator v{pipeline_version}"
        )

    def create_source_review(
        self,
        review: Review,
    ) -> SourceReview:
        """Convert a review to its attribution representation."""
        return SourceReview(
            text=review.text,
            reviewer=review.username,
            rating=review.rating,
        )

    def describe_applied_changes(
        self,
        change_records: List[ChangeRecord],
    ) -> str:
        """Describe only changes that were actually applied."""
        if not change_records:
            return "No recipe changes were applied."

        descriptions = []

        for change in change_records:
            if change.operation == "replace":
                descriptions.append(
                    f"Replaced '{change.from_text}' "
                    f"with '{change.to_text}'"
                )

            elif change.operation == "remove":
                descriptions.append(
                    f"Removed '{change.from_text}'"
                )

            elif change.operation == "add":
                if change.from_text:
                    descriptions.append(
                        f"Added '{change.to_text}' "
                        f"after '{change.from_text}'"
                    )
                else:
                    descriptions.append(
                        f"Added '{change.to_text}'"
                    )

            else:
                descriptions.append(
                    f"Applied {change.operation} to the recipe"
                )

        return "; ".join(descriptions) + "."

    def edit_matches_change(
        self,
        edit: ModificationEdit,
        change: ChangeRecord,
    ) -> bool:
        """
        Check whether a recorded change corresponds to an edit.

        Matching is exact and conservative. A successful match
        provides evidence that the proposed edit was applied.
        """
        expected_type = (
            "ingredient"
            if edit.target == "ingredients"
            else "instruction"
        )

        if change.type != expected_type:
            return False

        if edit.operation == "replace":
            return (
                change.operation == "replace"
                and change.from_text == edit.find
                and change.to_text == edit.replace
            )

        if edit.operation == "remove":
            return (
                change.operation == "remove"
                and change.from_text == edit.find
            )

        if edit.operation == "add_after":
            return (
                change.operation == "add"
                and change.from_text == edit.find
                and change.to_text == edit.add
            )

        return False

    def all_proposed_edits_applied(
        self,
        modification: ModificationObject,
        change_records: List[ChangeRecord],
    ) -> bool:
        """
        Return True only when each proposed edit has a unique
        corresponding successful change record.

        Each change record may satisfy at most one proposed edit.
        This avoids treating duplicate edits as successfully
        applied based on a single recorded change.
        """
        if not modification.edits:
            return False

        if len(change_records) < len(modification.edits):
            return False

        used_change_indices = set()

        for edit in modification.edits:
            matched_index = None

            for index, change in enumerate(change_records):
                if index in used_change_indices:
                    continue

                if self.edit_matches_change(edit, change):
                    matched_index = index
                    break

            if matched_index is None:
                return False

            used_change_indices.add(matched_index)

        return True

    def get_safe_reasoning(
        self,
        modification: ModificationObject,
        change_records: List[ChangeRecord],
    ) -> str:
        """
        Preserve original reasoning only when all proposed edits
        can be verified as applied.

        For partial success, use the factual change description
        rather than risk attributing rejected edits.
        """
        if not change_records:
            return "No recipe changes were applied."

        if self.all_proposed_edits_applied(
            modification,
            change_records,
        ):
            reasoning = modification.reasoning.strip()

            if reasoning:
                return reasoning

        return self.describe_applied_changes(change_records)

    def create_modification_applied(
        self,
        modification: ModificationObject,
        source_review: Review,
        change_records: List[ChangeRecord],
    ) -> ModificationApplied:
        """Create attribution for successfully applied changes."""
        return ModificationApplied(
            source_review=self.create_source_review(source_review),
            modification_type=modification.modification_type,
            reasoning=self.get_safe_reasoning(
                modification,
                change_records,
            ),
            changes_made=change_records,
        )

    def calculate_enhancement_summary(
        self,
        modifications_applied: List[ModificationApplied],
    ) -> EnhancementSummary:
        """Summarize applied modifications and their attribution."""
        applied = [
            modification
            for modification in modifications_applied
            if modification.changes_made
        ]

        total_changes = sum(
            len(modification.changes_made)
            for modification in applied
        )

        change_types = list(
            dict.fromkeys(
                modification.modification_type
                for modification in applied
            )
        )

        descriptions = [
            modification.reasoning
            for modification in applied
            if modification.reasoning
        ]

        expected_impact = "; ".join(descriptions[:3])

        if len(descriptions) > 3:
            expected_impact += (
                f" (and {len(descriptions) - 3} "
                "more applied modifications)"
            )

        return EnhancementSummary(
            total_changes=total_changes,
            change_types=change_types,
            expected_impact=(
                expected_impact
                or "No recipe changes were applied."
            ),
        )

    def generate_enhanced_recipe(
        self,
        original_recipe: Recipe,
        modified_recipe: Recipe,
        modification: ModificationObject,
        source_review: Review,
        change_records: List[ChangeRecord],
    ) -> EnhancedRecipe:
        """Generate an enhanced recipe with source attribution."""
        logger.info(
            f"Generating enhanced recipe for: "
            f"{original_recipe.title}"
        )

        modifications_applied = []

        if change_records:
            modification_applied = (
                self.create_modification_applied(
                    modification,
                    source_review,
                    change_records,
                )
            )
            modifications_applied.append(
                modification_applied
            )

        enhancement_summary = (
            self.calculate_enhancement_summary(
                modifications_applied
            )
        )

        enhanced_recipe = EnhancedRecipe(
            recipe_id=f"{original_recipe.recipe_id}_enhanced",
            original_recipe_id=original_recipe.recipe_id,
            title=(
                f"{original_recipe.title} "
                "(Community Enhanced)"
            ),
            ingredients=modified_recipe.ingredients,
            instructions=modified_recipe.instructions,
            modifications_applied=modifications_applied,
            enhancement_summary=enhancement_summary,
            description=original_recipe.description,
            servings=original_recipe.servings,
            prep_time=getattr(
                original_recipe, "prep_time", None
            ),
            cook_time=getattr(
                original_recipe, "cook_time", None
            ),
            total_time=getattr(
                original_recipe, "total_time", None
            ),
            created_at=datetime.now().isoformat(),
            pipeline_version=self.pipeline_version,
        )

        logger.info(
            f"Generated enhanced recipe with "
            f"{enhancement_summary.total_changes} changes "
            f"from {len(modifications_applied)} modifications"
        )

        return enhanced_recipe

    def generate_comparison_data(
        self,
        original_recipe: Recipe,
        enhanced_recipe: EnhancedRecipe,
    ) -> Dict[str, Any]:
        """Compare original and enhanced recipes."""
        return {
            "original": {
                "title": original_recipe.title,
                "ingredients": original_recipe.ingredients,
                "instructions": original_recipe.instructions,
                "servings": original_recipe.servings,
            },
            "enhanced": {
                "title": enhanced_recipe.title,
                "ingredients": enhanced_recipe.ingredients,
                "instructions": enhanced_recipe.instructions,
                "servings": enhanced_recipe.servings,
            },
            "changes": {
                "total_modifications": len(
                    enhanced_recipe.modifications_applied
                ),
                "total_changes": (
                    enhanced_recipe.enhancement_summary.total_changes
                ),
                "change_types": (
                    enhanced_recipe.enhancement_summary.change_types
                ),
                "expected_impact": (
                    enhanced_recipe.enhancement_summary.expected_impact
                ),
            },
            "citations": [
                {
                    "reviewer": mod.source_review.reviewer,
                    "rating": mod.source_review.rating,
                    "modification_type": mod.modification_type,
                    "reasoning": mod.reasoning,
                    "changes": [
                        {
                            "type": change.type,
                            "from": change.from_text,
                            "to": change.to_text,
                            "operation": change.operation,
                        }
                        for change in mod.changes_made
                    ],
                }
                for mod in enhanced_recipe.modifications_applied
            ],
        }

    def save_enhanced_recipe(
        self,
        enhanced_recipe: EnhancedRecipe,
        output_path: str,
    ) -> str:
        """Save an enhanced recipe as formatted JSON."""
        import json
        import os

        output_directory = os.path.dirname(output_path)

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                enhanced_recipe.model_dump(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            f"Saved enhanced recipe to: {output_path}"
        )

        return output_path




