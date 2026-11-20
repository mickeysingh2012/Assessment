
"""
Offline regression tests for the Casper Studios recipe pipeline.

Run all tests:
    uv run python -m unittest -v test_recipe_regressions

These tests do not make OpenAI API calls.
"""

import json
import unittest

from llm_pipeline.models import (
    ChangeRecord,
    ModificationEdit,
    ModificationObject,
    Recipe,
)
from llm_pipeline.recipe_modifier import RecipeModifier
from llm_pipeline.enhanced_recipe_generator import EnhancedRecipeGenerator
from llm_pipeline.tweak_extractor import reject_duplicate_json_keys


class TestRecipeRegressions(unittest.TestCase):

    def setUp(self):
        self.modifier = RecipeModifier()

        self.recipe = Recipe(
            recipe_id="10813",
            title="Best Chocolate Chip Cookies",
            ingredients=[
                "1 cup white sugar",
                "1 cup packed brown sugar",
                "3 cups all-purpose flour",
                "2 teaspoons hot water",
            ],
            instructions=[
                "Preheat oven to 350 degrees F.",
                "Bake for 10 minutes.",
            ],
        )

    def make_modification(self, edits):
        return ModificationObject(
            modification_type="quantity_adjustment",
            reasoning="Test recipe changes",
            edits=edits,
        )

    def test_duplicate_json_keys_rejected(self):
        """Duplicate edits arrays must not overwrite data."""
        malformed_json = (
            '{"edits": [{"find": "brown sugar"}], '
            '"edits": [{"find": "flour"}]}'
        )

        with self.assertRaisesRegex(
            ValueError, "Duplicate JSON key"
        ):
            json.loads(
                malformed_json,
                object_pairs_hook=reject_duplicate_json_keys,
            )

    def test_valid_json_preserves_multiple_edits(self):
        """A valid JSON edits array must retain every entry."""
        valid_json = (
            '{"edits": ['
            '{"find": "brown sugar"},'
            '{"find": "flour"},'
            '{"find": "cinnamon"}'
            ']}'
        )

        result = json.loads(
            valid_json,
            object_pairs_hook=reject_duplicate_json_keys,
        )

        self.assertEqual(len(result["edits"]), 3)

    def test_exact_replacement(self):
        """An exact ingredient match should be replaced."""
        edit = ModificationEdit(
            target="ingredients",
            operation="replace",
            find="1 cup packed brown sugar",
            replace="0.5 cup packed brown sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit,
            self.recipe.ingredients,
        )

        self.assertIn(
            "0.5 cup packed brown sugar",
            result,
        )
        self.assertEqual(len(changes), 1)

    def test_no_op_replacement_not_counted(self):
        """Replacing an ingredient with itself is not a change."""
        edit = ModificationEdit(
            target="ingredients",
            operation="replace",
            find="1 cup white sugar",
            replace="1 cup white sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit,
            self.recipe.ingredients,
        )

        self.assertEqual(
            result,
            self.recipe.ingredients,
        )
        self.assertEqual(len(changes), 0)

    def test_fuzzy_only_replacement_rejected(self):
        """Similar text must not trigger an unsafe replacement."""
        edit = ModificationEdit(
            target="ingredients",
            operation="replace",
            find="1 cup brown sugar",
            replace="0.5 cup brown sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit,
            self.recipe.ingredients,
        )

        self.assertEqual(
            result,
            self.recipe.ingredients,
        )
        self.assertEqual(len(changes), 0)

    def test_multiple_edits_applied(self):
        """All valid edits from one modification should apply."""
        modification = self.make_modification([
            ModificationEdit(
                target="ingredients",
                operation="replace",
                find="1 cup packed brown sugar",
                replace="0.5 cup packed brown sugar",
            ),
            ModificationEdit(
                target="ingredients",
                operation="replace",
                find="3 cups all-purpose flour",
                replace="2.5 cups all-purpose flour",
            ),
            ModificationEdit(
                target="ingredients",
                operation="add_after",
                find="2 teaspoons hot water",
                add="a tiny dash of cinnamon",
            ),
        ])

        result, changes = self.modifier.apply_modification(
            self.recipe,
            modification,
        )

        self.assertEqual(len(changes), 3)

        self.assertIn(
            "0.5 cup packed brown sugar",
            result.ingredients,
        )
        self.assertIn(
            "2.5 cups all-purpose flour",
            result.ingredients,
        )
        self.assertIn(
            "a tiny dash of cinnamon",
            result.ingredients,
        )

    def test_original_recipe_not_mutated(self):
        """Modifying a recipe must preserve the original."""
        original_ingredients = self.recipe.ingredients.copy()

        modification = self.make_modification([
            ModificationEdit(
                target="ingredients",
                operation="replace",
                find="3 cups all-purpose flour",
                replace="2.5 cups all-purpose flour",
            )
        ])

        modified_recipe, _ = self.modifier.apply_modification(
            self.recipe,
            modification,
        )

        self.assertEqual(
            self.recipe.ingredients,
            original_ingredients,
        )
        self.assertNotEqual(
            modified_recipe.ingredients,
            self.recipe.ingredients,
        )

    def test_removal_of_nonexistent_entry_rejected(self):
        """Removal must not delete a similar ingredient."""
        edit = ModificationEdit(
            target="ingredients",
            operation="remove",
            find="1 cup brown sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit,
            self.recipe.ingredients,
        )

        self.assertEqual(
            result,
            self.recipe.ingredients,
        )
        self.assertEqual(len(changes), 0)

    def test_removed_water_conflicts_with_instructions(self):
        """
        Removing hot water must be rejected if the recipe
        still instructs the cook to use hot water.
        """
        recipe = self.recipe.model_copy(deep=True)

        recipe.instructions = [
            "Dissolve baking soda in hot water.",
            "Bake for 10 minutes.",
        ]

        modification = self.make_modification([
            ModificationEdit(
                target="ingredients",
                operation="remove",
                find="2 teaspoons hot water",
            )
        ])

        modified_recipe, changes = (
            self.modifier.apply_modification(
                recipe,
                modification,
            )
        )

        # Ingredient must remain because instructions depend on it.
        self.assertIn(
            "2 teaspoons hot water",
            modified_recipe.ingredients,
        )

        # Rejected removal must not create a change record.
        self.assertEqual(len(changes), 0)

        # Dependent instruction must remain intact.
        self.assertIn(
            "Dissolve baking soda in hot water.",
            modified_recipe.instructions,
        )

    def test_removed_walnuts_conflicts_with_instructions(self):
        """
        Removing chopped walnuts must be rejected when
        instructions still reference walnuts.

        Valid unrelated edits must still be applied.
        """
        recipe = self.recipe.model_copy(deep=True)

        recipe.ingredients.extend([
            "0.5 teaspoon salt",
            "1 cup chopped walnuts",
        ])

        recipe.instructions = [
            "Add salt.",
            "Stir in flour, chocolate chips, and walnuts.",
        ]

        original_ingredients = recipe.ingredients.copy()
        original_instructions = recipe.instructions.copy()

        modification = ModificationObject(
            modification_type="ingredient_substitution",
            reasoning="Increase salt and omit walnuts.",
            edits=[
                ModificationEdit(
                    target="ingredients",
                    operation="replace",
                    find="0.5 teaspoon salt",
                    replace="1 teaspoon salt",
                ),
                ModificationEdit(
                    target="ingredients",
                    operation="remove",
                    find="1 cup chopped walnuts",
                ),
            ],
        )

        modified_recipe, changes = (
            self.modifier.apply_modification(
                recipe,
                modification,
            )
        )

        # Valid salt adjustment must be applied.
        self.assertIn(
            "1 teaspoon salt",
            modified_recipe.ingredients,
        )

        self.assertNotIn(
            "0.5 teaspoon salt",
            modified_recipe.ingredients,
        )

        # Walnut removal must be rejected because
        # instructions still reference walnuts.
        self.assertIn(
            "1 cup chopped walnuts",
            modified_recipe.ingredients,
        )

        # Only the successful salt replacement counts.
        self.assertEqual(len(changes), 1)

        self.assertEqual(
            changes[0].operation,
            "replace",
        )

        self.assertEqual(
            changes[0].from_text,
            "0.5 teaspoon salt",
        )

        self.assertEqual(
            changes[0].to_text,
            "1 teaspoon salt",
        )

        # Instructions must remain unchanged.
        self.assertEqual(
            modified_recipe.instructions,
            original_instructions,
        )

        # Original recipe must not be mutated.
        self.assertEqual(
            recipe.ingredients,
            original_ingredients,
        )

        self.assertEqual(
            recipe.instructions,
            original_instructions,
        )

    def test_attribution_excludes_rejected_edits(self):
        """
        Attribution must describe only successfully applied changes,
        not rejected suggestions such as cream of tartar.
        """
        generator = EnhancedRecipeGenerator()

        changes = [
            ChangeRecord(
                type="ingredient",
                from_text="1 cup white sugar",
                to_text="0.5 cup white sugar",
                operation="replace",
            ),
            ChangeRecord(
                type="ingredient",
                from_text="1 cup packed brown sugar",
                to_text="1.5 cups packed brown sugar",
                operation="replace",
            ),
        ]

        explanation = generator.describe_applied_changes(changes)

        # Successfully applied changes must be included.
        self.assertIn(
            "0.5 cup white sugar",
            explanation,
        )

        self.assertIn(
            "1.5 cups packed brown sugar",
            explanation,
        )

        # Rejected edits must not appear in attribution.
        self.assertNotIn(
            "cream of tartar",
            explanation,
        )

        self.assertNotIn(
            "hot water",
            explanation,
        )


if __name__ == "__main__":
    unittest.main()

