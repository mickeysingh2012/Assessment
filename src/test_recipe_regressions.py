
"""
Offline regression tests for the Casper Studios recipe pipeline.

Run:
    uv run python -m unittest -v test_recipe_regressions

These tests make no OpenAI API calls.
"""

import json
import unittest

from llm_pipeline.models import (
    ModificationEdit,
    ModificationObject,
    Recipe,
)
from llm_pipeline.recipe_modifier import RecipeModifier
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
        """Duplicate edits arrays must not silently overwrite data."""
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
        """A single edits array must retain all its entries."""
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
            edit, self.recipe.ingredients
        )

        self.assertIn(
            "0.5 cup packed brown sugar", result
        )
        self.assertEqual(len(changes), 1)

    def test_no_op_replacement_not_counted(self):
        """Replacing text with itself must not count as a change."""
        edit = ModificationEdit(
            target="ingredients",
            operation="replace",
            find="1 cup white sugar",
            replace="1 cup white sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit, self.recipe.ingredients
        )

        self.assertEqual(
            result, self.recipe.ingredients
        )
        self.assertEqual(len(changes), 0)

    def test_fuzzy_only_replacement_rejected(self):
        """A similar but nonmatching ingredient must not be changed."""
        edit = ModificationEdit(
            target="ingredients",
            operation="replace",
            find="1 cup brown sugar",
            replace="0.5 cup brown sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit, self.recipe.ingredients
        )

        self.assertEqual(
            result, self.recipe.ingredients
        )
        self.assertEqual(len(changes), 0)

    def test_multiple_edits_applied(self):
        """All valid edits should be applied to one recipe."""
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
            self.recipe, modification
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
        """Applying edits must preserve the original recipe."""
        original_ingredients = (
            self.recipe.ingredients.copy()
        )

        modification = self.make_modification([
            ModificationEdit(
                target="ingredients",
                operation="replace",
                find="3 cups all-purpose flour",
                replace="2.5 cups all-purpose flour",
            )
        ])

        modified_recipe, _ = (
            self.modifier.apply_modification(
                self.recipe, modification
            )
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
        """A removal must not delete a similar but different entry."""
        edit = ModificationEdit(
            target="ingredients",
            operation="remove",
            find="1 cup brown sugar",
        )

        result, changes = self.modifier.apply_edit(
            edit, self.recipe.ingredients
        )

        self.assertEqual(
            result, self.recipe.ingredients
        )
        self.assertEqual(len(changes), 0)


if __name__ == "__main__":
    unittest.main()
