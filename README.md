# Recipe Enhancement Platform

Automatically enhances recipes by analyzing and applying community-tested modifications from AllRecipes.com. Uses LLM processing to extract meaningful recipe tweaks and apply them with full citation tracking.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

- Python 3.13+
- `uv` package manager

## Setup

```bash
# Install dependencies
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

### 1. Scrape Recipes (Optional - data already provided)

```bash
uv run python src/scraper_v2.py
```

### 2. Run Recipe Enhancement Pipeline

```bash
cd src

# Test single recipe (chocolate chip cookies)
uv run python test_pipeline.py single

# Process all recipes
uv run python test_pipeline.py all
```

## Output

### Enhanced Recipes

Enhanced recipes are saved in `src/data/enhanced/`:

- `enhanced_[recipe_id]_[recipe-name].json` - Individual enhanced recipes with modifications applied
- `pipeline_summary_report.json` - Summary of all processing results

### Data Structure

Original scraped recipes in `data/` directory contain reviews with `has_modification: true` flags. Enhanced recipes include:

```json
{
  "recipe_id": "10813_enhanced",
  "title": "Best Chocolate Chip Cookies (Community Enhanced)",
  "ingredients": ["1 cup butter", "1 additional egg yolk", ...],
  "modifications_applied": [
    {
      "source_review": {
        "text": "I added an extra egg yolk for chewier texture",
        "rating": 5
      },
      "modification_type": "addition",
      "reasoning": "Improves texture and chewiness",
      "changes_made": [...]
    }
  ],
  "enhancement_summary": {
    "total_changes": 1,
    "change_types": ["addition"],
    "expected_impact": "Chewier texture and improved consistency"
  }
}
```

## How It Works

The LLM Analysis Pipeline enhances recipes using community feedback through three stages:

### 1. Tweak Extraction

- Selects one review marked as containing a recipe modification.
- Uses GPT-4o-mini to extract structured modification proposals.
- Identifies ingredient substitutions, quantity adjustments, additions, removals, and technique changes.
- Produces structured edits containing the target, operation, and relevant recipe text.

### 2. Recipe Modification

- Applies validated edits to the original recipe.
- Supports ingredient and instruction replacements, additions, and removals.
- Requires exact, unambiguous target matches before modifying recipe content.
- Rejects unsafe removals when ingredients are still referenced in cooking instructions.
- Skips invalid edits rather than applying uncertain modifications.
- Preserves the original recipe and records only successfully applied changes.

### 3. Enhanced Recipe Generation

- Generates an enhanced recipe containing the modified ingredients and instructions.
- Links applied modifications to their original community reviews.
- Records each successful change, including its operation and before/after text.
- Preserves the extracted reasoning when all proposed edits are successfully applied.
- Uses factual change descriptions when only some proposed edits succeed, avoiding unsupported attribution.
- Produces an enhancement summary containing change counts, modification types, and expected impact or applied-change descriptions.

### Batch Processing

The pipeline can process all supplied recipe files and generate a summary report.

Recipes without reviews containing modifications cannot be enhanced and are reported as unsuccessful processing attempts. The pipeline continues processing the remaining recipes.

### Validation

The implementation includes offline regression tests covering exact-match editing, multiple edits, duplicate JSON keys, original recipe preservation, rejected removals, no-op changes, and accurate attribution.

During batch validation, the pipeline processed six supplied recipes, successfully enhanced four, and generated `pipeline_summary_report.json`.
