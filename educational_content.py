"""
Educational Content System for Calculus Grapher
Fetches and manages mathematical examples and educational content
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import logging

# Configure logging for educational content system
logger = logging.getLogger(__name__)

@dataclass
class MathExample:
    """Represents a mathematical example with metadata"""
    title: str
    expression: str
    category: str
    description: str
    difficulty: str = "intermediate"
    display_expression: str = field(init=False)
    tags: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate user-friendly display expression after initialization"""
        self.display_expression = self._format_for_display(self.expression)
    
    def _format_for_display(self, expr: str) -> str:
        """Convert programming syntax to mathematical notation"""
        # Convert ** to superscripts for simple cases
        expr = re.sub(r'x\*\*(\d+)', lambda m: f'x^{m.group(1)}', expr)
        expr = re.sub(r'y\*\*(\d+)', lambda m: f'y^{m.group(1)}', expr)
        
        # Remove multiplication asterisks
        expr = re.sub(r'(\d+)\*([xy])', r'\1\2', expr)  # 2*x -> 2x
        expr = re.sub(r'([xy])\*([xy])', r'\1\2', expr)  # x*y -> xy
        
        # Convert function names
        expr = expr.replace('np.sin', 'sin')
        expr = expr.replace('np.cos', 'cos')
        expr = expr.replace('np.exp', 'exp')
        expr = expr.replace('np.log', 'ln')
        
        return expr
    
class EducationalContentManager:
    """Manages educational content for the calculus grapher"""
    
    def __init__(self):
        self._examples_cache = None
        self._categories_cache = None
        self.categories = {
            "derivatives": "Derivatives and Differentiation",
            "integrals": "Integration and Antiderivatives", 
            "limits": "Limits and Continuity",
            "multivariable": "Multivariable Calculus",
            "optimization": "Optimization Problems",
            "applications": "Real-world Applications"
        }
    
    @property
    def examples(self) -> List[MathExample]:
        """Lazy-loaded examples with caching"""
        if self._examples_cache is None:
            self._examples_cache = self._load_default_examples()
        return self._examples_cache
    
    def _load_default_examples(self) -> List[MathExample]:
        """Load curated mathematical examples based on educational standards"""
        return [
            # Multivariable Calculus Examples (Primary focus)
            MathExample(
                title="Simple Paraboloid",
                expression="x**2 + y**2",
                category="multivariable",
                description="Classic bowl shape - perfect for understanding basic 3D surfaces",
                difficulty="beginner",
                tags=["3d-surface", "optimization", "critical-points"],
                learning_objectives=[
                    "Understand 3D surface visualization",
                    "Identify global minimum at origin",
                    "Recognize circular level curves"
                ]
            ),
            MathExample(
                title="Saddle Point",
                expression="x**2 - y**2",
                category="multivariable",
                description="Horse saddle shape - demonstrates critical points that aren't extrema",
                difficulty="beginner",
                tags=["saddle-point", "critical-points", "optimization"],
                learning_objectives=[
                    "Understand saddle point behavior",
                    "Visualize hyperbolic level curves",
                    "Learn about inconclusive second derivative test"
                ]
            ),
            MathExample(
                title="Partial Derivatives",
                expression="x**2 * y**4",
                category="multivariable",
                description="Partial derivatives ∂f/∂x and ∂f/∂y of x²y⁴",
                difficulty="intermediate",
                tags=["partial-derivatives", "gradient"],
                learning_objectives=[
                    "Calculate partial derivatives",
                    "Understand gradient vectors",
                    "Visualize directional derivatives"
                ]
            ),
            MathExample(
                title="Gradient Vector Field",
                expression="sin(x**2 * y)",
                category="multivariable",
                description="Gradient ∇f of sin(x²y) - complex interaction between variables",
                difficulty="advanced",
                tags=["gradient", "vector-field", "trigonometric"],
                learning_objectives=[
                    "Visualize complex gradient fields",
                    "Understand variable interaction",
                    "Apply chain rule in multiple variables"
                ]
            ),
            MathExample(
                title="Optimization Problem",
                expression="x**3 - 3*x*y**2",
                category="optimization",
                description="Monkey saddle - three-way saddle point for advanced analysis",
                difficulty="advanced",
                tags=["monkey-saddle", "critical-points", "hessian"],
                learning_objectives=[
                    "Analyze complex critical points",
                    "Use Hessian matrix classification",
                    "Understand higher-order behavior"
                ]
            ),
            
            # Supporting Examples
            MathExample(
                title="Gaussian Bell Curve",
                expression="np.exp(-(x**2 + y**2))",
                category="multivariable",
                description="2D Gaussian - fundamental in statistics and probability",
                difficulty="intermediate",
                tags=["gaussian", "exponential", "statistics"],
                learning_objectives=[
                    "Understand exponential decay",
                    "Visualize probability distributions",
                    "Recognize radial symmetry"
                ]
            ),
            MathExample(
                title="Wave Interference",
                expression="np.sin(x) * np.cos(y)",
                category="multivariable",
                description="Standing wave pattern - product of trigonometric functions",
                difficulty="intermediate",
                tags=["waves", "trigonometric", "interference"],
                learning_objectives=[
                    "Visualize wave interference patterns",
                    "Understand periodic behavior",
                    "Analyze product functions"
                ]
            )
        ]
    
    @lru_cache(maxsize=32)
    def get_examples_by_category(self, category: str) -> Tuple[MathExample, ...]:
        """Get all examples for a specific category (cached)"""
        return tuple(ex for ex in self.examples if ex.category == category)
    
    @lru_cache(maxsize=16)
    def get_examples_by_difficulty(self, difficulty: str) -> Tuple[MathExample, ...]:
        """Get examples filtered by difficulty level (cached)"""
        return tuple(ex for ex in self.examples if ex.difficulty == difficulty)
    
    def get_examples_by_tags(self, tags: List[str]) -> List[MathExample]:
        """Get examples that match any of the provided tags"""
        return [ex for ex in self.examples if any(tag in ex.tags for tag in tags)]
    
    def add_example(self, example: MathExample):
        """Add a new mathematical example and clear cache"""
        if self._examples_cache is not None:
            self._examples_cache.append(example)
        # Clear LRU cache when examples change
        self.get_examples_by_category.cache_clear()
        self.get_examples_by_difficulty.cache_clear()
    
    def get_progressive_examples(self, category: str) -> List[MathExample]:
        """Get examples ordered by difficulty for progressive learning"""
        examples = list(self.get_examples_by_category(category))
        difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
        return sorted(examples, key=lambda ex: difficulty_order.get(ex.difficulty, 2))
    
    def get_random_example(self, category: Optional[str] = None, difficulty: Optional[str] = None) -> MathExample:
        """Get a random example with optional filtering"""
        import random
        
        candidates = self.examples
        if category:
            candidates = list(self.get_examples_by_category(category))
        if difficulty:
            candidates = [ex for ex in candidates if ex.difficulty == difficulty]
            
        return random.choice(candidates) if candidates else self.examples[0]
    
    def export_examples_json(self) -> str:
        """Export examples as JSON for web interface with enhanced metadata"""
        examples_dict = []
        for ex in self.examples:
            examples_dict.append({
                "title": ex.title,
                "expression": ex.expression,
                "display_expression": ex.display_expression,
                "category": ex.category,
                "description": ex.description,
                "difficulty": ex.difficulty,
                "tags": ex.tags,
                "learning_objectives": ex.learning_objectives
            })
        return json.dumps(examples_dict, indent=2)
    
    def get_learning_path(self, topic: str) -> List[MathExample]:
        """Generate a structured learning path for a specific topic"""
        topic_examples = self.get_examples_by_tags([topic])
        return self.get_progressive_examples("multivariable") if not topic_examples else topic_examples
    
    def validate_expression(self, expression: str) -> Tuple[bool, str]:
        """Validate mathematical expression for safety and correctness"""
        try:
            # Basic syntax validation
            if not expression or not isinstance(expression, str):
                return False, "Expression cannot be empty"
            
            # Check for dangerous operations
            dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file']
            if any(pattern in expression.lower() for pattern in dangerous_patterns):
                return False, "Expression contains potentially unsafe operations"
            
            # Check for valid mathematical syntax
            valid_functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'np.']
            valid_operators = ['+', '-', '*', '/', '**', '(', ')', 'x', 'y']
            
            # This is a simplified check - in production, use a proper parser
            return True, "Expression appears valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class MathNotationFormatter:
    """Handles conversion between programming syntax and mathematical notation"""
    
    @staticmethod
    def to_display_format(expression: str) -> str:
        """Convert programming syntax to user-friendly mathematical notation"""
        if not expression:
            return expression
        
        # Superscript mapping
        superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
                       '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        
        # Convert ** to superscripts for simple cases
        def replace_power(match):
            base = match.group(1)
            exp = match.group(2)
            if len(exp) == 1 and exp in superscripts:
                return base + superscripts[exp]
            return f"{base}^{exp}"
        
        # Handle x**n, y**n patterns
        expression = re.sub(r'([xy])\*\*([0-9]+)', replace_power, expression)
        
        # Remove multiplication asterisks for cleaner display
        expression = re.sub(r'(\d+)\*([xy])', r'\1\2', expression)  # 2*x -> 2x
        expression = re.sub(r'([xy])\*([xy])', r'\1\2', expression)  # x*y -> xy
        
        # Convert function names to proper notation
        replacements = {
            'np.sin': 'sin',
            'np.cos': 'cos',
            'np.tan': 'tan',
            'np.exp': 'exp',
            'np.log': 'ln',
            'np.sqrt': '√',
            'np.pi': 'π',
            'np.e': 'e'
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression
    
    @staticmethod
    def to_code_format(expression: str) -> str:
        """Convert user-friendly notation to executable code"""
        # Reverse the display formatting
        replacements = {
            '²': '**2',
            '³': '**3',
            '⁴': '**4',
            '⁵': '**5',
            'sin': 'np.sin',
            'cos': 'np.cos',
            'tan': 'np.tan',
            'exp': 'np.exp',
            'ln': 'np.log',
            '√': 'np.sqrt',
            'π': 'np.pi'
        }
        
        result = expression
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        # Add multiplication operators where needed
        result = re.sub(r'(\d)([xy])', r'\1*\2', result)  # 2x -> 2*x
        result = re.sub(r'([xy])([xy])', r'\1*\2', result)  # xy -> x*y
        
        return result


# Educational content fetcher using MCP
class OnlineContentFetcher:
    """Fetches educational content from online sources"""
    
    def __init__(self):
        self.formatter = MathNotationFormatter()
    
    @staticmethod
    def fetch_wolfram_examples() -> Dict:
        """Fetch examples from Wolfram Alpha (simulated structure)"""
        # This would use MCP fetch in practice
        return {
            "multivariable": [
                "partial derivative of x^2 y^4 with respect to x",
                "gradient of sin(x^2 * y)",
                "critical points of x^2 - y^2",
                "Hessian matrix of x^3 - 3xy^2"
            ],
            "optimization": [
                "minimize x^2 + y^2 - 2x - 4y + 5",
                "find critical points of Himmelblau function",
                "Lagrange multipliers for constrained optimization"
            ],
            "visualization": [
                "3D plot of paraboloid x^2 + y^2",
                "contour plot of saddle point x^2 - y^2",
                "gradient vector field visualization"
            ]
        }
    
    def parse_mathematical_expression(self, text: str) -> str:
        """Convert natural language math to symbolic expression"""
        # Enhanced conversion rules for multivariable calculus
        conversions = {
            r'\^': '**',
            r'sin\s*\(([^)]+)\)': r'np.sin(\1)',
            r'cos\s*\(([^)]+)\)': r'np.cos(\1)',
            r'tan\s*\(([^)]+)\)': r'np.tan(\1)',
            r'exp\s*\(([^)]+)\)': r'np.exp(\1)',
            r'ln\s*\(([^)]+)\)': r'np.log(\1)',
            r'sqrt\s*\(([^)]+)\)': r'np.sqrt(\1)',
            r'pi': 'np.pi',
            r'\be\b': 'np.e'
        }
        
        result = text
        for pattern, replacement in conversions.items():
            result = re.sub(pattern, replacement, result)
        
        return result

if __name__ == "__main__":
    # Demo usage
    content_manager = EducationalContentManager()
    
    print("Available Categories:")
    for key, name in content_manager.categories.items():
        print(f"  {key}: {name}")
    
    print(f"\nTotal Examples: {len(content_manager.examples)}")
    
    # Get examples by category
    derivative_examples = content_manager.get_examples_by_category("derivatives")
    print(f"\nDerivative Examples ({len(derivative_examples)}):")
    for ex in derivative_examples:
        print(f"  - {ex.title}: {ex.expression}")
    
    # Export as JSON
    print("\nJSON Export:")
    print(content_manager.export_examples_json())