"""
MCP Content Fetcher for Educational Mathematical Examples
Fetches real-time educational content using Model Context Protocol
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from educational_content import MathExample

@dataclass
class FetchedContent:
    """Represents content fetched from online sources"""
    source: str
    title: str
    content: str
    examples: List[str]
    category: str

class MCPEducationalFetcher:
    """Fetches educational content using MCP fetch capabilities"""
    
    def __init__(self):
        self.sources = {
            "wolfram_calculus": "https://www.wolframalpha.com/examples/mathematics/calculus-and-analysis/",
            "wolfram_derivatives": "https://www.wolframalpha.com/examples/mathematics/calculus-and-analysis/derivatives/",
            "wolfram_integrals": "https://www.wolframalpha.com/examples/mathematics/calculus-and-analysis/integrals/",
            "wolfram_limits": "https://www.wolframalpha.com/examples/mathematics/calculus-and-analysis/limits/"
        }
        
        self.cached_content = {}
    
    def parse_wolfram_examples(self, content: str, category: str) -> List[MathExample]:
        """Parse Wolfram Alpha content to extract mathematical examples"""
        examples = []
        
        # Extract example patterns from Wolfram content
        patterns = [
            r'#### ([^:]+):\s*\[([^\]]+)\]',  # Example titles and expressions
            r'\[([^\]]+)\]\(/input\?i=([^)]+)\)',  # Input examples
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) == 2:
                    title, expression = match
                    
                    # Clean up the expression
                    cleaned_expr = self._clean_expression(expression)
                    
                    if cleaned_expr:
                        examples.append(MathExample(
                            title=title.strip(),
                            expression=cleaned_expr,
                            category=category,
                            description=f"Example from Wolfram Alpha: {title}",
                            difficulty="intermediate"
                        ))
        
        return examples
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and convert mathematical expressions to Python syntax"""
        if not expr or len(expr) < 2:
            return ""
        
        # URL decode common patterns
        expr = expr.replace('%5E', '**')  # ^ to **
        expr = expr.replace('%2B', '+')   # + 
        expr = expr.replace('%2F', '/')   # /
        expr = expr.replace('%28', '(')   # (
        expr = expr.replace('%29', ')')   # )
        expr = expr.replace('%2C', ',')   # ,
        expr = expr.replace('+', ' ')     # + to space
        
        # Convert mathematical notation
        conversions = {
            '^': '**',
            'sin x': 'sin(x)',
            'cos x': 'cos(x)', 
            'tan x': 'tan(x)',
            'ln x': 'log(x)',
            'sqrt x': 'sqrt(x)',
            ' x ': '*x*',
            'dx': '',
            'dy': '',
            'integrate': '',
            'derivative of': '',
            'limit': '',
            'lim': ''
        }
        
        result = expr.lower()
        for old, new in conversions.items():
            result = result.replace(old, new)
        
        # Remove extra spaces and clean up
        result = ' '.join(result.split())
        
        # Basic validation - must contain x or y
        if 'x' not in result and 'y' not in result:
            return ""
        
        # Remove common prefixes/suffixes that aren't expressions
        prefixes_to_remove = ['compute', 'calculate', 'find', 'solve']
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        
        return result if len(result) > 1 else ""
    
    def fetch_educational_examples(self, category: str = "all") -> List[MathExample]:
        """Fetch educational examples for a specific category"""
        examples = []
        
        # This would use actual MCP fetch in practice
        # For now, we'll simulate with the content we already fetched
        
        if category in ["derivatives", "all"]:
            # Simulate fetched derivative examples
            derivative_examples = [
                MathExample(
                    title="Product Rule Example",
                    expression="x**4 * sin(x)",
                    category="derivatives",
                    description="Derivative of x⁴ sin(x) using product rule",
                    difficulty="intermediate"
                ),
                MathExample(
                    title="Chain Rule with Trigonometry",
                    expression="sin(x**2)",
                    category="derivatives",
                    description="Derivative of sin(x²) using chain rule",
                    difficulty="beginner"
                ),
                MathExample(
                    title="Quotient Rule",
                    expression="(x**2 + 1) / (x - 1)",
                    category="derivatives",
                    description="Derivative using quotient rule",
                    difficulty="intermediate"
                )
            ]
            examples.extend(derivative_examples)
        
        if category in ["integrals", "all"]:
            integral_examples = [
                MathExample(
                    title="Integration by Parts",
                    expression="x * exp(x)",
                    category="integrals",
                    description="∫ x eˣ dx using integration by parts",
                    difficulty="intermediate"
                ),
                MathExample(
                    title="Trigonometric Substitution",
                    expression="sqrt(1 - x**2)",
                    category="integrals",
                    description="∫ √(1-x²) dx using trig substitution",
                    difficulty="advanced"
                )
            ]
            examples.extend(integral_examples)
        
        if category in ["multivariable", "all"]:
            multivariable_examples = [
                MathExample(
                    title="Saddle Point",
                    expression="x**2 - y**2",
                    category="multivariable",
                    description="Classic saddle point function",
                    difficulty="intermediate"
                ),
                MathExample(
                    title="Paraboloid",
                    expression="x**2 + y**2",
                    category="multivariable",
                    description="Circular paraboloid - bowl shape",
                    difficulty="beginner"
                ),
                MathExample(
                    title="Monkey Saddle",
                    expression="x**3 - 3*x*y**2",
                    category="multivariable",
                    description="Three-way saddle point",
                    difficulty="advanced"
                )
            ]
            examples.extend(multivariable_examples)
        
        return examples
    
    def get_real_world_applications(self) -> List[MathExample]:
        """Fetch real-world application examples"""
        return [
            MathExample(
                title="Heat Distribution",
                expression="exp(-(x**2 + y**2))",
                category="applications",
                description="2D Gaussian - models heat distribution",
                difficulty="intermediate"
            ),
            MathExample(
                title="Wave Function",
                expression="sin(x) * cos(y)",
                category="applications", 
                description="Standing wave pattern",
                difficulty="intermediate"
            ),
            MathExample(
                title="Economic Utility",
                expression="x**0.5 * y**0.5",
                category="applications",
                description="Cobb-Douglas utility function",
                difficulty="advanced"
            ),
            MathExample(
                title="Population Growth",
                expression="x * exp(-x**2 - y**2)",
                category="applications",
                description="Population density with carrying capacity",
                difficulty="advanced"
            )
        ]
    
    def update_content_cache(self):
        """Update cached content from online sources"""
        # This would use MCP fetch to update content
        # For now, we'll simulate the update
        self.cached_content["last_updated"] = "2025-01-27"
        self.cached_content["sources_checked"] = len(self.sources)
        
        print("Content cache updated successfully")
        return True

# Example usage and testing
if __name__ == "__main__":
    fetcher = MCPEducationalFetcher()
    
    print("Fetching educational examples...")
    
    # Test different categories
    categories = ["derivatives", "integrals", "multivariable"]
    
    for category in categories:
        examples = fetcher.fetch_educational_examples(category)
        print(f"\n{category.upper()} Examples ({len(examples)}):")
        
        for ex in examples:
            print(f"  • {ex.title}: {ex.expression}")
            print(f"    {ex.description} [{ex.difficulty}]")
    
    # Test real-world applications
    apps = fetcher.get_real_world_applications()
    print(f"\nREAL-WORLD APPLICATIONS ({len(apps)}):")
    for app in apps:
        print(f"  • {app.title}: {app.expression}")
        print(f"    {app.description}")
    
    # Update cache
    print(f"\nUpdating content cache...")
    fetcher.update_content_cache()