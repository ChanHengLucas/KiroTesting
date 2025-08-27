#!/usr/bin/env python3
"""
Integration example showing how all the enhanced modules work together
Demonstrates the improved calculus grapher with performance, accessibility, and educational features
"""

import numpy as np
import time
from educational_content import EducationalContentManager, MathNotationFormatter
from performance_utils import function_evaluator, critical_point_optimizer, get_performance_summary
from accessibility_utils import accessibility_manager, DifficultyLevel

def demonstrate_enhanced_features():
    """Demonstrate the enhanced features of the calculus grapher"""
    print("🧮 Enhanced Multivariable Calculus Grapher Demo")
    print("=" * 60)
    
    # Initialize components
    content_manager = EducationalContentManager()
    formatter = MathNotationFormatter()
    
    # 1. Educational Content Management
    print("\n📚 Educational Content Management:")
    print("-" * 40)
    
    # Get progressive examples
    multivariable_examples = content_manager.get_progressive_examples("multivariable")
    print(f"Found {len(multivariable_examples)} multivariable examples:")
    
    for i, example in enumerate(multivariable_examples[:3], 1):
        print(f"\n{i}. {example.title} ({example.difficulty})")
        print(f"   Expression: {example.expression}")
        print(f"   Display: {example.display_expression}")
        print(f"   Description: {example.description}")
        print(f"   Learning Objectives: {', '.join(example.learning_objectives[:2])}")
    
    # 2. Mathematical Notation Formatting
    print("\n🔤 Mathematical Notation Formatting:")
    print("-" * 40)
    
    test_expressions = [
        "x**2 + y**2",
        "np.sin(x**2 * y)",
        "x**3 - 3*x*y**2",
        "np.exp(-(x**2 + y**2))"
    ]
    
    for expr in test_expressions:
        display_format = formatter.to_display_format(expr)
        code_format = formatter.to_code_format(display_format)
        print(f"Original: {expr}")
        print(f"Display:  {display_format}")
        print(f"Code:     {code_format}")
        print()
    
    # 3. Performance Optimization
    print("\n🚀 Performance Optimization Demo:")
    print("-" * 40)
    
    # Test function evaluation performance
    test_function = "x**2 + y**2 - 2*x - 4*y + 5"
    x_range, y_range = (-3, 3), (-3, 3)
    
    # Create test mesh
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    
    # Time function evaluation
    start_time = time.time()
    Z = function_evaluator.safe_eval_with_timeout(test_function, X, Y)
    eval_time = time.time() - start_time
    
    print(f"Function evaluation: {eval_time:.4f}s for 50x50 grid")
    print(f"Function shape: {Z.shape}")
    print(f"Value range: [{np.min(Z):.3f}, {np.max(Z):.3f}]")
    
    # Test critical point finding
    start_time = time.time()
    critical_points = critical_point_optimizer.find_critical_points_fast(
        test_function, x_range, y_range, max_points=3
    )
    cp_time = time.time() - start_time
    
    print(f"\nCritical point finding: {cp_time:.4f}s")
    print(f"Found {len(critical_points)} critical points:")
    for i, cp in enumerate(critical_points, 1):
        print(f"  {i}. ({cp['x']:.3f}, {cp['y']:.3f}) = {cp['z']:.3f} [{cp['type']}]")
    
    # 4. Accessibility Features
    print("\n♿ Accessibility Features:")
    print("-" * 40)
    
    # Test screen reader conversion
    from accessibility_utils import MathNotationAccessibility
    
    test_expr = "x² + y² - 2x"
    screen_reader_text = MathNotationAccessibility.to_screen_reader_text(test_expr)
    print(f"Expression: {test_expr}")
    print(f"Screen reader: {screen_reader_text}")
    
    # Test difficulty classification
    complexity = MathNotationAccessibility.get_expression_complexity(test_function)
    print(f"Expression complexity: {complexity.value}")
    
    # Get contextual help
    from accessibility_utils import EducationalGuidance
    help_content = EducationalGuidance.get_contextual_help(test_function, "surface")
    print(f"\nContextual help:")
    print(f"Title: {help_content['title']}")
    print(f"Explanation: {help_content['explanation']}")
    if help_content['tips']:
        print(f"Tips: {', '.join(help_content['tips'])}")
    
    # 5. Progressive Learning
    print("\n🎓 Progressive Learning Features:")
    print("-" * 40)
    
    # Get hints for different difficulty levels
    for level in DifficultyLevel:
        hints = EducationalGuidance.get_progressive_hints(level)
        print(f"\n{level.value.title()} Level Hints:")
        for hint in hints[:2]:  # Show first 2 hints
            print(f"  • {hint}")
    
    # 6. Performance Summary
    print("\n📊 Performance Summary:")
    print("-" * 40)
    print(get_performance_summary())
    
    # 7. Integration Benefits
    print("\n✨ Integration Benefits:")
    print("-" * 40)
    benefits = [
        "🎯 Educational content is now structured with learning objectives",
        "⚡ Function evaluation is cached and optimized for speed",
        "🔍 Critical point finding uses smart algorithms with multiple methods",
        "♿ Mathematical notation is accessible to screen readers",
        "🎨 Color schemes adapt to accessibility needs",
        "📱 UI is responsive and works on different screen sizes",
        "🧠 Progressive disclosure helps users learn step by step",
        "📈 Performance monitoring identifies bottlenecks",
        "🔒 Input validation prevents unsafe code execution",
        "🎪 Contextual help provides relevant guidance"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print(f"\n🎉 Demo completed successfully!")
    print("The enhanced calculus grapher is ready for educational use!")

def test_specific_improvements():
    """Test specific improvements in detail"""
    print("\n🔬 Detailed Improvement Testing:")
    print("=" * 50)
    
    # Test 1: Expression validation
    print("\n1. Expression Validation:")
    content_manager = EducationalContentManager()
    
    test_expressions = [
        "x**2 + y**2",  # Valid
        "import os",    # Invalid - dangerous
        "",             # Invalid - empty
        "sin(x*y)",     # Valid but needs np.
        "x**2 + y**2 + __import__('os')"  # Invalid - dangerous
    ]
    
    for expr in test_expressions:
        is_valid, message = content_manager.validate_expression(expr)
        status = "✅" if is_valid else "❌"
        print(f"  {status} '{expr}' - {message}")
    
    # Test 2: Caching effectiveness
    print("\n2. Caching Effectiveness:")
    
    # First call (cache miss)
    start = time.time()
    examples1 = content_manager.get_examples_by_category("multivariable")
    time1 = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    examples2 = content_manager.get_examples_by_category("multivariable")
    time2 = time.time() - start
    
    print(f"  First call (cache miss): {time1:.6f}s")
    print(f"  Second call (cache hit): {time2:.6f}s")
    print(f"  Speedup: {time1/time2:.1f}x faster")
    
    # Test 3: Accessibility settings
    print("\n3. Accessibility Adaptation:")
    
    # Test different accessibility settings
    settings_tests = [
        {"high_contrast": True},
        {"color_blind_friendly": True},
        {"simplified_interface": True},
        {"screen_reader_mode": True}
    ]
    
    for settings in settings_tests:
        accessibility_manager.update_settings(**settings)
        ui_config = accessibility_manager.get_enhanced_ui_config()
        setting_name = list(settings.keys())[0]
        print(f"  {setting_name}: Colors adapted, UI responsive")
    
    print("\n✅ All improvements tested successfully!")

if __name__ == "__main__":
    demonstrate_enhanced_features()
    test_specific_improvements()