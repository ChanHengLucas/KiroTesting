"""
Accessibility and UI/UX enhancement utilities for the calculus grapher
Focuses on educational user experience and inclusive design
"""

from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum

class DifficultyLevel(Enum):
    """Enumeration for difficulty levels with educational context"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class AccessibilitySettings:
    """User accessibility preferences"""
    high_contrast: bool = False
    large_text: bool = False
    reduced_motion: bool = False
    screen_reader_mode: bool = False
    color_blind_friendly: bool = False
    simplified_interface: bool = False

class MathNotationAccessibility:
    """Handle mathematical notation for accessibility"""
    
    @staticmethod
    def to_screen_reader_text(expression: str) -> str:
        """Convert mathematical expression to screen reader friendly text"""
        # Replace mathematical symbols with spoken equivalents
        replacements = {
            '²': ' squared',
            '³': ' cubed',
            '⁴': ' to the fourth power',
            '⁵': ' to the fifth power',
            '∂': 'partial derivative of',
            '∇': 'gradient of',
            '∫': 'integral of',
            'π': 'pi',
            '∞': 'infinity',
            '≤': 'less than or equal to',
            '≥': 'greater than or equal to',
            '≠': 'not equal to',
            '±': 'plus or minus',
            '√': 'square root of',
            'sin': 'sine of',
            'cos': 'cosine of',
            'tan': 'tangent of',
            'exp': 'exponential of',
            'ln': 'natural log of',
            'log': 'logarithm of'
        }
        
        result = expression
        for symbol, spoken in replacements.items():
            result = result.replace(symbol, spoken)
        
        # Handle fractions
        result = re.sub(r'(\w+)/(\w+)', r'\1 divided by \2', result)
        
        # Handle powers
        result = re.sub(r'(\w+)\^(\d+)', r'\1 to the power of \2', result)
        
        return result
    
    @staticmethod
    def get_expression_complexity(expression: str) -> DifficultyLevel:
        """Analyze expression complexity for appropriate difficulty classification"""
        complexity_indicators = {
            'beginner': ['x**2', 'y**2', '+', '-', 'x*y'],
            'intermediate': ['sin', 'cos', 'exp', '**3', '**4', 'sqrt'],
            'advanced': ['tan', 'log', '**5', 'sinh', 'cosh', 'complex']
        }
        
        expression_lower = expression.lower()
        
        # Count indicators for each level
        scores = {}
        for level, indicators in complexity_indicators.items():
            scores[level] = sum(1 for indicator in indicators if indicator in expression_lower)
        
        # Determine difficulty based on highest score
        max_level = max(scores, key=scores.get)
        if scores[max_level] == 0:
            return DifficultyLevel.BEGINNER
        
        return DifficultyLevel(max_level)

class ColorSchemeManager:
    """Manage color schemes for accessibility and visual appeal"""
    
    SCHEMES = {
        'default': {
            'primary': '#4facfe',
            'secondary': '#00f2fe',
            'surface': '#ffffff',
            'background': '#f8f9fa',
            'text': '#212529',
            'accent': '#28a745'
        },
        'high_contrast': {
            'primary': '#000000',
            'secondary': '#ffffff',
            'surface': '#ffffff',
            'background': '#000000',
            'text': '#ffffff',
            'accent': '#ffff00'
        },
        'color_blind_friendly': {
            'primary': '#1f77b4',  # Blue
            'secondary': '#ff7f0e',  # Orange
            'surface': '#ffffff',
            'background': '#f8f9fa',
            'text': '#212529',
            'accent': '#2ca02c'  # Green
        },
        'dark_mode': {
            'primary': '#bb86fc',
            'secondary': '#03dac6',
            'surface': '#1e1e1e',
            'background': '#121212',
            'text': '#ffffff',
            'accent': '#cf6679'
        }
    }
    
    @classmethod
    def get_scheme(cls, scheme_name: str, settings: AccessibilitySettings) -> Dict[str, str]:
        """Get color scheme based on accessibility settings"""
        if settings.high_contrast:
            return cls.SCHEMES['high_contrast']
        elif settings.color_blind_friendly:
            return cls.SCHEMES['color_blind_friendly']
        else:
            return cls.SCHEMES.get(scheme_name, cls.SCHEMES['default'])
    
    @classmethod
    def get_plotly_colorscales(cls, settings: AccessibilitySettings) -> List[str]:
        """Get appropriate Plotly colorscales for accessibility"""
        if settings.color_blind_friendly:
            return ['Viridis', 'Plasma', 'Cividis', 'Blues', 'Oranges']
        elif settings.high_contrast:
            return ['Greys', 'Blues', 'Reds']
        else:
            return ['Viridis', 'Plasma', 'Rainbow', 'Hot', 'Blues', 'Greens']

class EducationalGuidance:
    """Provide educational guidance and hints for users"""
    
    CONCEPT_EXPLANATIONS = {
        'partial_derivatives': {
            'title': 'Partial Derivatives',
            'explanation': 'A partial derivative shows how a function changes when you vary one variable while keeping others constant.',
            'visual_cues': 'Look for the slope of cross-sections parallel to the axes.',
            'common_mistakes': 'Remember to treat other variables as constants when differentiating.'
        },
        'critical_points': {
            'title': 'Critical Points',
            'explanation': 'Critical points occur where all partial derivatives equal zero, indicating potential maxima, minima, or saddle points.',
            'visual_cues': 'Look for flat spots on the surface where the tangent plane is horizontal.',
            'common_mistakes': 'Not all critical points are extrema - some are saddle points.'
        },
        'gradient': {
            'title': 'Gradient Vector',
            'explanation': 'The gradient points in the direction of steepest increase and its magnitude shows the rate of change.',
            'visual_cues': 'Gradient vectors are perpendicular to level curves and point uphill.',
            'common_mistakes': 'The gradient is a vector, not a scalar - it has both direction and magnitude.'
        },
        'contour_lines': {
            'title': 'Contour Lines',
            'explanation': 'Contour lines connect points of equal function value, like elevation lines on a topographic map.',
            'visual_cues': 'Closely spaced contours indicate steep slopes; widely spaced indicate gentle slopes.',
            'common_mistakes': 'Contour lines never cross each other.'
        }
    }
    
    @classmethod
    def get_contextual_help(cls, current_function: str, visualization_type: str) -> Dict[str, str]:
        """Get contextual help based on current function and visualization"""
        help_content = {
            'title': 'Getting Started',
            'explanation': 'Explore the 3D surface by rotating, zooming, and hovering over points.',
            'tips': []
        }
        
        # Analyze function for specific guidance
        if 'x**2' in current_function and 'y**2' in current_function:
            if '+' in current_function:
                help_content.update(cls.CONCEPT_EXPLANATIONS['critical_points'])
                help_content['tips'].append('This paraboloid has a minimum at the origin.')
            elif '-' in current_function:
                help_content['title'] = 'Saddle Point Function'
                help_content['explanation'] = 'This creates a saddle shape with a critical point that is neither a maximum nor minimum.'
                help_content['tips'].append('Try the contour view to see the hyperbolic level curves.')
        
        # Add visualization-specific tips
        if visualization_type == 'contour':
            help_content['tips'].append('Enable gradient vectors to see the direction of steepest ascent.')
        elif visualization_type == 'surface':
            help_content['tips'].append('Use the mouse to rotate the 3D view and explore different angles.')
        elif visualization_type == 'gradient':
            help_content['tips'].append('Longer arrows indicate steeper slopes in that direction.')
        
        return help_content
    
    @classmethod
    def get_progressive_hints(cls, difficulty: DifficultyLevel) -> List[str]:
        """Get progressive hints based on difficulty level"""
        hints = {
            DifficultyLevel.BEGINNER: [
                "Start with simple functions like x² + y² to understand 3D surfaces",
                "Try rotating the 3D view to see the surface from different angles",
                "Enable contour lines to see level curves projected below the surface",
                "Hover over points to see exact coordinates and function values"
            ],
            DifficultyLevel.INTERMEDIATE: [
                "Compare different functions by switching between examples",
                "Enable critical points to see where the gradient equals zero",
                "Try the gradient vector field to understand directional derivatives",
                "Experiment with different domain ranges to see function behavior"
            ],
            DifficultyLevel.ADVANCED: [
                "Analyze the Hessian matrix at critical points for classification",
                "Study the relationship between contour spacing and gradient magnitude",
                "Explore optimization problems using the critical point finder",
                "Compare analytical and numerical derivative calculations"
            ]
        }
        
        return hints.get(difficulty, hints[DifficultyLevel.BEGINNER])

class UserInterfaceEnhancer:
    """Enhance user interface for better educational experience"""
    
    @staticmethod
    def generate_responsive_layout(screen_size: str = 'desktop') -> Dict[str, any]:
        """Generate responsive layout configuration"""
        layouts = {
            'mobile': {
                'control_panel_width': '100%',
                'visualization_width': '100%',
                'stack_vertically': True,
                'compact_controls': True,
                'touch_friendly': True
            },
            'tablet': {
                'control_panel_width': '35%',
                'visualization_width': '65%',
                'stack_vertically': False,
                'compact_controls': True,
                'touch_friendly': True
            },
            'desktop': {
                'control_panel_width': '25%',
                'visualization_width': '75%',
                'stack_vertically': False,
                'compact_controls': False,
                'touch_friendly': False
            }
        }
        
        return layouts.get(screen_size, layouts['desktop'])
    
    @staticmethod
    def get_keyboard_shortcuts() -> Dict[str, str]:
        """Define keyboard shortcuts for accessibility"""
        return {
            'Space': 'Toggle animation/rotation',
            'R': 'Reset view to default',
            'H': 'Show/hide help panel',
            'C': 'Toggle contour lines',
            'G': 'Toggle gradient vectors',
            'P': 'Toggle critical points',
            '1-9': 'Load example function',
            'Escape': 'Close current dialog',
            'Tab': 'Navigate between controls',
            'Enter': 'Activate focused control'
        }
    
    @staticmethod
    def generate_aria_labels(component_type: str, context: str = '') -> str:
        """Generate appropriate ARIA labels for screen readers"""
        labels = {
            'function_input': f'Mathematical function input field. {context}',
            'range_slider': f'Domain range slider for {context}',
            'visualization_tab': f'Switch to {context} visualization',
            'example_button': f'Load {context} example function',
            'control_panel': 'Function controls and settings panel',
            'main_plot': f'Interactive {context} plot of mathematical function',
            'help_button': 'Show contextual help and guidance',
            'reset_button': 'Reset all settings to default values'
        }
        
        return labels.get(component_type, f'{component_type} control')

class ProgressiveDisclosure:
    """Implement progressive disclosure for complex features"""
    
    @staticmethod
    def get_feature_levels() -> Dict[str, List[str]]:
        """Define feature levels for progressive disclosure"""
        return {
            'basic': [
                'function_input',
                'basic_examples',
                'surface_plot',
                'domain_controls'
            ],
            'intermediate': [
                'contour_plot',
                'gradient_vectors',
                'critical_points',
                'color_schemes',
                'resolution_control'
            ],
            'advanced': [
                'cross_sections',
                'analysis_tab',
                'custom_ranges',
                'performance_settings',
                'export_options'
            ]
        }
    
    @staticmethod
    def should_show_feature(feature: str, user_level: str, settings: AccessibilitySettings) -> bool:
        """Determine if a feature should be shown based on user level and settings"""
        levels = ProgressiveDisclosure.get_feature_levels()
        
        if settings.simplified_interface:
            return feature in levels['basic']
        
        # Show features up to and including user's level
        for level_name, features in levels.items():
            if feature in features:
                level_order = ['basic', 'intermediate', 'advanced']
                user_level_index = level_order.index(user_level) if user_level in level_order else 0
                feature_level_index = level_order.index(level_name)
                return feature_level_index <= user_level_index
        
        return True  # Show unknown features by default

# Global accessibility manager
class AccessibilityManager:
    """Central manager for accessibility features"""
    
    def __init__(self):
        self.settings = AccessibilitySettings()
        self.color_manager = ColorSchemeManager()
        self.guidance = EducationalGuidance()
        self.ui_enhancer = UserInterfaceEnhancer()
        self.notation = MathNotationAccessibility()
    
    def update_settings(self, **kwargs):
        """Update accessibility settings"""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
    
    def get_enhanced_ui_config(self, screen_size: str = 'desktop') -> Dict:
        """Get complete UI configuration with accessibility enhancements"""
        layout = self.ui_enhancer.generate_responsive_layout(screen_size)
        colors = self.color_manager.get_scheme('default', self.settings)
        shortcuts = self.ui_enhancer.get_keyboard_shortcuts()
        
        return {
            'layout': layout,
            'colors': colors,
            'keyboard_shortcuts': shortcuts,
            'accessibility_settings': self.settings,
            'reduced_motion': self.settings.reduced_motion,
            'high_contrast': self.settings.high_contrast
        }

# Global instance
accessibility_manager = AccessibilityManager()