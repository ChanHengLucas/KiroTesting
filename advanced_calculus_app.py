#!/usr/bin/env python3
"""
Advanced Multivariable Calculus Educational Tool
Comprehensive visualization and analysis platform
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy import integrate
import pandas as pd
import json
import re

# Initialize Dash app with custom CSS
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Advanced Multivariable Calculus Explorer"

# Enhanced function library with educational examples
FUNCTION_LIBRARY = {
    "Basic Functions": {
        "Paraboloid": "x**2 + y**2",
        "Saddle Point": "x**2 - y**2",
        "Plane": "2*x + 3*y + 1",
        "Cylinder": "x**2",
    },
    "Trigonometric": {
        "Wave Pattern": "np.sin(x) * np.cos(y)",
        "Ripple": "np.sin(np.sqrt(x**2 + y**2))",
        "Standing Wave": "np.sin(x) * np.sin(y)",
        "Twisted Surface": "np.sin(x*y)",
    },
    "Exponential & Logarithmic": {
        "Gaussian": "np.exp(-(x**2 + y**2))",
        "Exponential Decay": "np.exp(-np.abs(x) - np.abs(y))",
        "Log Surface": "np.log(x**2 + y**2 + 1)",
        "Mexican Hat": "(1 - (x**2 + y**2)) * np.exp(-(x**2 + y**2)/2)",
    },
    "Advanced": {
        "Monkey Saddle": "x**3 - 3*x*y**2",
        "Himmelblau": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "Rosenbrock": "100*(y - x**2)**2 + (1 - x)**2",
        "Ackley": "-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20",
    }
}

# Custom CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
            }
            .main-container {
                background: white;
                margin: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .control-panel {
                background: #f8f9fa;
                border-right: 2px solid #e9ecef;
                height: 100vh;
                overflow-y: auto;
            }
            .section-header {
                background: #343a40;
                color: white;
                padding: 10px;
                margin: 0;
                font-size: 14px;
                font-weight: bold;
            }
            .control-group {
                padding: 15px;
                border-bottom: 1px solid #e9ecef;
            }
            .info-panel {
                background: #e3f2fd;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
            .error-panel {
                background: #ffebee;
                border: 1px solid #f44336;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
            .success-panel {
                background: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Main app layout
app.layout = html.Div([
    html.Div([
        # Header
        html.Div([
            html.H1("🧮 Advanced Multivariable Calculus Explorer", style={'margin': 0}),
            html.P("Interactive visualization and analysis platform for multivariable functions", 
                   style={'margin': '10px 0 0 0', 'opacity': 0.9})
        ], className='header'),
        
        html.Div([
            # Left Control Panel
            html.Div([
                # Function Selection
                html.Div([
                    html.H4("📊 Function Selection", className='section-header'),
                    html.Div([
                        html.Label("Function Library:"),
                        dcc.Dropdown(
                            id='function-category',
                            options=[{'label': k, 'value': k} for k in FUNCTION_LIBRARY.keys()],
                            value='Basic Functions',
                            style={'marginBottom': 10}
                        ),
                        dcc.Dropdown(
                            id='function-dropdown',
                            style={'marginBottom': 15}
                        ),
                        
                        html.Label("Custom Function f(x,y):"),
                        dcc.Textarea(
                            id='custom-function',
                            value='x**2 + y**2',
                            placeholder='Enter function like: x**2 + y**2\nSupports: np.sin, np.cos, np.exp, np.log, np.sqrt, etc.',
                            style={'width': '100%', 'height': 80, 'marginBottom': 15}
                        ),
                        
                        html.Button("🔄 Reset to Default", id='reset-function', n_clicks=0,
                                   style={'width': '100%', 'marginBottom': 10}),
                    ], className='control-group')
                ]),
                
                # Range Controls
                html.Div([
                    html.H4("📐 Domain Settings", className='section-header'),
                    html.Div([
                        html.Label("X Range:"),
                        html.Div([
                            dcc.Input(id='x-min', type='number', value=-3, step=0.1, 
                                     style={'width': '45%', 'marginRight': '10%'}),
                            dcc.Input(id='x-max', type='number', value=3, step=0.1, 
                                     style={'width': '45%'})
                        ]),
                        
                        html.Label("Y Range:", style={'marginTop': 10}),
                        html.Div([
                            dcc.Input(id='y-min', type='number', value=-3, step=0.1, 
                                     style={'width': '45%', 'marginRight': '10%'}),
                            dcc.Input(id='y-max', type='number', value=3, step=0.1, 
                                     style={'width': '45%'})
                        ]),
                        
                        html.Label("Quick Presets:", style={'marginTop': 15}),
                        html.Div([
                            html.Button("Micro [-1,1]", id='preset-micro', n_clicks=0, 
                                       style={'width': '30%', 'marginRight': '5%', 'fontSize': '10px'}),
                            html.Button("Normal [-3,3]", id='preset-normal', n_clicks=0, 
                                       style={'width': '30%', 'marginRight': '5%', 'fontSize': '10px'}),
                            html.Button("Macro [-10,10]", id='preset-macro', n_clicks=0, 
                                       style={'width': '30%', 'fontSize': '10px'})
                        ], style={'marginBottom': 10}),
                        
                        html.Button("🌌 Extreme Range", id='preset-extreme', n_clicks=0,
                                   style={'width': '100%', 'fontSize': '10px'}),
                    ], className='control-group')
                ]),
                
                # Visualization Settings
                html.Div([
                    html.H4("🎨 Visualization", className='section-header'),
                    html.Div([
                        html.Label("Resolution:"),
                        dcc.Slider(
                            id='resolution',
                            min=20, max=200, step=10, value=50,
                            marks={i: str(i) for i in [20, 50, 100, 150, 200]},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Color Scheme:", style={'marginTop': 15}),
                        dcc.Dropdown(
                            id='colorscale',
                            options=[
                                {'label': '🌈 Viridis', 'value': 'Viridis'},
                                {'label': '🔥 Plasma', 'value': 'Plasma'},
                                {'label': '🌊 Blues', 'value': 'Blues'},
                                {'label': '🌋 Hot', 'value': 'Hot'},
                                {'label': '🎨 Rainbow', 'value': 'Rainbow'},
                                {'label': '🌙 Greys', 'value': 'Greys'}
                            ],
                            value='Viridis',
                            style={'marginBottom': 15}
                        ),
                        
                        html.Label("Display Options:"),
                        dcc.Checklist(
                            id='viz-options',
                            options=[
                                {'label': '📈 Show Contour Lines', 'value': 'contour'},
                                {'label': '🎯 Show Critical Points', 'value': 'critical'},
                                {'label': '🧭 Show Gradient Vectors', 'value': 'gradient'},
                                {'label': '📊 Show Wireframe', 'value': 'wireframe'},
                                {'label': '🎯 Smart Hover (Snap to Key Points)', 'value': 'hover'},
                                {'label': '📐 Show Coordinate Grid', 'value': 'grid'}
                            ],
                            value=['contour', 'hover'],
                            style={'marginBottom': 15},
                            labelStyle={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px', 'fontSize': '13px'}
                        )
                    ], className='control-group')
                ])
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}, 
               className='control-panel'),
            
            # Main Content Area
            html.Div([
                # Tabs for different visualizations
                dcc.Tabs(id="main-tabs", value="3d-surface", children=[
                    dcc.Tab(label="🏔️ 3D Surface", value="3d-surface"),
                    dcc.Tab(label="🗺️ Contour Map", value="contour"),
                    dcc.Tab(label="🧭 Vector Field", value="vector-field"),
                    dcc.Tab(label="📊 Cross Sections", value="cross-sections"),
                    dcc.Tab(label="🔬 Analysis", value="analysis")
                ]),
                
                # Content area with loading indicator
                dcc.Loading(
                    id="main-loading",
                    type="default",
                    children=html.Div(id="main-content", style={'padding': 20}),
                    style={'margin': '20px 0'}
                )
            ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'display': 'flex'})
    ], className='main-container'),
    
    # Hidden divs for storing data
    html.Div(id='function-data', style={'display': 'none'}),
    html.Div(id='analysis-data', style={'display': 'none'}),
    
    # Hidden cross-section controls (values controlled from Analysis tab)
    dcc.Input(id='cross-x', type='number', value=0, step=0.1, style={'display': 'none'}),
    dcc.Input(id='cross-y', type='number', value=0, step=0.1, style={'display': 'none'}),
    dcc.Input(id='cross-x-analysis', type='number', value=0, step=0.1, style={'display': 'none'}),
    dcc.Input(id='cross-y-analysis', type='number', value=0, step=0.1, style={'display': 'none'})
])

# Utility functions
def format_math_expression(expr_str):
    """Convert mathematical expressions to user-friendly format with proper superscripts"""
    if not expr_str:
        return expr_str
    
    # Convert to string if it's a sympy expression
    if hasattr(expr_str, '__str__'):
        expr_str = str(expr_str)
    
    # Dictionary for superscript conversion
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
                   '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
    
    # Dictionary for subscript conversion
    subscripts = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', 
                 '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    
    # Replace ** with ^ first for easier processing
    expr_str = expr_str.replace('**', '^')
    
    # Handle simple exponents like x^2, y^3, etc.
    def replace_exponent(match):
        base = match.group(1)
        exp = match.group(2)
        # Handle multi-digit exponents
        if all(c in superscripts for c in exp):
            return base + ''.join(superscripts[c] for c in exp)
        else:
            return f"{base}^{{{exp}}}"  # Use curly braces for complex exponents
    
    # Pattern for variable^number (including multi-digit)
    expr_str = re.sub(r'([a-zA-Z])[\^]([0-9]+)', replace_exponent, expr_str)
    
    # Handle parentheses with exponents like (x + y)^2
    def replace_paren_exponent(match):
        content = match.group(1)
        exp = match.group(2)
        if all(c in superscripts for c in exp):
            return f"({content})" + ''.join(superscripts[c] for c in exp)
        else:
            return f"({content})^{{{exp}}}"
    
    expr_str = re.sub(r'\(([^)]+)\)[\^]([0-9]+)', replace_paren_exponent, expr_str)
    
    # Remove multiplication asterisks for cleaner display
    expr_str = re.sub(r'(\d+)\*([a-zA-Z])', r'\1\2', expr_str)  # 2*x -> 2x
    expr_str = re.sub(r'([a-zA-Z])\*([a-zA-Z])', r'\1\2', expr_str)  # x*y -> xy
    expr_str = re.sub(r'([a-zA-Z])\*(\d+)', r'\1\2', expr_str)  # x*2 -> x2
    
    # Handle superscripted variables in multiplication
    expr_str = re.sub(r'([a-zA-Z⁰¹²³⁴⁵⁶⁷⁸⁹]+)\*([a-zA-Z])', r'\1\2', expr_str)  # x²*y -> x²y
    expr_str = re.sub(r'([a-zA-Z])\*([a-zA-Z⁰¹²³⁴⁵⁶⁷⁸⁹]+)', r'\1\2', expr_str)  # x*y² -> xy²
    expr_str = re.sub(r'(\d+)\*([a-zA-Z⁰¹²³⁴⁵⁶⁷⁸⁹]+)', r'\1\2', expr_str)  # 4*x² -> 4x²
    
    # Handle subscripts for variables like x_1, y_2, etc.
    def replace_subscript(match):
        base = match.group(1)
        sub = match.group(2)
        if all(c in subscripts for c in sub):
            return base + ''.join(subscripts[c] for c in sub)
        else:
            return f"{base}_{{{sub}}}"  # Use curly braces for complex subscripts
    
    expr_str = re.sub(r'([a-zA-Z])_([0-9]+)', replace_subscript, expr_str)
    
    # Handle mathematical notation like fxx, fyy, fxy -> f_xx, f_yy, f_xy
    expr_str = re.sub(r'\bf([a-z])([a-z])\b', lambda m: f'f_{m.group(1)}{m.group(2)}', expr_str)
    
    # Now convert the subscripts we just created
    expr_str = re.sub(r'([a-zA-Z])_([a-z]+)', lambda m: m.group(1) + ''.join(subscripts.get(c, c) for c in m.group(2)), expr_str)
    
    # Replace common mathematical symbols
    replacements = {
        'sqrt': '√',
        'pi': 'π',
        'inf': '∞',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'theta': 'θ',
        'lambda': 'λ',
        'mu': 'μ',
        'sigma': 'σ',
        'phi': 'φ',
        'omega': 'ω'
    }
    
    for old, new in replacements.items():
        expr_str = expr_str.replace(old, new)
    
    return expr_str

def safe_eval_function(func_str, x, y):
    """Safely evaluate function string with comprehensive math support"""
    try:
        # Create comprehensive namespace
        namespace = {
            'x': x, 'y': y, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs, 'sign': np.sign,
            'pi': np.pi, 'e': np.e,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
            'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
            'min': np.minimum, 'max': np.maximum
        }
        result = eval(func_str, {"__builtins__": {}}, namespace)
        
        # Handle scalar results
        if np.isscalar(result):
            if isinstance(x, np.ndarray):
                result = np.full_like(x, result)
            else:
                result = float(result)
        
        return result
    except Exception as e:
        # Return a simple paraboloid as fallback
        return x**2 + y**2

def symbolic_analysis(func_str):
    """Perform symbolic analysis of the function"""
    try:
        x, y = sp.symbols('x y', real=True)
        
        # Convert numpy functions to sympy with better mapping
        func_str_sympy = func_str.replace('np.', '')
        
        # Handle common numpy functions first (before pi/e replacement)
        replacements = {
            'sin(': 'sp.sin(',
            'cos(': 'sp.cos(',
            'tan(': 'sp.tan(',
            'exp(': 'sp.exp(',
            'log(': 'sp.log(',
            'sqrt(': 'sp.sqrt(',
            'abs(': 'sp.Abs('
        }
        
        for np_func, sp_func in replacements.items():
            func_str_sympy = func_str_sympy.replace(np_func, sp_func)
        
        # Handle constants
        func_str_sympy = func_str_sympy.replace('pi', 'sp.pi')
        func_str_sympy = func_str_sympy.replace('e', 'sp.E')
        
        # Define the function symbolically with proper namespace
        namespace = {
            'x': x, 'y': y, 'sp': sp,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'pi': sp.pi, 'E': sp.E, 'abs': sp.Abs
        }
        
        f = sp.sympify(func_str_sympy, locals=namespace)
        
        # Calculate partial derivatives
        df_dx = sp.diff(f, x)
        df_dy = sp.diff(f, y)
        
        # Calculate second derivatives (Hessian)
        d2f_dx2 = sp.diff(df_dx, x)
        d2f_dy2 = sp.diff(df_dy, y)
        d2f_dxdy = sp.diff(df_dx, y)
        
        # Calculate gradient magnitude
        grad_magnitude = sp.sqrt(df_dx**2 + df_dy**2)
        
        return {
            'function': f,
            'df_dx': df_dx,
            'df_dy': df_dy,
            'd2f_dx2': d2f_dx2,
            'd2f_dy2': d2f_dy2,
            'd2f_dxdy': d2f_dxdy,
            'grad_magnitude': grad_magnitude,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def find_critical_points_advanced(func_str, x_range, y_range, num_attempts=16):
    """Fast critical point finding with optimized methods"""
    try:
        def f(vars):
            x, y = vars
            return safe_eval_function(func_str, x, y)
        
        def gradient(vars):
            x, y = vars
            h = 1e-6
            dx = (f([x + h, y]) - f([x - h, y])) / (2 * h)
            dy = (f([x, y + h]) - f([x, y - h])) / (2 * h)
            return np.array([dx, dy])
        
        critical_points = []
        
        # Reduced grid search for speed
        grid_size = 4  # Much smaller grid
        x_guesses = np.linspace(x_range[0], x_range[1], grid_size)
        y_guesses = np.linspace(y_range[0], y_range[1], grid_size)
        
        # Always include key points
        key_points = [(0, 0)]  # Origin
        if x_range[0] <= 0 <= x_range[1] and y_range[0] <= 0 <= y_range[1]:
            x_guesses = np.append(x_guesses, 0)
            y_guesses = np.append(y_guesses, 0)
        
        for x0 in x_guesses:
            for y0 in y_guesses:
                try:
                    # Quick gradient check first
                    grad = gradient([x0, y0])
                    grad_norm = np.linalg.norm(grad)
                    
                    if grad_norm < 1e-3:  # Relaxed tolerance for speed
                        x_crit, y_crit = x0, y0
                    else:
                        # Try only one fast method
                        try:
                            result = minimize(lambda p: np.linalg.norm(gradient(p))**2, 
                                            [x0, y0], method='BFGS', 
                                            options={'maxiter': 20})  # Limit iterations
                            if result.success:
                                test_grad = gradient(result.x)
                                if np.linalg.norm(test_grad) < 1e-3:
                                    x_crit, y_crit = result.x
                                else:
                                    continue
                            else:
                                continue
                        except:
                            continue
                    
                    # Quick domain check
                    if not (x_range[0] <= x_crit <= x_range[1] and 
                           y_range[0] <= y_crit <= y_range[1]):
                        continue
                    
                    # Check if already found
                    is_new = True
                    for existing in critical_points:
                        if (abs(existing['x'] - x_crit) < 1e-3 and 
                            abs(existing['y'] - y_crit) < 1e-3):
                            is_new = False
                            break
                    
                    if is_new:
                        z_crit = f([x_crit, y_crit])
                        
                        # Fast Hessian calculation
                        h = 1e-5
                        fxx = (f([x_crit + h, y_crit]) - 2*f([x_crit, y_crit]) + 
                              f([x_crit - h, y_crit])) / h**2
                        fyy = (f([x_crit, y_crit + h]) - 2*f([x_crit, y_crit]) + 
                              f([x_crit, y_crit - h])) / h**2
                        fxy = (f([x_crit + h, y_crit + h]) - f([x_crit + h, y_crit - h]) -
                              f([x_crit - h, y_crit + h]) + f([x_crit - h, y_crit - h])) / (4*h**2)
                        
                        discriminant = fxx * fyy - fxy**2
                        
                        if discriminant > 0:
                            point_type = "Local Minimum" if fxx > 0 else "Local Maximum"
                        elif discriminant < 0:
                            point_type = "Saddle Point"
                        else:
                            point_type = "Inconclusive"
                        
                        critical_points.append({
                            'x': x_crit, 'y': y_crit, 'z': z_crit,
                            'type': point_type,
                            'discriminant': discriminant,
                            'fxx': fxx, 'fyy': fyy, 'fxy': fxy
                        })
                        
                        if len(critical_points) >= 5:  # Limit for speed
                            break
                
                except Exception:
                    continue
            
            if len(critical_points) >= 5:
                break
        
        return critical_points
    except Exception as e:
        return []

def smart_round(value, threshold=0.15):
    """Smart rounding that aggressively snaps to clean integer and half-integer values"""
    # Priority list: integers first, then halves, then quarters
    integers = list(range(-5, 6))  # -5 to 5
    
    # Very aggressive snapping to integers
    for key_val in integers:
        if abs(value - key_val) < threshold:
            return int(key_val)
    
    # Then try half-integers
    half_integers = [i + 0.5 for i in range(-5, 5)]
    for key_val in half_integers:
        if abs(value - key_val) < threshold * 0.8:
            return key_val
    
    # Finally quarters
    quarters = [i + 0.25 for i in range(-5, 5)] + [i + 0.75 for i in range(-5, 5)]
    for key_val in quarters:
        if abs(value - key_val) < threshold * 0.6:
            return key_val
    
    # Force to zero if very close
    if abs(value) < 0.05:
        return 0
    
    # Otherwise round to 1 decimal place
    return round(value, 1)

def create_smart_hover_template(enable_smart_hover=True):
    """Create hover template with smart coordinate display"""
    if enable_smart_hover:
        return (
            "<b>Position:</b><br>" +
            "x = %{customdata[0]}<br>" +
            "y = %{customdata[1]}<br>" +
            "f(x,y) = %{customdata[2]}<br>" +
            "<extra></extra>"
        )
    else:
        return (
            "<b>Position:</b><br>" +
            "x = %{x:.4f}<br>" +
            "y = %{y:.4f}<br>" +
            "f(x,y) = %{z:.4f}<br>" +
            "<extra></extra>"
        )

def create_enhanced_surface_plot(X, Y, Z, func_str, viz_options, colorscale, x_range, y_range, critical_points=None):
    """Create enhanced 3D surface plot with all features"""
    fig = go.Figure()
    
    # Create smart hover data
    if 'hover' in viz_options:
        # Create custom data for smart hovering
        smart_x = np.vectorize(smart_round)(X)
        smart_y = np.vectorize(smart_round)(Y)
        smart_z = np.vectorize(lambda z: round(z, 4))(Z)
        
        # Stack the smart coordinates
        customdata = np.stack((smart_x, smart_y, smart_z), axis=-1)
        hovertemplate = create_smart_hover_template(True)
    else:
        customdata = None
        hovertemplate = "x=%{x}, y=%{y}, z=%{z}<extra></extra>"
    
    # Main surface
    surface_trace = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        name=f'f(x,y) = {func_str}',
        opacity=0.9,
        hovertemplate=hovertemplate,
        showscale=True,
        colorbar=dict(title="f(x,y)", titleside="right")
    )
    
    # Add custom data for smart hovering if enabled
    if customdata is not None:
        surface_trace.customdata = customdata
    
    fig.add_trace(surface_trace)
    
    # Add wireframe if requested
    if 'wireframe' in viz_options:
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.zeros_like(Z),
            showscale=False,
            opacity=0.1,
            contours=dict(
                x=dict(show=True, color="black", width=1),
                y=dict(show=True, color="black", width=1),
                z=dict(show=True, color="black", width=1)
            )
        ))
    
    # Add contour projections if requested
    if 'contour' in viz_options:
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=Z,
            colorscale=colorscale,
            showscale=False,
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True, width=2)
            ),
            opacity=0.3
        ))
    
    # Add critical points if available and requested
    if critical_points and 'critical' in viz_options:
        for cp in critical_points:
            color = 'red' if 'Saddle' in cp['type'] else 'blue' if 'Maximum' in cp['type'] else 'green'
            fig.add_trace(go.Scatter3d(
                x=[cp['x']], y=[cp['y']], z=[cp['z']],
                mode='markers',
                marker=dict(size=10, color=color, symbol='diamond'),
                name=f"{cp['type']} ({cp['x']:.2f}, {cp['y']:.2f})",
                hovertemplate=f"<b>{cp['type']}</b><br>x={cp['x']:.4f}<br>y={cp['y']:.4f}<br>f(x,y)={cp['z']:.4f}<extra></extra>"
            ))
    
    # Add coordinate grid if requested
    if 'grid' in viz_options:
        # Add grid lines at integer coordinates
        grid_range_x = range(int(np.floor(x_range[0])), int(np.ceil(x_range[1])) + 1)
        grid_range_y = range(int(np.floor(y_range[0])), int(np.ceil(y_range[1])) + 1)
        
        # Vertical grid lines (constant x)
        for x_grid in grid_range_x:
            if x_range[0] <= x_grid <= x_range[1]:
                y_grid_line = np.linspace(y_range[0], y_range[1], 50)
                x_grid_line = np.full_like(y_grid_line, x_grid)
                z_grid_line = safe_eval_function(func_str, x_grid_line, y_grid_line)
                
                fig.add_trace(go.Scatter3d(
                    x=x_grid_line, y=y_grid_line, z=z_grid_line,
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Horizontal grid lines (constant y)
        for y_grid in grid_range_y:
            if y_range[0] <= y_grid <= y_range[1]:
                x_grid_line = np.linspace(x_range[0], x_range[1], 50)
                y_grid_line = np.full_like(x_grid_line, y_grid)
                z_grid_line = safe_eval_function(func_str, x_grid_line, y_grid_line)
                
                fig.add_trace(go.Scatter3d(
                    x=x_grid_line, y=y_grid_line, z=z_grid_line,
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title=f'3D Surface: f(x,y) = {format_math_expression(func_str)}',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x,y)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_enhanced_contour_plot(X, Y, Z, func_str, viz_options, colorscale, critical_points=None):
    """Create enhanced contour plot with smart hovering"""
    fig = go.Figure()
    
    # Create 1D arrays for contour plot
    x_1d = X[0]
    y_1d = Y[:, 0]
    
    # Main contour plot
    fig.add_trace(go.Contour(
        x=x_1d, y=y_1d, z=Z,
        colorscale=colorscale,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white'),
            start=np.min(Z),
            end=np.max(Z),
            size=(np.max(Z) - np.min(Z)) / 20
        ),
        hovertemplate="<b>Position:</b><br>x=%{x}<br>y=%{y}<br>f(x,y)=%{z:.4f}<extra></extra>",
        colorbar=dict(title="f(x,y)"),
        hoverinfo='x+y+z'
    ))
    
    # Add gradient vectors if requested
    if 'gradient' in viz_options:
        # Subsample for gradient vectors
        step = max(1, len(X) // 20)
        X_sub = X[::step, ::step]
        Y_sub = Y[::step, ::step]
        
        # Compute gradient numerically
        h = 0.01
        dx = (safe_eval_function(func_str, X_sub + h, Y_sub) - 
              safe_eval_function(func_str, X_sub - h, Y_sub)) / (2 * h)
        dy = (safe_eval_function(func_str, X_sub, Y_sub + h) - 
              safe_eval_function(func_str, X_sub, Y_sub - h)) / (2 * h)
        
        # Normalize for better visualization
        magnitude = np.sqrt(dx**2 + dy**2)
        scale = 0.3 * (X[0, 1] - X[0, 0])  # Scale based on grid spacing
        dx_norm = dx / (magnitude + 1e-8) * scale
        dy_norm = dy / (magnitude + 1e-8) * scale
        
        # Add arrows
        for i in range(X_sub.shape[0]):
            for j in range(X_sub.shape[1]):
                if magnitude[i, j] > 1e-6:  # Only show significant gradients
                    fig.add_annotation(
                        x=X_sub[i,j], y=Y_sub[i,j],
                        ax=X_sub[i,j] + dx_norm[i,j], ay=Y_sub[i,j] + dy_norm[i,j],
                        arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red",
                        showarrow=True, axref="x", ayref="y"
                    )
    
    # Add critical points if available
    if critical_points and 'critical' in viz_options:
        for cp in critical_points:
            color = 'red' if 'Saddle' in cp['type'] else 'blue' if 'Maximum' in cp['type'] else 'green'
            fig.add_trace(go.Scatter(
                x=[cp['x']], y=[cp['y']],
                mode='markers',
                marker=dict(size=12, color=color, symbol='x'),
                name=f"{cp['type']} ({cp['x']:.2f}, {cp['y']:.2f})",
                hovertemplate=f"<b>{cp['type']}</b><br>x={cp['x']:.4f}<br>y={cp['y']:.4f}<br>f(x,y)={cp['z']:.4f}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f'Contour Plot: f(x,y) = {format_math_expression(func_str)}',
        xaxis_title='x',
        yaxis_title='y',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_vector_field_plot(func_str, x_range, y_range, colorscale, resolution=20):
    """Create vector field plot showing gradient with proper coloring"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute gradient
    h = 0.01
    dx = (safe_eval_function(func_str, X + h, Y) - safe_eval_function(func_str, X - h, Y)) / (2 * h)
    dy = (safe_eval_function(func_str, X, Y + h) - safe_eval_function(func_str, X, Y - h)) / (2 * h)
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap background showing gradient magnitude
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=magnitude,
        colorscale=colorscale,
        name='Gradient Magnitude',
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>|∇f|=%{z:.3f}<extra></extra>",
        colorbar=dict(title="|∇f|")
    ))
    
    # Add arrows using annotations
    for i in range(0, len(x), max(1, len(x)//15)):
        for j in range(0, len(y), max(1, len(y)//15)):
            mag = magnitude[j,i]
            if mag > 1e-6:
                scale = 0.3 * (x[1] - x[0])
                dx_norm = dx[j,i] / mag * scale
                dy_norm = dy[j,i] / mag * scale
                
                fig.add_annotation(
                    x=X[j,i], y=Y[j,i],
                    ax=X[j,i] + dx_norm, ay=Y[j,i] + dy_norm,
                    arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black",
                    showarrow=True, axref="x", ayref="y"
                )
    
    fig.update_layout(
        title=f'Gradient Vector Field: ∇f(x,y) for f(x,y) = {format_math_expression(func_str)}',
        xaxis_title='x',
        yaxis_title='y',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    
    return fig

def create_cross_sections_content(x, y, func_str, cross_x, cross_y, colorscale):
    """Create cross-sections visualization with smart hovering"""
    if cross_x is None:
        cross_x = 0
    if cross_y is None:
        cross_y = 0
    
    # Smart round the cross-section coordinates
    smart_cross_x = smart_round(cross_x)
    smart_cross_y = smart_round(cross_y)
    
    # Extend the domain to include cross-section coordinates with some padding
    x_min, x_max = min(x.min(), cross_x - 1), max(x.max(), cross_x + 1)
    y_min, y_max = min(y.min(), cross_y - 1), max(y.max(), cross_y + 1)
    
    # Create extended arrays for cross-sections
    x_extended = np.linspace(x_min, x_max, len(x))
    y_extended = np.linspace(y_min, y_max, len(y))
    
    # Create cross-sections using extended domains
    x_section = safe_eval_function(func_str, x_extended, cross_y)
    y_section = safe_eval_function(func_str, cross_x, y_extended)
    
    fig = go.Figure()
    
    # Create smart hover data for x-section
    smart_x_coords = [smart_round(xi) for xi in x_extended]
    smart_x_values = [round(val, 4) for val in x_section]
    
    fig.add_trace(go.Scatter(
        x=x_extended, y=x_section,
        mode='lines+markers',
        name=f'f(x, {smart_cross_y})',
        line=dict(color='red', width=3),
        marker=dict(size=4),
        hovertemplate=f"<b>X-Section:</b><br>x=%{{x}}<br>f(x, {smart_cross_y})=%{{y:.4f}}<extra></extra>",
        customdata=list(zip(smart_x_coords, smart_x_values))
    ))
    
    # Create smart hover data for y-section
    smart_y_coords = [smart_round(yi) for yi in y_extended]
    smart_y_values = [round(val, 4) for val in y_section]
    
    fig.add_trace(go.Scatter(
        x=y_extended, y=y_section,
        mode='lines+markers',
        name=f'f({smart_cross_x}, y)',
        line=dict(color='blue', width=3),
        marker=dict(size=4),
        hovertemplate=f"<b>Y-Section:</b><br>y=%{{x}}<br>f({smart_cross_x}, y)=%{{y:.4f}}<extra></extra>",
        customdata=list(zip(smart_y_coords, smart_y_values))
    ))
    
    # Add intersection point
    intersection_z = safe_eval_function(func_str, cross_x, cross_y)
    fig.add_trace(go.Scatter(
        x=[cross_x], y=[intersection_z],
        mode='markers',
        name=f'Intersection f({cross_x}, {cross_y})',
        marker=dict(size=12, color='green', symbol='diamond'),
        hovertemplate=f"Intersection<br>f({cross_x}, {cross_y}) = {intersection_z:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f'Cross-sections of f(x,y) = {func_str}',
        xaxis_title='x or y',
        yaxis_title='f(x,y)',
        height=600,
        legend=dict(x=0.02, y=0.98)
    )
    
    return [
        html.Div([
            html.H3("📊 Cross-Section Analysis"),
            html.Div([
                html.H4("Set Cross-Section Coordinates:"),
                html.Div([
                    html.Label("x = ", style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '16px'}),
                    dcc.Input(
                        id='cross-x-temp', 
                        type='number', 
                        value=cross_x, 
                        step=0.1,
                        style={'width': '100px', 'padding': '8px', 'marginRight': '20px', 'fontSize': '14px', 'border': '2px solid #ddd', 'borderRadius': '5px'}
                    ),
                    html.Label("y = ", style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '16px'}),
                    dcc.Input(
                        id='cross-y-temp', 
                        type='number', 
                        value=cross_y, 
                        step=0.1,
                        style={'width': '100px', 'padding': '8px', 'marginRight': '20px', 'fontSize': '14px', 'border': '2px solid #ddd', 'borderRadius': '5px'}
                    ),
                    html.Button(
                        "✅ Update Cross-Sections", 
                        id='update-cross-sections-btn', 
                        n_clicks=0,
                        style={
                            'padding': '8px 16px', 
                            'backgroundColor': '#28a745', 
                            'color': 'white', 
                            'border': 'none', 
                            'borderRadius': '5px', 
                            'cursor': 'pointer', 
                            'fontSize': '14px',
                            'fontWeight': 'bold'
                        }
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
                html.P("💡 Adjust the x and y values above, then click 'Update Cross-Sections' to refresh the graph.", 
                       style={'fontSize': '13px', 'fontStyle': 'italic', 'color': '#666', 'marginBottom': '20px'})
            ]),
            html.P(f"🔴 Red line: f(x, {smart_cross_y}) - function behavior along x-axis at y = {smart_cross_y}"),
            html.P(f"🔵 Blue line: f({smart_cross_x}, y) - function behavior along y-axis at x = {smart_cross_x}"),
            html.P(f"💎 Green diamond: Intersection point f({smart_cross_x}, {smart_cross_y}) = {intersection_z:.4f}"),
        ], className='info-panel'),
        dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
    ]

# Callback functions
@app.callback(
    Output('function-dropdown', 'options'),
    Output('function-dropdown', 'value'),
    Input('function-category', 'value')
)
def update_function_dropdown(category):
    if category and category in FUNCTION_LIBRARY:
        options = [{'label': k, 'value': v} for k, v in FUNCTION_LIBRARY[category].items()]
        return options, list(FUNCTION_LIBRARY[category].values())[0]
    return [], None

@app.callback(
    Output('custom-function', 'value'),
    Input('function-dropdown', 'value'),
    Input('reset-function', 'n_clicks')
)
def update_custom_function(selected_func, reset_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'x**2 + y**2'
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-function':
        return 'x**2 + y**2'
    elif trigger_id == 'function-dropdown' and selected_func:
        return selected_func
    
    return 'x**2 + y**2'

@app.callback(
    [Output('x-min', 'value'), Output('x-max', 'value'),
     Output('y-min', 'value'), Output('y-max', 'value')],
    [Input('preset-micro', 'n_clicks'),
     Input('preset-normal', 'n_clicks'),
     Input('preset-macro', 'n_clicks'),
     Input('preset-extreme', 'n_clicks')]
)
def update_range_presets(micro, normal, macro, extreme):
    ctx = dash.callback_context
    if not ctx.triggered:
        return -3, 3, -3, 3
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'preset-micro':
        return -1, 1, -1, 1
    elif trigger_id == 'preset-normal':
        return -3, 3, -3, 3
    elif trigger_id == 'preset-macro':
        return -10, 10, -10, 10
    elif trigger_id == 'preset-extreme':
        return -50, 50, -50, 50
    
    return -3, 3, -3, 3

@app.callback(
    Output('main-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('custom-function', 'value'),
     Input('x-min', 'value'), Input('x-max', 'value'),
     Input('y-min', 'value'), Input('y-max', 'value'),
     Input('resolution', 'value'),
     Input('colorscale', 'value'),
     Input('viz-options', 'value'),
     Input('cross-x', 'value'),
     Input('cross-y', 'value')]
)
def update_main_content(active_tab, func_str, x_min, x_max, y_min, y_max, 
                       resolution, colorscale, viz_options, cross_x, cross_y):
    
    # Validate inputs
    if not func_str:
        func_str = 'x**2 + y**2'
    
    if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max, y_min, y_max = -3, 3, -3, 3
    
    if x_min >= x_max:
        x_min, x_max = -3, 3
    if y_min >= y_max:
        y_min, y_max = -3, 3
    
    if resolution is None or resolution < 10:
        resolution = 50
    
    if viz_options is None:
        viz_options = []
    
    # Create coordinate arrays
    x_range = [x_min, x_max]
    y_range = [y_min, y_max]
    
    # Limit resolution for performance
    resolution = min(resolution, 150)
    
    try:
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = safe_eval_function(func_str, X, Y)
        
        # Find critical points if needed
        critical_points = []
        if 'critical' in viz_options or active_tab == 'analysis':
            critical_points = find_critical_points_advanced(func_str, x_range, y_range)
        
        # Generate content based on active tab
        if active_tab == '3d-surface':
            fig = create_enhanced_surface_plot(X, Y, Z, func_str, viz_options, colorscale, x_range, y_range, critical_points)
            return dcc.Graph(figure=fig, style={'height': '600px'})
        
        elif active_tab == 'contour':
            fig = create_enhanced_contour_plot(X, Y, Z, func_str, viz_options, colorscale, critical_points)
            return dcc.Graph(figure=fig, style={'height': '600px'})
        
        elif active_tab == 'vector-field':
            fig = create_vector_field_plot(func_str, x_range, y_range, colorscale, min(resolution//3, 25))
            return dcc.Graph(figure=fig, style={'height': '600px'})
        
        elif active_tab == 'cross-sections':
            return create_cross_sections_content(x, y, func_str, cross_x or 0, cross_y or 0, colorscale)
        
        elif active_tab == 'analysis':
            return create_analysis_content(func_str, x_range, y_range, critical_points)
        
    except Exception as e:
        return html.Div([
            html.H4("⚠️ Error", style={'color': 'red'}),
            html.P(f"Error processing function: {str(e)}"),
            html.P("Please check your function syntax and try again."),
            html.Hr(),
            html.H5("Function Syntax Help:"),
            html.Ul([
                html.Li("Use ** for exponents: x**2, y**3"),
                html.Li("Available functions: np.sin, np.cos, np.exp, np.log, np.sqrt"),
                html.Li("Constants: np.pi, np.e"),
                html.Li("Example: np.sin(x) * np.cos(y) + x**2")
            ])
        ], className='error-panel')

def create_analysis_content(func_str, x_range, y_range, critical_points):
    """Create comprehensive analysis content with all information displayed automatically"""
    content = []
    
    # Function information header
    content.append(html.Div([
        html.H3("Complete Function Analysis"),
        html.P(f"Function: f(x,y) = {format_math_expression(func_str)}"),
        html.P(f"Domain: x ∈ [{x_range[0]}, {x_range[1]}], y ∈ [{y_range[0]}, {y_range[1]}]"),
        html.P(f"Domain area: {(x_range[1] - x_range[0]) * (y_range[1] - y_range[0]):.2f} square units")
    ], className='info-panel'))
    
    # Symbolic analysis - always show
    analysis = symbolic_analysis(func_str)
    if analysis['success']:
        content.append(html.Div([
            html.H4("📝 Symbolic Derivatives"),
            html.Div([
                html.P([html.Strong("First-order partial derivatives:")]),
                html.P([html.Strong("∂f/∂x = "), format_math_expression(str(analysis['df_dx']))]),
                html.P([html.Strong("∂f/∂y = "), format_math_expression(str(analysis['df_dy']))]),
                html.Hr(),
                html.P([html.Strong("Second-order partial derivatives (Hessian matrix):")]),
                html.P([html.Strong("∂²f/∂x² = "), format_math_expression(str(analysis['d2f_dx2']))]),
                html.P([html.Strong("∂²f/∂y² = "), format_math_expression(str(analysis['d2f_dy2']))]),
                html.P([html.Strong("∂²f/∂x∂y = "), format_math_expression(str(analysis['d2f_dxdy']))]),
                html.Hr(),
                html.P([html.Strong("Gradient magnitude: |∇f| = "), format_math_expression(str(analysis['grad_magnitude']))])
            ])
        ], className='success-panel'))
        
        # Integration analysis
        try:
            x, y = sp.symbols('x y', real=True)
            f = analysis['function']
            
            # Try partial integrals
            try:
                integral_x = sp.integrate(f, x)
                integral_y = sp.integrate(f, y)
                
                content.append(html.Div([
                    html.H4("∫ Symbolic Integration"),
                    html.P([html.Strong("Partial integrals:")]),
                    html.P([html.Strong("∫ f(x,y) dx = "), format_math_expression(str(integral_x))]),
                    html.P([html.Strong("∫ f(x,y) dy = "), format_math_expression(str(integral_y))])
                ], className='success-panel'))
            except:
                content.append(html.Div([
                    html.H4("∫ Symbolic Integration"),
                    html.P("Integration is too complex for symbolic computation."),
                    html.P("This function may require numerical integration methods.")
                ], className='info-panel'))
        except:
            pass
            
    else:
        content.append(html.Div([
            html.H4("📝 Symbolic Analysis"),
            html.P(f"❌ Symbolic analysis failed: {analysis.get('error', 'Unknown error')}"),
            html.P("This function may be too complex for symbolic differentiation."),
            html.P("Numerical methods are being used for critical point detection."),
            html.Hr(),
            html.H5("📊 Numerical Analysis at Origin (0,0)"),
            create_numerical_derivatives_at_point(func_str, 0, 0)
        ], className='error-panel'))
    
    # Critical points analysis - always show
    if critical_points:
        content.append(html.Div([
            html.H4("🎯 Critical Points Analysis"),
            html.P(f"Found {len(critical_points)} critical point(s) in the domain:"),
            dash_table.DataTable(
                data=[
                    {
                        'Point': f"({cp['x']:.4f}, {cp['y']:.4f})",
                        'Value': f"{cp['z']:.4f}",
                        'Type': cp['type'],
                        'Discriminant': f"{cp['discriminant']:.4f}",
                        'fxx': f"{cp['fxx']:.4f}",
                        'fyy': f"{cp['fyy']:.4f}",
                        'fxy': f"{cp['fxy']:.4f}"
                    }
                    for cp in critical_points
                ],
                columns=[
                    {'name': 'Point (x,y)', 'id': 'Point'},
                    {'name': 'f(x,y)', 'id': 'Value'},
                    {'name': 'Type', 'id': 'Type'},
                    {'name': 'Discriminant D', 'id': 'Discriminant'},
                    {'name': 'fxx', 'id': 'fxx'},
                    {'name': 'fyy', 'id': 'fyy'},
                    {'name': 'fxy', 'id': 'fxy'}
                ],
                style_cell={'textAlign': 'center', 'fontSize': '12px', 'padding': '8px'},
                style_header={'backgroundColor': '#4facfe', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Type} contains "Maximum"'},
                        'backgroundColor': '#ffebee',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Type} contains "Minimum"'},
                        'backgroundColor': '#e8f5e8',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Type} contains "Saddle"'},
                        'backgroundColor': '#fff3e0',
                        'color': 'black',
                    }
                ]
            ),
            html.P("Note: D = fxx·fyy - (fxy)². D > 0: extremum, D < 0: saddle point, D = 0: inconclusive")
        ], className='success-panel'))
    else:
        content.append(html.Div([
            html.H4("🎯 Critical Points Analysis"),
            html.P("No critical points found in the current domain."),
            html.P("Possible reasons:"),
            html.Ul([
                html.Li("Function has no critical points in this range"),
                html.Li("Critical points are outside the specified domain"),
                html.Li("Function is monotonic in this region"),
                html.Li("Numerical optimization failed to converge")
            ]),
            html.P("Try expanding the domain range or using a different function."),
            html.Hr(),
            html.H5("🔍 Sample Analysis at Key Points"),
            html.Div([
                html.P("Analysis at origin (0,0):"),
                create_numerical_derivatives_at_point(func_str, 0, 0)
            ])
        ], className='info-panel'))
    
    # Function statistics
    try:
        # Sample the function for statistics
        x_sample = np.linspace(x_range[0], x_range[1], 50)
        y_sample = np.linspace(y_range[0], y_range[1], 50)
        X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
        Z_sample = safe_eval_function(func_str, X_sample, Y_sample)
        
        # Calculate statistics
        if np.any(np.isfinite(Z_sample)):
            valid_mask = np.isfinite(Z_sample)
            valid_values = Z_sample[valid_mask]
            
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            
            content.append(html.Div([
                html.H4("📊 Function Statistics"),
                html.P(f"Range: f(x,y) ∈ [{min_val:.6f}, {max_val:.6f}]"),
                html.P(f"Mean value: {mean_val:.6f}"),
                html.P(f"Standard deviation: {std_val:.6f}"),
                html.P(f"Valid samples: {np.sum(valid_mask)}/{Z_sample.size} ({100*np.sum(valid_mask)/Z_sample.size:.1f}%)")
            ], className='info-panel'))
    except Exception as e:
        content.append(html.Div([
            html.H4("📊 Function Statistics"),
            html.P(f"❌ Statistics calculation failed: {str(e)}"),
            html.P("The function may have domain restrictions or numerical issues.")
        ], className='error-panel'))
    


    # Educational notes
    content.append(html.Div([
        html.H4("📚 Mathematical Reference"),
        html.Div([
            html.H5("Critical Point Classification:"),
            html.Ul([
                html.Li("Critical points occur where ∇f = 0 (both ∂f/∂x = 0 and ∂f/∂y = 0)"),
                html.Li("Second derivative test uses discriminant D = fxx·fyy - (fxy)²:"),
                html.Ul([
                    html.Li("D > 0 and fxx > 0: Local minimum"),
                    html.Li("D > 0 and fxx < 0: Local maximum"),
                    html.Li("D < 0: Saddle point"),
                    html.Li("D = 0: Test inconclusive (requires further analysis)")
                ])
            ]),
            html.H5("Visualization Guide:"),
            html.Ul([
                html.Li("3D Surface: Shows the complete function behavior"),
                html.Li("Contour Map: Level curves showing constant function values"),
                html.Li("Vector Field: Gradient vectors showing direction of steepest increase"),
                html.Li("Cross Sections: 2D slices showing function behavior along lines")
            ])
        ])
    ], className='info-panel'))
    
    return content

# Callback to update cross-section values only when button is clicked
@app.callback(
    [Output('cross-x', 'value'), Output('cross-y', 'value')],
    [Input('update-cross-sections-btn', 'n_clicks')],
    [State('cross-x-temp', 'value'), State('cross-y-temp', 'value')]
)
def update_cross_section_values(n_clicks, cross_x_temp, cross_y_temp):
    if n_clicks > 0:
        return cross_x_temp or 0, cross_y_temp or 0
    return 0, 0

# Analysis content is now generated automatically in create_analysis_content function

if __name__ == '__main__':
    print("🚀 Starting Advanced Multivariable Calculus Explorer...")
    print("📊 Comprehensive educational visualization platform")
    print("🌐 URL: http://localhost:8050")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)