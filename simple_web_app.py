#!/usr/bin/env python3
"""
Simple Web-based Multivariable Calculus Visualization
Interactive plotting in browser with customizable functions
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import minimize

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Multivariable Calculus Explorer"

# Predefined functions
FUNCTIONS = {
    "Paraboloid": "x**2 + y**2",
    "Saddle Point": "x**2 - y**2", 
    "Gaussian": "np.exp(-(x**2 + y**2))",
    "Ripple": "np.sin(np.sqrt(x**2 + y**2))",
    "Monkey Saddle": "x**3 - 3*x*y**2",
    "Custom": "x**2 + y**2"  # Default for custom
}

# App layout
app.layout = html.Div([
    html.H1("Multivariable Calculus Explorer", style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        # Controls Panel
        html.Div([
            html.H3("Function Controls"),
            
            html.Label("Select Function:"),
            dcc.Dropdown(
                id='function-dropdown',
                options=[{'label': k, 'value': k} for k in FUNCTIONS.keys()],
                value='Paraboloid',
                style={'marginBottom': 15}
            ),
            
            html.Label("Custom Function f(x,y):"),
            dcc.Input(
                id='custom-function',
                type='text',
                value='x**2 + y**2',
                placeholder='Enter function like: x**2 + y**2',
                style={'width': '100%', 'marginBottom': 15}
            ),
            
            html.Label("X Range:"),
            html.Div([
                html.Div([
                    html.Label("Min X:", style={'fontSize': '12px'}),
                    dcc.Input(
                        id='x-min',
                        type='number',
                        value=-3,
                        step=0.1,
                        style={'width': '80px', 'marginRight': '10px'}
                    )
                ], style={'display': 'inline-block'}),
                html.Div([
                    html.Label("Max X:", style={'fontSize': '12px'}),
                    dcc.Input(
                        id='x-max',
                        type='number',
                        value=3,
                        step=0.1,
                        style={'width': '80px'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'marginBottom': '10px'}),
            
            dcc.RangeSlider(
                id='x-range-slider',
                min=-10, max=10, step=0.1,
                marks={i: str(i) for i in range(-10, 11, 2)},
                value=[-3, 3],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Y Range:", style={'marginTop': 15}),
            html.Div([
                html.Div([
                    html.Label("Min Y:", style={'fontSize': '12px'}),
                    dcc.Input(
                        id='y-min',
                        type='number',
                        value=-3,
                        step=0.1,
                        style={'width': '80px', 'marginRight': '10px'}
                    )
                ], style={'display': 'inline-block'}),
                html.Div([
                    html.Label("Max Y:", style={'fontSize': '12px'}),
                    dcc.Input(
                        id='y-max',
                        type='number',
                        value=3,
                        step=0.1,
                        style={'width': '80px'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'marginBottom': '10px'}),
            
            dcc.RangeSlider(
                id='y-range-slider',
                min=-10, max=10, step=0.1,
                marks={i: str(i) for i in range(-10, 11, 2)},
                value=[-3, 3],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Resolution:", style={'marginTop': 15}),
            dcc.Slider(
                id='resolution',
                min=20, max=100, step=10,
                marks={i: str(i) for i in range(20, 101, 20)},
                value=50
            ),
            
            html.Hr(),
            
            html.Label("Visualization Options:"),
            dcc.Checklist(
                id='viz-options',
                options=[
                    {'label': ' Show Contour Lines', 'value': 'contour'},
                    {'label': ' Show Critical Points', 'value': 'critical'}
                ],
                value=['contour'],
                style={'marginBottom': 15}
            ),
            
            html.Label("Quick Range Presets:", style={'marginTop': 15}),
            html.Div([
                html.Button("Zoom In [-1,1]", id='preset-small', n_clicks=0, 
                           style={'marginRight': '5px', 'fontSize': '10px', 'padding': '5px'}),
                html.Button("Default [-3,3]", id='preset-default', n_clicks=0, 
                           style={'marginRight': '5px', 'fontSize': '10px', 'padding': '5px'}),
                html.Button("Zoom Out [-5,5]", id='preset-large', n_clicks=0, 
                           style={'fontSize': '10px', 'padding': '5px'})
            ], style={'marginBottom': 15}),
            
            html.Label("Color Scheme:"),
            dcc.Dropdown(
                id='colorscale',
                options=[
                    {'label': 'Viridis', 'value': 'Viridis'},
                    {'label': 'Plasma', 'value': 'Plasma'},
                    {'label': 'Rainbow', 'value': 'Rainbow'},
                    {'label': 'Hot', 'value': 'Hot'},
                    {'label': 'Blues', 'value': 'Blues'}
                ],
                value='Viridis'
            )
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': 20}),
        
        # Main visualization area
        html.Div([
            dcc.Tabs(id="tabs", value="surface", children=[
                dcc.Tab(label="3D Surface", value="surface"),
                dcc.Tab(label="Contour Plot", value="contour"),
                dcc.Tab(label="Gradient Field", value="gradient")
            ]),
            
            html.Div(id="tab-content", style={'marginTop': 20})
        ], style={'width': '70%', 'display': 'inline-block', 'padding': 20})
    ]),
    
    html.Hr(),
    html.Div(id="function-info", style={'padding': 20})
])

def safe_eval_function(func_str, x, y):
    """Safely evaluate function string"""
    try:
        # Create safe namespace
        namespace = {
            'x': x, 'y': y, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'abs': np.abs, 'pi': np.pi, 'e': np.e
        }
        return eval(func_str, {"__builtins__": {}}, namespace)
    except:
        return x**2 + y**2  # Fallback to paraboloid

def find_critical_points(func_str, x_range, y_range):
    """Find critical points numerically"""
    try:
        def f(vars):
            x, y = vars
            return safe_eval_function(func_str, x, y)
        
        # Try multiple starting points
        critical_points = []
        for x0 in np.linspace(x_range[0], x_range[1], 3):
            for y0 in np.linspace(y_range[0], y_range[1], 3):
                try:
                    result = minimize(f, [x0, y0], method='BFGS')
                    if result.success:
                        x_crit, y_crit = result.x
                        if x_range[0] <= x_crit <= x_range[1] and y_range[0] <= y_crit <= y_range[1]:
                            # Check if this point is already found
                            is_new = True
                            for existing in critical_points:
                                if abs(existing[0] - x_crit) < 0.1 and abs(existing[1] - y_crit) < 0.1:
                                    is_new = False
                                    break
                            if is_new:
                                z_crit = safe_eval_function(func_str, x_crit, y_crit)
                                critical_points.append((x_crit, y_crit, z_crit))
                except:
                    continue
        
        return critical_points[:3]  # Limit to 3 points
    except:
        return []

# Sync input boxes with sliders
@callback(
    [Output('x-range-slider', 'value'),
     Output('y-range-slider', 'value')],
    [Input('x-min', 'value'),
     Input('x-max', 'value'),
     Input('y-min', 'value'),
     Input('y-max', 'value')]
)
def update_sliders_from_inputs(x_min, x_max, y_min, y_max):
    # Validate inputs
    if x_min is None or x_max is None or y_min is None or y_max is None:
        return [-3, 3], [-3, 3]
    
    if x_min >= x_max:
        x_max = x_min + 0.1
    if y_min >= y_max:
        y_max = y_min + 0.1
    
    return [x_min, x_max], [y_min, y_max]

# Sync sliders with input boxes
@callback(
    [Output('x-min', 'value'),
     Output('x-max', 'value'),
     Output('y-min', 'value'),
     Output('y-max', 'value')],
    [Input('x-range-slider', 'value'),
     Input('y-range-slider', 'value')]
)
def update_inputs_from_sliders(x_range, y_range):
    return x_range[0], x_range[1], y_range[0], y_range[1]

# Handle preset range buttons
@callback(
    [Output('x-range-slider', 'value', allow_duplicate=True),
     Output('y-range-slider', 'value', allow_duplicate=True)],
    [Input('preset-small', 'n_clicks'),
     Input('preset-default', 'n_clicks'),
     Input('preset-large', 'n_clicks')],
    prevent_initial_call=True
)
def update_range_presets(small_clicks, default_clicks, large_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [-3, 3], [-3, 3]
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'preset-small':
        return [-1, 1], [-1, 1]
    elif button_id == 'preset-default':
        return [-3, 3], [-3, 3]
    elif button_id == 'preset-large':
        return [-5, 5], [-5, 5]
    
    return [-3, 3], [-3, 3]

@callback(
    Output('custom-function', 'value'),
    Input('function-dropdown', 'value')
)
def update_custom_function(selected_func):
    if selected_func in FUNCTIONS:
        return FUNCTIONS[selected_func]
    return 'x**2 + y**2'

@callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('function-dropdown', 'value'),
     Input('custom-function', 'value'),
     Input('x-range-slider', 'value'),
     Input('y-range-slider', 'value'),
     Input('resolution', 'value'),
     Input('viz-options', 'value'),
     Input('colorscale', 'value')]
)
def update_visualization(active_tab, selected_func, custom_func, x_range, y_range, resolution, viz_options, colorscale):
    try:
        # Use custom function if "Custom" is selected
        func_str = custom_func if selected_func == "Custom" else FUNCTIONS[selected_func]
        
        # Validate ranges
        if not x_range or not y_range or len(x_range) != 2 or len(y_range) != 2:
            x_range, y_range = [-3, 3], [-3, 3]
        
        if x_range[0] >= x_range[1]:
            x_range = [x_range[0], x_range[0] + 0.1]
        if y_range[0] >= y_range[1]:
            y_range = [y_range[0], y_range[0] + 0.1]
        
        # Validate resolution
        if not resolution or resolution < 10:
            resolution = 50
        
        # Create meshgrid
        x = np.linspace(x_range[0], x_range[1], int(resolution))
        y = np.linspace(y_range[0], y_range[1], int(resolution))
        X, Y = np.meshgrid(x, y)
        Z = safe_eval_function(func_str, X, Y)
        
        # Check for invalid Z values
        if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            return html.Div([
                html.H4("⚠️ Function Error"),
                html.P(f"The function '{func_str}' produced invalid values (NaN or Inf)."),
                html.P("Try adjusting the range or using a different function."),
                html.P("Common issues: division by zero, log of negative numbers, etc.")
            ], style={'padding': 20, 'textAlign': 'center'})
        
        if active_tab == "surface":
            return create_surface_plot(X, Y, Z, func_str, viz_options, colorscale, x_range, y_range)
        elif active_tab == "contour":
            return create_contour_plot(X, Y, Z, func_str, viz_options, colorscale)
        elif active_tab == "gradient":
            return create_gradient_plot(X, Y, Z, func_str, colorscale)
    
    except Exception as e:
        return html.Div([
            html.H4("❌ Visualization Error"),
            html.P(f"Error: {str(e)}"),
            html.P("Please check your function syntax and try again."),
            html.P("Example valid functions: x**2 + y**2, np.sin(x)*np.cos(y), x**3 - 3*x*y**2")
        ], style={'padding': 20, 'textAlign': 'center'})

def create_surface_plot(X, Y, Z, func_str, viz_options, colorscale, x_range, y_range):
    fig = go.Figure()
    
    # Main surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        name=f'f(x,y) = {func_str}',
        opacity=0.8
    ))
    
    # Add contour lines if requested
    if 'contour' in viz_options:
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=Z,
            colorscale=colorscale,
            showscale=False,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            ),
            opacity=0.3
        ))
    
    # Add critical points if requested
    if 'critical' in viz_options:
        critical_points = find_critical_points(func_str, x_range, y_range)
        if critical_points:
            x_crit = [p[0] for p in critical_points]
            y_crit = [p[1] for p in critical_points]
            z_crit = [p[2] for p in critical_points]
            
            fig.add_trace(go.Scatter3d(
                x=x_crit, y=y_crit, z=z_crit,
                mode='markers',
                marker=dict(size=8, color='red'),
                name='Critical Points'
            ))
    
    fig.update_layout(
        title=f'3D Surface: f(x,y) = {func_str}',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x,y)'
        ),
        height=600
    )
    
    return dcc.Graph(figure=fig)

def create_contour_plot(X, Y, Z, func_str, viz_options, colorscale):
    fig = go.Figure()
    
    # Contour plot
    fig.add_trace(go.Contour(
        x=X[0], y=Y[:, 0], z=Z,
        colorscale=colorscale,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        )
    ))
    
    fig.update_layout(
        title=f'Contour Plot: f(x,y) = {func_str}',
        xaxis_title='x',
        yaxis_title='y',
        height=600
    )
    
    return dcc.Graph(figure=fig)

def create_gradient_plot(X, Y, Z, func_str, colorscale):
    # Compute gradient
    h = 0.01
    dx = (safe_eval_function(func_str, X + h, Y) - safe_eval_function(func_str, X - h, Y)) / (2 * h)
    dy = (safe_eval_function(func_str, X, Y + h) - safe_eval_function(func_str, X, Y - h)) / (2 * h)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    
    fig = go.Figure()
    
    # Gradient magnitude as heatmap
    fig.add_trace(go.Heatmap(
        x=X[0], y=Y[:, 0], z=magnitude,
        colorscale=colorscale,
        name='Gradient Magnitude'
    ))
    
    fig.update_layout(
        title=f'Gradient Magnitude: |∇f| where f(x,y) = {func_str}',
        xaxis_title='x',
        yaxis_title='y',
        height=600
    )
    
    return dcc.Graph(figure=fig)

@callback(
    Output('function-info', 'children'),
    [Input('custom-function', 'value'),
     Input('x-range-slider', 'value'),
     Input('y-range-slider', 'value')]
)
def update_function_info(func_str, x_range, y_range):
    try:
        # Validate inputs
        if not func_str or not x_range or not y_range:
            return html.P("Function analysis unavailable.")
        
        # Find critical points
        critical_points = find_critical_points(func_str, x_range, y_range)
        
        # Calculate domain size
        domain_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        
        info = [
            html.H4("📊 Function Analysis"),
            html.P(f"📝 Current function: f(x,y) = {func_str}"),
            html.P(f"📐 Domain: x ∈ [{x_range[0]:.2f}, {x_range[1]:.2f}], y ∈ [{y_range[0]:.2f}, {y_range[1]:.2f}]"),
            html.P(f"📏 Domain area: {domain_area:.2f} square units")
        ]
        
        if critical_points:
            info.append(html.P("🎯 Critical points found:"))
            for i, (x_crit, y_crit, z_crit) in enumerate(critical_points):
                info.append(html.P(f"  • Point {i+1}: ({x_crit:.3f}, {y_crit:.3f}) → f = {z_crit:.3f}", 
                                  style={'marginLeft': '20px'}))
        else:
            info.append(html.P("🔍 No critical points found in the current domain."))
        
        # Add some tips
        info.extend([
            html.Hr(),
            html.P("💡 Tips:", style={'fontWeight': 'bold'}),
            html.P("• Use number inputs or sliders to adjust viewing range", style={'fontSize': '12px'}),
            html.P("• Try functions like: x**2-y**2, np.sin(x)*np.cos(y), x**3-3*x*y**2", style={'fontSize': '12px'}),
            html.P("• Enable 'Show Critical Points' to see optimization results", style={'fontSize': '12px'})
        ])
        
        return info
    except Exception as e:
        return html.Div([
            html.P("⚠️ Function analysis error"),
            html.P(f"Error: {str(e)}", style={'fontSize': '12px', 'color': 'red'})
        ])

if __name__ == '__main__':
    print("🚀 Starting Multivariable Calculus Explorer...")
    print("📊 Interactive visualizations will open in your browser")
    print("🌐 URL: http://localhost:8050")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)