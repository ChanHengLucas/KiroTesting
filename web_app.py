#!/usr/bin/env python3
"""
Web-based Multivariable Calculus Visualization
Interactive plotting in browser with customizable functions
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from scipy.optimize import minimize

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multivariable Calculus Explorer"

# Predefined functions
FUNCTIONS = {
    "Paraboloid": "x**2 + y**2",
    "Saddle Point": "x**2 - y**2", 
    "Gaussian": "np.exp(-(x**2 + y**2))",
    "Ripple": "np.sin(np.sqrt(x**2 + y**2))",
    "Monkey Saddle": "x**3 - 3*x*y**2",
    "Himmelblau": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
    "Rosenbrock": "100*(y - x**2)**2 + (1 - x)**2",
    "Custom": "x**2 + y**2"  # Default for custom
}

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Multivariable Calculus Explorer", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        # Controls Panel
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Function Controls", className="card-title"),
                    
                    html.Label("Select Function:"),
                    dcc.Dropdown(
                        id='function-dropdown',
                        options=[{'label': k, 'value': k} for k in FUNCTIONS.keys()],
                        value='Paraboloid',
                        className="mb-3"
                    ),
                    
                    html.Label("Custom Function f(x,y):"),
                    dcc.Input(
                        id='custom-function',
                        type='text',
                        value='x**2 + y**2',
                        placeholder='Enter function like: x**2 + y**2',
                        className="form-control mb-3"
                    ),
                    
                    html.Label("X Range:"),
                    dcc.RangeSlider(
                        id='x-range',
                        min=-5, max=5, step=0.5,
                        marks={i: str(i) for i in range(-5, 6)},
                        value=[-3, 3],
                        className="mb-3"
                    ),
                    
                    html.Label("Y Range:"),
                    dcc.RangeSlider(
                        id='y-range',
                        min=-5, max=5, step=0.5,
                        marks={i: str(i) for i in range(-5, 6)},
                        value=[-3, 3],
                        className="mb-3"
                    ),
                    
                    html.Label("Resolution:"),
                    dcc.Slider(
                        id='resolution',
                        min=20, max=100, step=10,
                        marks={i: str(i) for i in range(20, 101, 20)},
                        value=50,
                        className="mb-3"
                    ),
                    
                    html.Hr(),
                    
                    html.H5("Visualization Options"),
                    dbc.Checklist(
                        id='viz-options',
                        options=[
                            {'label': 'Show Gradient Vectors', 'value': 'gradient'},
                            {'label': 'Show Contour Lines', 'value': 'contour'},
                            {'label': 'Show Critical Points', 'value': 'critical'}
                        ],
                        value=['contour'],
                        className="mb-3"
                    ),
                    
                    html.Label("Color Scheme:"),
                    dcc.Dropdown(
                        id='colorscale',
                        options=[
                            {'label': 'Viridis', 'value': 'Viridis'},
                            {'label': 'Plasma', 'value': 'Plasma'},
                            {'label': 'Rainbow', 'value': 'Rainbow'},
                            {'label': 'Hot', 'value': 'Hot'},
                            {'label': 'Cool', 'value': 'Blues'}
                        ],
                        value='Viridis',
                        className="mb-3"
                    )
                ])
            ])
        ], width=3),
        
        # Main visualization area
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="3D Surface", tab_id="surface"),
                dbc.Tab(label="Contour Plot", tab_id="contour"),
                dbc.Tab(label="Gradient Field", tab_id="gradient"),
                dbc.Tab(label="Cross Sections", tab_id="sections")
            ], id="tabs", active_tab="surface"),
            
            html.Div(id="tab-content", className="mt-3")
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.Div(id="function-info", className="mt-3")
        ])
    ])
], fluid=True)

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
        for x0 in np.linspace(x_range[0], x_range[1], 5):
            for y0 in np.linspace(y_range[0], y_range[1], 5):
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
        
        return critical_points[:5]  # Limit to 5 points
    except:
        return []

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
    [Input('tabs', 'active_tab'),
     Input('function-dropdown', 'value'),
     Input('custom-function', 'value'),
     Input('x-range', 'value'),
     Input('y-range', 'value'),
     Input('resolution', 'value'),
     Input('viz-options', 'value'),
     Input('colorscale', 'value')]
)
def update_visualization(active_tab, selected_func, custom_func, x_range, y_range, resolution, viz_options, colorscale):
    # Use custom function if "Custom" is selected
    func_str = custom_func if selected_func == "Custom" else FUNCTIONS[selected_func]
    
    # Create meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = safe_eval_function(func_str, X, Y)
    
    if active_tab == "surface":
        return create_surface_plot(X, Y, Z, func_str, viz_options, colorscale, x_range, y_range)
    elif active_tab == "contour":
        return create_contour_plot(X, Y, Z, func_str, viz_options, colorscale)
    elif active_tab == "gradient":
        return create_gradient_plot(X, Y, Z, func_str, colorscale)
    elif active_tab == "sections":
        return create_sections_plot(x, y, func_str, colorscale)

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
    
    # Add gradient vectors if requested
    if 'gradient' in viz_options:
        # Subsample for gradient vectors
        step = max(1, len(X) // 15)
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
        dx_norm = dx / (magnitude + 1e-8) * 0.2
        dy_norm = dy / (magnitude + 1e-8) * 0.2
        
        for i in range(X_sub.shape[0]):
            for j in range(X_sub.shape[1]):
                fig.add_annotation(
                    x=X_sub[i,j], y=Y_sub[i,j],
                    ax=X_sub[i,j] + dx_norm[i,j], ay=Y_sub[i,j] + dy_norm[i,j],
                    xref='x', yref='y', axref='x', ayref='y',
                    arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='red'
                )
    
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

def create_sections_plot(x, y, func_str, colorscale):
    # Create cross-sections at x=0 and y=0
    x_section = safe_eval_function(func_str, x, 0)
    y_section = safe_eval_function(func_str, 0, y)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=x_section,
        mode='lines',
        name='f(x, 0)',
        line=dict(color='red', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=y, y=y_section,
        mode='lines',
        name='f(0, y)',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title=f'Cross-sections: f(x,y) = {func_str}',
        xaxis_title='x or y',
        yaxis_title='f(x,0) or f(0,y)',
        height=600
    )
    
    return dcc.Graph(figure=fig)

@callback(
    Output('function-info', 'children'),
    [Input('custom-function', 'value'),
     Input('x-range', 'value'),
     Input('y-range', 'value')]
)
def update_function_info(func_str, x_range, y_range):
    try:
        # Find critical points
        critical_points = find_critical_points(func_str, x_range, y_range)
        
        info = [
            html.H5("Function Analysis"),
            html.P(f"Current function: f(x,y) = {func_str}"),
            html.P(f"Domain: x ∈ [{x_range[0]}, {x_range[1]}], y ∈ [{y_range[0]}, {y_range[1]}]")
        ]
        
        if critical_points:
            info.append(html.P("Critical points found:"))
            for i, (x_crit, y_crit, z_crit) in enumerate(critical_points):
                info.append(html.P(f"  Point {i+1}: ({x_crit:.3f}, {y_crit:.3f}) → f = {z_crit:.3f}"))
        else:
            info.append(html.P("No critical points found in the current domain."))
        
        return info
    except:
        return html.P("Function analysis unavailable.")

if __name__ == '__main__':
    print("Starting Multivariable Calculus Explorer...")
    print("Open your browser to: http://localhost:8050")
    app.run_server(debug=True, host='0.0.0.0', port=8050)