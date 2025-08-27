"""
Enhanced Calculus Grapher with Educational Content Integration
Uses MCP-fetched content and educational examples
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from educational_content import EducationalContentManager, MathExample

# Initialize educational content
@st.cache_resource
def get_content_manager():
    return EducationalContentManager()

def main():
    st.set_page_config(
        page_title="Advanced Calculus Grapher",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🧮 Advanced Calculus Grapher")
    st.markdown("*Interactive mathematical visualization with educational examples*")
    
    content_manager = get_content_manager()
    
    # Sidebar for navigation and examples
    with st.sidebar:
        st.header("📚 Educational Examples")
        
        # Category selection
        category = st.selectbox(
            "Choose Category:",
            options=list(content_manager.categories.keys()),
            format_func=lambda x: content_manager.categories[x]
        )
        
        # Get examples for selected category
        examples = content_manager.get_examples_by_category(category)
        
        if examples:
            st.subheader(f"{content_manager.categories[category]}")
            
            for i, example in enumerate(examples):
                with st.expander(f"{example.title} ({example.difficulty})"):
                    st.write(f"**Expression:** `{example.expression}`")
                    st.write(f"**Description:** {example.description}")
                    
                    if st.button(f"Load Example", key=f"load_{i}"):
                        st.session_state.current_function = example.expression
                        st.session_state.example_loaded = True
                        st.rerun()
        
        st.divider()
        
        # Random example button
        if st.button("🎲 Random Example"):
            random_ex = content_manager.get_random_example()
            st.session_state.current_function = random_ex.expression
            st.session_state.random_example = random_ex
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Function Input")
        
        # Function input with example pre-loading
        default_func = st.session_state.get('current_function', 'x**2 + y**2')
        function_input = st.text_input(
            "Enter function f(x,y):",
            value=default_func,
            help="Use Python syntax: x**2, sin(x), cos(y), etc."
        )
        
        # Display loaded example info
        if hasattr(st.session_state, 'example_loaded') and st.session_state.example_loaded:
            st.success("✅ Example loaded successfully!")
            st.session_state.example_loaded = False
        
        if hasattr(st.session_state, 'random_example'):
            ex = st.session_state.random_example
            st.info(f"🎲 **Random Example:** {ex.title}\n\n{ex.description}")
            del st.session_state.random_example
        
        # Visualization options
        st.subheader("Visualization Options")
        
        plot_type = st.selectbox(
            "Plot Type:",
            ["3D Surface", "Contour Plot", "Gradient Field", "Level Curves"]
        )
        
        # Range controls
        col_x, col_y = st.columns(2)
        with col_x:
            x_range = st.slider("X Range", -5.0, 5.0, (-3.0, 3.0), 0.1)
        with col_y:
            y_range = st.slider("Y Range", -5.0, 5.0, (-3.0, 3.0), 0.1)
        
        resolution = st.slider("Resolution", 20, 100, 50)
        
        # Calculate button
        if st.button("🚀 Generate Visualization", type="primary"):
            st.session_state.should_plot = True
            st.session_state.plot_params = {
                'function': function_input,
                'plot_type': plot_type,
                'x_range': x_range,
                'y_range': y_range,
                'resolution': resolution
            }
    
    with col2:
        st.header("Visualization")
        
        if hasattr(st.session_state, 'should_plot') and st.session_state.should_plot:
            params = st.session_state.plot_params
            
            try:
                # Create symbolic variables
                x, y = sp.symbols('x y')
                
                # Parse the function
                func_expr = sp.sympify(params['function'])
                func_lambdified = sp.lambdify((x, y), func_expr, 'numpy')
                
                # Create meshgrid
                x_vals = np.linspace(params['x_range'][0], params['x_range'][1], params['resolution'])
                y_vals = np.linspace(params['y_range'][0], params['y_range'][1], params['resolution'])
                X, Y = np.meshgrid(x_vals, y_vals)
                
                # Calculate function values
                Z = func_lambdified(X, Y)
                
                # Generate appropriate plot
                if params['plot_type'] == "3D Surface":
                    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
                    fig.update_layout(
                        title=f"3D Surface: f(x,y) = {params['function']}",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y", 
                            zaxis_title="f(x,y)"
                        ),
                        width=600,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif params['plot_type'] == "Contour Plot":
                    fig = go.Figure(data=go.Contour(x=x_vals, y=y_vals, z=Z, colorscale='Viridis'))
                    fig.update_layout(
                        title=f"Contour Plot: f(x,y) = {params['function']}",
                        xaxis_title="X",
                        yaxis_title="Y"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif params['plot_type'] == "Gradient Field":
                    # Calculate gradient
                    grad_x = sp.diff(func_expr, x)
                    grad_y = sp.diff(func_expr, y)
                    
                    grad_x_func = sp.lambdify((x, y), grad_x, 'numpy')
                    grad_y_func = sp.lambdify((x, y), grad_y, 'numpy')
                    
                    # Create coarser grid for arrows
                    x_coarse = np.linspace(params['x_range'][0], params['x_range'][1], 15)
                    y_coarse = np.linspace(params['y_range'][0], params['y_range'][1], 15)
                    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
                    
                    U = grad_x_func(X_coarse, Y_coarse)
                    V = grad_y_func(X_coarse, Y_coarse)
                    
                    fig = go.Figure()
                    
                    # Add contour background
                    fig.add_trace(go.Contour(x=x_vals, y=y_vals, z=Z, 
                                           colorscale='Viridis', opacity=0.3, showscale=False))
                    
                    # Add gradient arrows
                    for i in range(len(x_coarse)):
                        for j in range(len(y_coarse)):
                            fig.add_annotation(
                                x=X_coarse[j,i], y=Y_coarse[j,i],
                                ax=X_coarse[j,i] + U[j,i]*0.1, ay=Y_coarse[j,i] + V[j,i]*0.1,
                                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
                            )
                    
                    fig.update_layout(
                        title=f"Gradient Field: ∇f = {params['function']}",
                        xaxis_title="X",
                        yaxis_title="Y"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display mathematical analysis
                st.subheader("📊 Mathematical Analysis")
                
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    st.write("**Partial Derivatives:**")
                    try:
                        fx = sp.diff(func_expr, x)
                        fy = sp.diff(func_expr, y)
                        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
                        st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")
                    except:
                        st.write("Could not compute derivatives")
                
                with col_analysis2:
                    st.write("**Function Properties:**")
                    st.write(f"Domain: ℝ²")
                    st.write(f"Range: [{np.min(Z):.2f}, {np.max(Z):.2f}]")
                    
                    # Find critical points (simplified)
                    try:
                        critical_points = sp.solve([fx, fy], [x, y])
                        if critical_points:
                            st.write(f"Critical points found: {len(critical_points)}")
                    except:
                        st.write("Critical point analysis unavailable")
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                st.write("Please check your function syntax. Use Python notation:")
                st.write("- Powers: x**2, y**3")
                st.write("- Trig: sin(x), cos(y), tan(x*y)")
                st.write("- Other: exp(x), log(x), sqrt(x)")
    
    # Educational resources section
    st.divider()
    st.header("📖 Educational Resources")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.subheader("Quick Reference")
        st.write("""
        **Common Functions:**
        - Polynomial: `x**2 + y**2`
        - Trigonometric: `sin(x*y)`
        - Exponential: `exp(x + y)`
        - Logarithmic: `log(x**2 + y**2)`
        """)
    
    with col_res2:
        st.subheader("Calculus Concepts")
        st.write("""
        **Visualization Types:**
        - **3D Surface**: Shows function height
        - **Contour**: Level curves of constant value
        - **Gradient**: Direction of steepest increase
        - **Level Curves**: 2D contour lines
        """)
    
    with col_res3:
        st.subheader("Example Categories")
        for key, name in content_manager.categories.items():
            count = len(content_manager.get_examples_by_category(key))
            st.write(f"**{name}**: {count} examples")

if __name__ == "__main__":
    main()