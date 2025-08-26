"""
Partial derivatives visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PartialDerivativeDemo:
    """Visualize partial derivatives geometrically"""
    
    def run(self):
        """Run partial derivatives demonstration"""
        print("\nPartial Derivatives Demo")
        print("Function: f(x,y) = x²y + xy²")
        
        # Define function
        def f(x, y):
            return x**2 * y + x * y**2
        
        # Analytical partial derivatives
        def df_dx(x, y):
            return 2*x*y + y**2
        
        def df_dy(x, y):
            return x**2 + 2*x*y
        
        # Create meshgrid
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # Point of interest
        x0, y0 = 1, 1
        z0 = f(x0, y0)
        
        fig = plt.figure(figsize=(18, 6))
        
        # 3D surface with cross-sections
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        
        # Show cross-sections
        y_fixed = np.full_like(x, y0)
        z_x_section = f(x, y_fixed)
        ax1.plot(x, y_fixed, z_x_section, 'r-', linewidth=3, label='∂f/∂x section')
        
        x_fixed = np.full_like(y, x0)
        z_y_section = f(x_fixed, y)
        ax1.plot(x_fixed, y, z_y_section, 'b-', linewidth=3, label='∂f/∂y section')
        
        ax1.scatter([x0], [y0], [z0], color='black', s=100)
        ax1.set_title('3D Surface with Cross-sections')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        ax1.legend()
        
        # Partial derivative with respect to x
        ax2 = fig.add_subplot(132)
        ax2.plot(x, f(x, y0), 'r-', linewidth=2, label=f'f(x, {y0})')
        
        # Show tangent line at point
        slope_x = df_dx(x0, y0)
        tangent_x = slope_x * (x - x0) + f(x0, y0)
        ax2.plot(x, tangent_x, 'r--', alpha=0.7, label=f'Tangent (slope={slope_x:.2f})')
        ax2.plot(x0, f(x0, y0), 'ro', markersize=8)
        
        ax2.set_title(f'∂f/∂x at y = {y0}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x,y)')
        ax2.grid(True)
        ax2.legend()
        
        # Partial derivative with respect to y
        ax3 = fig.add_subplot(133)
        ax3.plot(y, f(x0, y), 'b-', linewidth=2, label=f'f({x0}, y)')
        
        # Show tangent line at point
        slope_y = df_dy(x0, y0)
        tangent_y = slope_y * (y - y0) + f(x0, y0)
        ax3.plot(y, tangent_y, 'b--', alpha=0.7, label=f'Tangent (slope={slope_y:.2f})')
        ax3.plot(y0, f(x0, y0), 'bo', markersize=8)
        
        ax3.set_title(f'∂f/∂y at x = {x0}')
        ax3.set_xlabel('y')
        ax3.set_ylabel('f(x,y)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nAt point ({x0}, {y0}):")
        print(f"∂f/∂x = {df_dx(x0, y0):.3f}")
        print(f"∂f/∂y = {df_dy(x0, y0):.3f}")
        print(f"Gradient = ({df_dx(x0, y0):.3f}, {df_dy(x0, y0):.3f})")