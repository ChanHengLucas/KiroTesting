"""
Chain rule demonstration for multivariable functions
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class ChainRuleDemo:
    """Demonstrate the multivariable chain rule"""
    
    def run(self):
        """Run chain rule demonstration"""
        print("\nChain Rule Demo")
        print("Composite function: f(g(t), h(t))")
        print("where f(x,y) = x² + y², g(t) = cos(t), h(t) = sin(t)")
        
        # Define symbolic variables
        t, x, y = sp.symbols('t x y')
        
        # Define functions
        f = x**2 + y**2  # f(x,y)
        g = sp.cos(t)    # g(t) = x(t)
        h = sp.sin(t)    # h(t) = y(t)
        
        # Composite function F(t) = f(g(t), h(t))
        F = f.subs([(x, g), (y, h)])
        
        print(f"\nF(t) = f(g(t), h(t)) = {F}")
        print(f"Simplified: F(t) = {sp.simplify(F)}")
        
        # Chain rule calculation
        df_dx = sp.diff(f, x)
        df_dy = sp.diff(f, y)
        dg_dt = sp.diff(g, t)
        dh_dt = sp.diff(h, t)
        
        # dF/dt using chain rule
        dF_dt_chain = df_dx.subs([(x, g), (y, h)]) * dg_dt + df_dy.subs([(x, g), (y, h)]) * dh_dt
        
        # Direct differentiation
        dF_dt_direct = sp.diff(F, t)
        
        print(f"\nUsing Chain Rule:")
        print(f"∂f/∂x = {df_dx}")
        print(f"∂f/∂y = {df_dy}")
        print(f"dg/dt = {dg_dt}")
        print(f"dh/dt = {dh_dt}")
        print(f"dF/dt = (∂f/∂x)(dg/dt) + (∂f/∂y)(dh/dt) = {sp.simplify(dF_dt_chain)}")
        
        print(f"\nDirect differentiation:")
        print(f"dF/dt = {sp.simplify(dF_dt_direct)}")
        
        print(f"\nVerification: Both methods give the same result: {sp.simplify(dF_dt_chain - dF_dt_direct) == 0}")
        
        # Numerical visualization
        t_vals = np.linspace(0, 4*np.pi, 1000)
        
        # Convert to numerical functions
        F_num = sp.lambdify(t, F, 'numpy')
        dF_dt_num = sp.lambdify(t, dF_dt_direct, 'numpy')
        g_num = sp.lambdify(t, g, 'numpy')
        h_num = sp.lambdify(t, h, 'numpy')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Parametric path in xy-plane
        axes[0,0].plot(g_num(t_vals), h_num(t_vals), 'b-', linewidth=2)
        axes[0,0].set_title('Parametric Path: (cos(t), sin(t))')
        axes[0,0].set_xlabel('x = cos(t)')
        axes[0,0].set_ylabel('y = sin(t)')
        axes[0,0].grid(True)
        axes[0,0].axis('equal')
        
        # Function values along path
        axes[0,1].plot(t_vals, F_num(t_vals), 'r-', linewidth=2)
        axes[0,1].set_title('F(t) = cos²(t) + sin²(t) = 1')
        axes[0,1].set_xlabel('t')
        axes[0,1].set_ylabel('F(t)')
        axes[0,1].grid(True)
        
        # Derivative
        axes[1,0].plot(t_vals, dF_dt_num(t_vals), 'g-', linewidth=2)
        axes[1,0].set_title("F'(t) = 0 (constant function)")
        axes[1,0].set_xlabel('t')
        axes[1,0].set_ylabel("F'(t)")
        axes[1,0].grid(True)
        
        # 3D visualization
        ax_3d = fig.add_subplot(224, projection='3d')
        
        # Create surface for f(x,y) = x² + y²
        x_surf = np.linspace(-1.5, 1.5, 50)
        y_surf = np.linspace(-1.5, 1.5, 50)
        X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
        Z_surf = X_surf**2 + Y_surf**2
        
        ax_3d.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.3, cmap='viridis')
        
        # Plot the parametric curve on the surface
        x_curve = g_num(t_vals)
        y_curve = h_num(t_vals)
        z_curve = F_num(t_vals)
        ax_3d.plot(x_curve, y_curve, z_curve, 'r-', linewidth=3, label='Path on surface')
        
        ax_3d.set_title('Path on Surface f(x,y) = x² + y²')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('f(x,y)')
        
        plt.tight_layout()
        plt.show()