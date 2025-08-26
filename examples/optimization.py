"""
Optimization examples and critical point finding
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

class OptimizationDemo:
    """Demonstrate optimization concepts visually"""
    
    def run(self):
        """Run optimization demonstration"""
        print("\nOptimization Demo: Finding Critical Points")
        print("Function: f(x,y) = x² + y² - 2x - 4y + 5")
        
        # Define function and its gradient
        def f(vars):
            x, y = vars
            return x**2 + y**2 - 2*x - 4*y + 5
        
        def gradient(x, y):
            df_dx = 2*x - 2
            df_dy = 2*y - 4
            return df_dx, df_dy
        
        # Find minimum
        result = minimize(f, [0, 0], method='BFGS')
        min_point = result.x
        min_value = result.fun
        
        print(f"Critical point found at: ({min_point[0]:.3f}, {min_point[1]:.3f})")
        print(f"Function value at minimum: {min_value:.3f}")
        
        # Visualization
        x = np.linspace(-2, 4, 100)
        y = np.linspace(-1, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2 - 2*X - 4*Y + 5
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D surface
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax1.scatter([min_point[0]], [min_point[1]], [min_value], 
                   color='red', s=100, label='Minimum')
        ax1.set_title('3D Surface with Minimum')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        
        # Contour plot
        ax2 = fig.add_subplot(132)
        contour = ax2.contour(X, Y, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.plot(min_point[0], min_point[1], 'ro', markersize=10, label='Minimum')
        ax2.set_title('Contour Plot')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.legend()
        ax2.grid(True)
        
        # Gradient descent path
        ax3 = fig.add_subplot(133)
        # Simulate gradient descent
        path = self.gradient_descent_path(gradient, start=[3, 4], learning_rate=0.1)
        
        ax3.contour(X, Y, Z, levels=20, alpha=0.6)
        ax3.plot([p[0] for p in path], [p[1] for p in path], 'r.-', 
                linewidth=2, markersize=8, label='Gradient Descent Path')
        ax3.plot(min_point[0], min_point[1], 'go', markersize=10, label='Minimum')
        ax3.set_title('Gradient Descent')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def gradient_descent_path(self, gradient_func, start, learning_rate=0.1, max_iter=50):
        """Simulate gradient descent path"""
        path = [start]
        current = np.array(start)
        
        for _ in range(max_iter):
            grad = np.array(gradient_func(current[0], current[1]))
            current = current - learning_rate * grad
            path.append(current.copy())
            
            # Stop if gradient is small
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return path