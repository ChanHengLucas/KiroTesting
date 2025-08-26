"""
Core visualization classes for multivariable calculus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SurfacePlotter:
    """3D surface plotting for functions of two variables"""
    
    def __init__(self, resolution=50):
        self.resolution = resolution
    
    def plot_surface(self, func, title="3D Surface", x_range=(-3, 3), y_range=(-3, 3)):
        """Plot a 3D surface for f(x,y)"""
        x = np.linspace(x_range[0], x_range[1], self.resolution)
        y = np.linspace(y_range[0], y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}: f(x,y)')
        
        plt.colorbar(surface)
        plt.show()

class GradientVisualizer:
    """Visualize gradient vector fields"""
    
    def show_gradient_field(self, func=None):
        """Show gradient vectors for a function"""
        if func is None:
            func = lambda x, y: x**2 + y**2  # Default paraboloid
        
        x = np.linspace(-2, 2, 15)
        y = np.linspace(-2, 2, 15)
        X, Y = np.meshgrid(x, y)
        
        # Compute gradient numerically
        h = 0.01
        dx = (func(X + h, Y) - func(X - h, Y)) / (2 * h)
        dy = (func(X, Y + h) - func(X, Y - h)) / (2 * h)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Contour plot with gradient vectors
        Z = func(X, Y)
        contour = ax1.contour(X, Y, Z, levels=10)
        ax1.clabel(contour, inline=True, fontsize=8)
        ax1.quiver(X, Y, dx, dy, alpha=0.7, color='red')
        ax1.set_title('Gradient Vector Field')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True)
        
        # 3D surface
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        ax2.set_title('3D Surface')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('f(x,y)')
        
        plt.tight_layout()
        plt.show()

class ContourPlotter:
    """Create contour plots and level curves"""
    
    def show_contours(self, func=None):
        """Display contour plots with different levels"""
        if func is None:
            func = lambda x, y: x**2 + y**2 - x*y  # Default function
        
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Filled contour
        cs1 = axes[0].contourf(X, Y, Z, levels=20, cmap='viridis')
        axes[0].set_title('Filled Contour Plot')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(cs1, ax=axes[0])
        
        # Line contours
        cs2 = axes[1].contour(X, Y, Z, levels=15, colors='black')
        axes[1].clabel(cs2, inline=True, fontsize=8)
        axes[1].set_title('Level Curves')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].grid(True)
        
        # 3D contour projection
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.contour(X, Y, Z, levels=15, cmap='viridis')
        ax3.set_title('3D Contour Lines')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('f(x,y)')
        
        plt.tight_layout()
        plt.show()