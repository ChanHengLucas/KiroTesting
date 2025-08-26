#!/usr/bin/env python3
"""
Multivariable Calculus Visualization Tool
Interactive plotting for understanding calculus concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visualizations import SurfacePlotter, GradientVisualizer, ContourPlotter
import subprocess
import sys

def main():
    """Main menu for calculus visualizations"""
    print("Multivariable Calculus Visualization Tool")
    print("=" * 40)
    print("1. 3D Surface Plots (matplotlib)")
    print("2. Gradient Vector Fields (matplotlib)") 
    print("3. Contour Plots (matplotlib)")
    print("4. Interactive Examples (matplotlib)")
    print("5. Launch Web Interface (browser)")
    print("6. Exit")
    
    while True:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            surface_demo()
        elif choice == "2":
            gradient_demo()
        elif choice == "3":
            contour_demo()
        elif choice == "4":
            interactive_examples()
        elif choice == "5":
            launch_web_interface()
        elif choice == "6":
            print("Thanks for exploring calculus!")
            break
        else:
            print("Invalid choice. Please select 1-6.")

def surface_demo():
    """Demonstrate 3D surface plotting"""
    plotter = SurfacePlotter()
    
    # Common multivariable functions
    functions = {
        "Paraboloid": lambda x, y: x**2 + y**2,
        "Saddle Point": lambda x, y: x**2 - y**2,
        "Gaussian": lambda x, y: np.exp(-(x**2 + y**2)),
        "Ripple": lambda x, y: np.sin(np.sqrt(x**2 + y**2))
    }
    
    print("\nAvailable functions:")
    for i, name in enumerate(functions.keys(), 1):
        print(f"{i}. {name}")
    
    try:
        choice = int(input("Select function (1-4): ")) - 1
        func_name = list(functions.keys())[choice]
        func = functions[func_name]
        plotter.plot_surface(func, func_name)
    except (ValueError, IndexError):
        print("Invalid selection")

def gradient_demo():
    """Demonstrate gradient vector fields"""
    visualizer = GradientVisualizer()
    visualizer.show_gradient_field()

def contour_demo():
    """Demonstrate contour plots"""
    plotter = ContourPlotter()
    plotter.show_contours()

def interactive_examples():
    """Show interactive calculus examples"""
    print("\nInteractive Examples:")
    print("1. Optimization - Finding Critical Points")
    print("2. Partial Derivatives Visualization")
    print("3. Chain Rule Demonstration")
    
    choice = input("Select example (1-3): ").strip()
    
    if choice == "1":
        optimization_example()
    elif choice == "2":
        partial_derivatives_example()
    elif choice == "3":
        chain_rule_example()

def optimization_example():
    """Show optimization problem visualization"""
    from examples.optimization import OptimizationDemo
    demo = OptimizationDemo()
    demo.run()

def partial_derivatives_example():
    """Show partial derivatives visualization"""
    from examples.partial_derivatives import PartialDerivativeDemo
    demo = PartialDerivativeDemo()
    demo.run()

def chain_rule_example():
    """Show chain rule demonstration"""
    from examples.chain_rule import ChainRuleDemo
    demo = ChainRuleDemo()
    demo.run()

def launch_web_interface():
    """Launch the web-based interface"""
    print("\nLaunching web interface...")
    print("This will open in your browser at http://localhost:8050")
    print("Press Ctrl+C to stop the server when done.")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "web_app.py"])
    except KeyboardInterrupt:
        print("\nWeb server stopped.")
    except FileNotFoundError:
        print("Error: web_app.py not found. Make sure all files are in the same directory.")
    except Exception as e:
        print(f"Error launching web interface: {e}")

if __name__ == "__main__":
    main()