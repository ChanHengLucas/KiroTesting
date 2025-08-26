#!/usr/bin/env python3
"""
Demo functions and educational examples for the Advanced Calculus Explorer
"""

DEMO_FUNCTIONS = {
    "🎓 Educational Examples": {
        "Simple Paraboloid": {
            "function": "x**2 + y**2",
            "description": "Classic bowl shape - perfect for understanding basic 3D surfaces",
            "range": [-3, 3],
            "features": ["One global minimum at origin", "Circular level curves", "Simple gradient field"]
        },
        
        "Saddle Point": {
            "function": "x**2 - y**2",
            "description": "Horse saddle shape - demonstrates critical points that aren't extrema",
            "range": [-3, 3],
            "features": ["Saddle point at origin", "Hyperbolic level curves", "Classic optimization example"]
        },
        
        "Monkey Saddle": {
            "function": "x**3 - 3*x*y**2",
            "description": "Three-way saddle point - advanced critical point analysis",
            "range": [-2, 2],
            "features": ["Unique 3-way saddle", "Complex gradient behavior", "Higher-order derivatives"]
        }
    },
    
    "🌊 Wave Functions": {
        "Standing Wave": {
            "function": "np.sin(x) * np.sin(y)",
            "description": "Product of sine functions - shows wave interference patterns",
            "range": [-6, 6],
            "features": ["Periodic in both directions", "Grid of critical points", "Beautiful contour patterns"]
        },
        
        "Ripple Effect": {
            "function": "np.sin(np.sqrt(x**2 + y**2))",
            "description": "Circular waves emanating from origin",
            "range": [-10, 10],
            "features": ["Radial symmetry", "Concentric circular contours", "Decreasing amplitude"]
        },
        
        "Twisted Surface": {
            "function": "np.sin(x*y)",
            "description": "Demonstrates interaction between variables",
            "range": [-4, 4],
            "features": ["Variable interaction", "Diagonal symmetry", "Complex critical point structure"]
        }
    },
    
    "🔥 Optimization Classics": {
        "Himmelblau's Function": {
            "function": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
            "description": "Famous test function with 4 global minima",
            "range": [-5, 5],
            "features": ["Four global minima", "Complex landscape", "Real-world optimization benchmark"]
        },
        
        "Rosenbrock Function": {
            "function": "100*(y - x**2)**2 + (1 - x)**2",
            "description": "Banana function - challenging optimization problem",
            "range": [-2, 2],
            "features": ["Narrow curved valley", "Global minimum at (1,1)", "Difficult for algorithms"]
        },
        
        "Ackley Function": {
            "function": "-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20",
            "description": "Highly multimodal function with many local minima",
            "range": [-5, 5],
            "features": ["Many local minima", "Global minimum at origin", "Exponential and trigonometric components"]
        }
    },
    
    "📊 Statistical Surfaces": {
        "2D Gaussian": {
            "function": "np.exp(-(x**2 + y**2))",
            "description": "Bell curve in 2D - fundamental in statistics",
            "range": [-3, 3],
            "features": ["Maximum at origin", "Exponential decay", "Circular symmetry"]
        },
        
        "Mexican Hat": {
            "function": "(1 - (x**2 + y**2)) * np.exp(-(x**2 + y**2)/2)",
            "description": "Wavelet function used in signal processing",
            "range": [-4, 4],
            "features": ["Central peak with surrounding trough", "Used in wavelet analysis", "Complex critical point structure"]
        },
        
        "Bivariate Normal": {
            "function": "np.exp(-0.5*((x-1)**2 + (y+0.5)**2))",
            "description": "Shifted Gaussian distribution",
            "range": [-3, 5],
            "features": ["Maximum at (1, -0.5)", "Statistical interpretation", "Probability density function"]
        }
    }
}

def print_demo_guide():
    """Print a guide to using the demo functions"""
    print("🎓 ADVANCED CALCULUS EXPLORER - DEMO GUIDE")
    print("=" * 60)
    print()
    
    for category, functions in DEMO_FUNCTIONS.items():
        print(f"{category}")
        print("-" * 40)
        
        for name, details in functions.items():
            print(f"📝 {name}")
            print(f"   Function: {details['function']}")
            print(f"   Range: {details['range']}")
            print(f"   Description: {details['description']}")
            print(f"   Key Features:")
            for feature in details['features']:
                print(f"     • {feature}")
            print()
    
    print("🚀 GETTING STARTED:")
    print("1. Run: python launch_advanced.py")
    print("2. Open browser to http://localhost:8050")
    print("3. Try the functions above in different visualization modes")
    print("4. Experiment with different ranges and resolution settings")
    print("5. Use the Analysis tab for detailed mathematical insights")
    print()
    print("💡 TIPS:")
    print("• Start with simple functions like 'x**2 + y**2'")
    print("• Use the hover feature to see exact coordinates")
    print("• Try different color schemes for better visualization")
    print("• Enable gradient vectors to see flow directions")
    print("• Use cross-sections to understand function behavior")
    print("• Check the Analysis tab for critical points and derivatives")

if __name__ == "__main__":
    print_demo_guide()