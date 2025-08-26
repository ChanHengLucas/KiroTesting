#!/usr/bin/env python3
"""
Launch the Advanced Multivariable Calculus Explorer
"""

if __name__ == "__main__":
    print("🧮 Launching Advanced Multivariable Calculus Explorer...")
    print("🎓 Comprehensive educational platform with:")
    print("   • Enhanced 3D visualizations with hover coordinates")
    print("   • Symbolic differentiation and integration")
    print("   • Advanced critical point analysis")
    print("   • Vector field visualizations")
    print("   • Cross-section analysis")
    print("   • Extensive function library")
    print("   • Educational analysis tools")
    print()
    print("🌐 Opening at: http://localhost:8050")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        from advanced_calculus_app import app
        app.run_server(debug=False, host='127.0.0.1', port=8050)
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Run: pip install -r requirements.txt")
        print("   or: conda install pandas sympy")
    except Exception as e:
        print(f"❌ Error: {e}")