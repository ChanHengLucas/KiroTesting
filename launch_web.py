#!/usr/bin/env python3
"""
Quick launcher for the web interface
"""

if __name__ == "__main__":
    print("🚀 Launching Multivariable Calculus Explorer...")
    print("📊 Interactive visualizations will open in your browser")
    print("🌐 URL: http://localhost:8050")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        from simple_web_app import app
        app.run_server(debug=False, host='127.0.0.1', port=8050)
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")