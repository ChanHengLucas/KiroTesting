"""
Performance optimization utilities for the multivariable calculus grapher
Focuses on computational efficiency and loading time improvements
"""

import numpy as np
import time
from functools import wraps, lru_cache
from typing import Callable, Tuple, Any, Dict
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

class PerformanceMonitor:
    """Monitor and optimize performance for calculus computations"""
    
    def __init__(self):
        self.computation_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                if func_name not in self.computation_times:
                    self.computation_times[func_name] = []
                self.computation_times[func_name].append(execution_time)
                
                # Log slow operations
                if execution_time > 1.0:
                    print(f"⚠️ Slow operation: {func_name} took {execution_time:.2f}s")
                
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            "average_times": {},
            "cache_efficiency": 0,
            "slow_operations": []
        }
        
        for func_name, times in self.computation_times.items():
            avg_time = sum(times) / len(times)
            report["average_times"][func_name] = avg_time
            
            if avg_time > 0.5:
                report["slow_operations"].append({
                    "function": func_name,
                    "average_time": avg_time,
                    "call_count": len(times)
                })
        
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations > 0:
            report["cache_efficiency"] = self.cache_hits / total_cache_operations
        
        return report

# Global performance monitor
perf_monitor = PerformanceMonitor()

class OptimizedMeshGenerator:
    """Generate optimized meshgrids for 3D plotting with adaptive resolution"""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def create_adaptive_mesh(x_range: Tuple[float, float], 
                           y_range: Tuple[float, float], 
                           base_resolution: int = 50,
                           complexity_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create adaptive mesh based on function complexity"""
        # Adjust resolution based on complexity and range
        range_factor = max(abs(x_range[1] - x_range[0]), abs(y_range[1] - y_range[0]))
        adjusted_resolution = max(20, min(200, int(base_resolution * complexity_factor * (range_factor / 6))))
        
        x = np.linspace(x_range[0], x_range[1], adjusted_resolution)
        y = np.linspace(y_range[0], y_range[1], adjusted_resolution)
        
        return np.meshgrid(x, y)
    
    @staticmethod
    def create_progressive_mesh(x_range: Tuple[float, float], 
                              y_range: Tuple[float, float],
                              levels: int = 3) -> list:
        """Create multiple resolution levels for progressive loading"""
        meshes = []
        base_resolution = 20
        
        for level in range(levels):
            resolution = base_resolution * (2 ** level)
            x = np.linspace(x_range[0], x_range[1], min(resolution, 100))
            y = np.linspace(y_range[0], y_range[1], min(resolution, 100))
            meshes.append(np.meshgrid(x, y))
        
        return meshes

class SafeFunctionEvaluator:
    """Safe and optimized function evaluation with error handling"""
    
    def __init__(self):
        self.evaluation_cache = {}
        self.error_count = 0
    
    @perf_monitor.time_function("function_evaluation")
    def safe_eval_with_timeout(self, func_str: str, x: np.ndarray, y: np.ndarray, 
                              timeout: float = 2.0) -> np.ndarray:
        """Evaluate function with timeout and error handling"""
        # Create cache key
        cache_key = (func_str, x.shape, tuple(x.flat[:5]), tuple(y.flat[:5]))
        
        if cache_key in self.evaluation_cache:
            perf_monitor.cache_hits += 1
            return self.evaluation_cache[cache_key]
        
        perf_monitor.cache_misses += 1
        
        try:
            # Use ThreadPoolExecutor for timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._evaluate_function, func_str, x, y)
                result = future.result(timeout=timeout)
                
                # Cache successful results
                if len(self.evaluation_cache) < 100:  # Limit cache size
                    self.evaluation_cache[cache_key] = result
                
                return result
                
        except Exception as e:
            self.error_count += 1
            print(f"⚠️ Function evaluation error: {str(e)}")
            # Return fallback function
            return self._fallback_function(x, y)
    
    def _evaluate_function(self, func_str: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Internal function evaluation"""
        # Comprehensive namespace for mathematical functions
        namespace = {
            'x': x, 'y': y, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs, 'sign': np.sign,
            'pi': np.pi, 'e': np.e,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
            'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
            'min': np.minimum, 'max': np.maximum
        }
        
        # Suppress numpy warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = eval(func_str, {"__builtins__": {}}, namespace)
        
        # Handle scalar results
        if np.isscalar(result):
            result = np.full_like(x, result)
        
        # Check for invalid values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            # Replace invalid values with interpolated values
            result = self._clean_invalid_values(result)
        
        return result
    
    def _fallback_function(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fallback to simple paraboloid when evaluation fails"""
        return x**2 + y**2
    
    def _clean_invalid_values(self, z: np.ndarray) -> np.ndarray:
        """Clean NaN and Inf values from function output"""
        # Replace NaN and Inf with interpolated values
        mask = np.isfinite(z)
        if np.any(mask):
            # Use median of finite values as replacement
            replacement_value = np.median(z[mask])
            z[~mask] = replacement_value
        else:
            # If all values are invalid, return zeros
            z = np.zeros_like(z)
        
        return z

class CriticalPointOptimizer:
    """Optimized critical point finding with smart algorithms"""
    
    def __init__(self):
        self.evaluator = SafeFunctionEvaluator()
    
    @perf_monitor.time_function("critical_point_finding")
    def find_critical_points_fast(self, func_str: str, 
                                 x_range: Tuple[float, float], 
                                 y_range: Tuple[float, float],
                                 max_points: int = 5) -> list:
        """Fast critical point finding with optimized algorithms"""
        from scipy.optimize import minimize
        
        def objective(vars):
            x, y = vars
            return self.evaluator.safe_eval_with_timeout(func_str, 
                                                       np.array([x]), 
                                                       np.array([y]))[0]
        
        def gradient_norm(vars):
            """Objective function for finding critical points (where gradient = 0)"""
            x, y = vars
            h = 1e-6
            
            # Compute numerical gradient
            fx_plus = objective([x + h, y])
            fx_minus = objective([x - h, y])
            fy_plus = objective([x, y + h])
            fy_minus = objective([x, y - h])
            
            dx = (fx_plus - fx_minus) / (2 * h)
            dy = (fy_plus - fy_minus) / (2 * h)
            
            return dx**2 + dy**2
        
        critical_points = []
        
        # Smart initial guesses
        initial_guesses = [
            (0, 0),  # Origin
            (x_range[0], y_range[0]),  # Corners
            (x_range[1], y_range[1]),
            (x_range[0], y_range[1]),
            (x_range[1], y_range[0]),
            ((x_range[0] + x_range[1])/2, (y_range[0] + y_range[1])/2)  # Center
        ]
        
        # Add some random points for better coverage
        import random
        for _ in range(4):
            x_rand = random.uniform(x_range[0], x_range[1])
            y_rand = random.uniform(y_range[0], y_range[1])
            initial_guesses.append((x_rand, y_rand))
        
        for x0, y0 in initial_guesses:
            try:
                # Use multiple methods for robustness
                methods = ['BFGS', 'L-BFGS-B', 'Powell']
                
                for method in methods:
                    try:
                        result = minimize(gradient_norm, [x0, y0], 
                                        method=method,
                                        options={'maxiter': 50})
                        
                        if result.success and result.fun < 1e-6:
                            x_crit, y_crit = result.x
                            
                            # Check if point is in domain
                            if (x_range[0] <= x_crit <= x_range[1] and 
                                y_range[0] <= y_crit <= y_range[1]):
                                
                                # Check if point is new
                                is_new = True
                                for existing in critical_points:
                                    if (abs(existing['x'] - x_crit) < 1e-3 and 
                                        abs(existing['y'] - y_crit) < 1e-3):
                                        is_new = False
                                        break
                                
                                if is_new:
                                    z_crit = objective([x_crit, y_crit])
                                    
                                    # Classify critical point using Hessian
                                    point_type = self._classify_critical_point(
                                        objective, x_crit, y_crit)
                                    
                                    critical_points.append({
                                        'x': x_crit,
                                        'y': y_crit,
                                        'z': z_crit,
                                        'type': point_type
                                    })
                                    
                                    if len(critical_points) >= max_points:
                                        return critical_points
                            break  # Success with this method
                    except:
                        continue  # Try next method
            except:
                continue  # Try next initial guess
        
        return critical_points
    
    def _classify_critical_point(self, func, x: float, y: float) -> str:
        """Classify critical point using Hessian determinant"""
        h = 1e-5
        
        try:
            # Compute second derivatives (Hessian)
            fxx = (func([x + h, y]) - 2*func([x, y]) + func([x - h, y])) / h**2
            fyy = (func([x, y + h]) - 2*func([x, y]) + func([x, y - h])) / h**2
            fxy = (func([x + h, y + h]) - func([x + h, y - h]) - 
                   func([x - h, y + h]) + func([x - h, y - h])) / (4*h**2)
            
            discriminant = fxx * fyy - fxy**2
            
            if discriminant > 0:
                return "Local Minimum" if fxx > 0 else "Local Maximum"
            elif discriminant < 0:
                return "Saddle Point"
            else:
                return "Inconclusive"
        except:
            return "Unknown"

# Global instances for easy access
mesh_generator = OptimizedMeshGenerator()
function_evaluator = SafeFunctionEvaluator()
critical_point_optimizer = CriticalPointOptimizer()

def get_performance_summary() -> str:
    """Get a formatted performance summary"""
    report = perf_monitor.get_performance_report()
    
    summary = "🚀 Performance Summary:\n"
    summary += f"Cache Efficiency: {report['cache_efficiency']:.1%}\n"
    
    if report['slow_operations']:
        summary += "\n⚠️ Slow Operations:\n"
        for op in report['slow_operations']:
            summary += f"  • {op['function']}: {op['average_time']:.2f}s avg ({op['call_count']} calls)\n"
    else:
        summary += "\n✅ All operations running efficiently\n"
    
    return summary