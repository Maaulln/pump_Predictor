"""
Enhanced error handling and recovery system
"""
import functools
import traceback
import time
from typing import Any, Callable, Optional, Dict
from enum import Enum
import asyncio
from contextlib import contextmanager

from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PumpPredictorError(Exception):
    """Base exception for pump predictor application"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[Dict] = None):
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()
        super().__init__(message)

class DataValidationError(PumpPredictorError):
    """Data validation specific errors"""
    def __init__(self, message: str, validation_errors: list = None, **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.validation_errors = validation_errors or []

class ModelError(PumpPredictorError):
    """Model-related errors"""
    def __init__(self, message: str, model_type: str = None, **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.model_type = model_type

class APIError(PumpPredictorError):
    """API-related errors"""
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.status_code = status_code

class CircuitBreakerError(PumpPredictorError):
    """Circuit breaker specific errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)

class CircuitBreaker:
    """Circuit breaker pattern implementation for resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {func.__name__} moving to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker for {func.__name__} closed successfully")
                return result
            
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker for {func.__name__} opened after {self.failure_count} failures")
                
                raise e
        
        return wrapper

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        break
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {str(e)}")
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

@contextmanager
def error_context(operation_name: str, **context):
    """Context manager for enhanced error reporting"""
    start_time = time.time()
    try:
        logger.info(f"Starting operation: {operation_name}")
        yield
        duration = time.time() - start_time
        logger.info(f"Operation {operation_name} completed successfully in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        error_info = {
            "operation": operation_name,
            "duration": duration,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            **context
        }
        logger.error(f"Operation {operation_name} failed after {duration:.2f}s", extra=error_info)
        raise

def handle_errors(
    default_return: Any = None,
    reraise: bool = True,
    log_level: str = "error"
):
    """Decorator for standardized error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Truncate for logging
                    "kwargs": str(kwargs)[:200],
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                # Log based on level
                if log_level == "error":
                    logger.error(f"Error in {func.__name__}: {str(e)}", extra=error_info)
                elif log_level == "warning":
                    logger.warning(f"Warning in {func.__name__}: {str(e)}", extra=error_info)
                elif log_level == "info":
                    logger.info(f"Info in {func.__name__}: {str(e)}", extra=error_info)
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator

class ErrorRecovery:
    """Error recovery strategies"""
    
    @staticmethod
    def fallback_prediction(input_data: Any) -> Dict[str, Any]:
        """Fallback prediction when main model fails"""
        logger.warning("Using fallback prediction due to model error")
        
        # Simple rule-based fallback
        # In practice, this could be a simpler model or business rules
        return {
            "needs_maintenance": False,
            "confidence": 0.5,
            "risk_level": "MEDIUM",
            "model_type": "fallback_rules",
            "note": "Fallback prediction used due to system error"
        }
    
    @staticmethod
    def graceful_degradation(error: Exception, operation: str) -> Dict[str, Any]:
        """Provide graceful degradation for failed operations"""
        logger.warning(f"Graceful degradation for {operation}: {str(error)}")
        
        return {
            "status": "degraded",
            "message": f"Service partially available. {operation} failed but system is operational.",
            "error_type": type(error).__name__,
            "fallback_available": True
        }

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
    
    def register_check(self, name: str, check_func: Callable, interval: int = 60):
        """Register a health check"""
        self.checks[name] = {
            "func": check_func,
            "interval": interval,
            "last_result": None,
            "last_error": None
        }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {"status": "unknown", "error": f"Check '{name}' not registered"}
        
        check = self.checks[name]
        
        # Check if we need to run the check
        last_check = self.last_check_time.get(name, 0)
        if time.time() - last_check < check["interval"]:
            return check["last_result"] or {"status": "pending"}
        
        try:
            result = check["func"]()
            check["last_result"] = {"status": "healthy", "result": result, "timestamp": time.time()}
            check["last_error"] = None
            self.last_check_time[name] = time.time()
            return check["last_result"]
        
        except Exception as e:
            error_result = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
            check["last_result"] = error_result
            check["last_error"] = e
            self.last_check_time[name] = time.time()
            logger.error(f"Health check '{name}' failed: {str(e)}")
            return error_result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_status = "healthy"
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            
            if result["status"] == "unhealthy":
                overall_status = "unhealthy"
            elif result["status"] == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "checks": results,
            "timestamp": time.time()
        }

# Global health checker instance
health_checker = HealthChecker()

def setup_default_health_checks():
    """Setup default health checks"""
    
    def check_memory():
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            raise Exception(f"High memory usage: {memory.percent}%")
        return {"memory_usage_percent": memory.percent}
    
    def check_disk():
        import psutil
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            raise Exception(f"High disk usage: {disk.percent}%")
        return {"disk_usage_percent": disk.percent}
    
    try:
        health_checker.register_check("memory", check_memory, interval=30)
        health_checker.register_check("disk", check_disk, interval=60)
    except ImportError:
        logger.warning("psutil not available, skipping system health checks")

# Initialize health checks
setup_default_health_checks()
