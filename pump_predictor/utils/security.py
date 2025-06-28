"""
Security and authentication utilities for the API
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from passlib.context import CryptContext
import time
from collections import defaultdict
import os

from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)

class SecurityManager:
    """Centralized security management"""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or database"""
        # In production, load from secure storage
        return {
            os.getenv("API_KEY_ADMIN", "admin_key_123"): {
                "name": "admin",
                "permissions": ["read", "write", "admin"],
                "rate_limit": 1000  # requests per hour
            },
            os.getenv("API_KEY_USER", "user_key_456"): {
                "name": "user",
                "permissions": ["read"],
                "rate_limit": 100
            }
        }
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return user info"""
        if api_key not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return self.api_keys[api_key]

# Global security manager instance
security_manager = SecurityManager()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_manager.security)):
    """Get current authenticated user"""
    token = credentials.credentials
    
    # Try JWT token first
    try:
        payload = security_manager.verify_token(token)
        return {"type": "jwt", "user": payload}
    except:
        pass
    
    # Try API key
    try:
        user_info = security_manager.verify_api_key(token)
        return {"type": "api_key", "user": user_info}
    except:
        pass
    
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: dict = Depends(get_current_user)):
        user_permissions = current_user["user"].get("permissions", [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, requests: int = 100, window: int = 3600):
        self.requests = requests
        self.window = window
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        
        # Clean old entries
        rate_limit_storage[identifier] = [
            timestamp for timestamp in rate_limit_storage[identifier]
            if now - timestamp < self.window
        ]
        
        # Check if within limit
        if len(rate_limit_storage[identifier]) >= self.requests:
            return False
        
        # Add current request
        rate_limit_storage[identifier].append(now)
        return True
    
    def get_identifier(self, request: Request, user: Optional[dict] = None) -> str:
        """Get identifier for rate limiting"""
        if user and user["type"] == "api_key":
            return f"api_key:{user['user']['name']}"
        elif user and user["type"] == "jwt":
            return f"jwt:{user['user'].get('sub', 'unknown')}"
        else:
            # Fallback to IP address
            client_ip = request.client.host
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()
            return f"ip:{client_ip}"

def rate_limit(requests: int = 100, window: int = 3600):
    """Rate limiting dependency"""
    limiter = RateLimiter(requests, window)
    
    def check_rate_limit(
        request: Request,
        current_user: Optional[dict] = Depends(get_current_user)
    ):
        identifier = limiter.get_identifier(request, current_user)
        
        # Get user-specific rate limit if available
        if current_user and current_user["type"] == "api_key":
            user_limit = current_user["user"].get("rate_limit", requests)
            limiter.requests = user_limit
        
        if not limiter.is_allowed(identifier):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(window)}
            )
        
        return current_user
    
    return check_rate_limit

def input_sanitizer(data: Any) -> Any:
    """Sanitize input data to prevent injection attacks"""
    if isinstance(data, str):
        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", "&", "'", '"', "script", "javascript", "eval", "exec"]
        sanitized = data
        for char in dangerous_chars:
            if char in sanitized.lower():
                logger.warning(f"Potentially dangerous input detected: {char}")
                sanitized = sanitized.replace(char, "")
        return sanitized
    elif isinstance(data, dict):
        return {k: input_sanitizer(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [input_sanitizer(item) for item in data]
    else:
        return data

def generate_api_key(name: str, permissions: list, rate_limit: int = 100) -> str:
    """Generate new API key"""
    # Create a secure random key
    key = secrets.token_urlsafe(32)
    
    # In production, store in database
    security_manager.api_keys[key] = {
        "name": name,
        "permissions": permissions,
        "rate_limit": rate_limit,
        "created_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Generated API key for {name}")
    return key

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging"""
    return hashlib.sha256(data.encode()).hexdigest()[:8]
