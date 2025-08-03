import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class BlueIrisClient:
    def __init__(self, config: Dict[str, Any]):
        """Initialize BlueIris client."""
        self.host = config['host']
        self.port = config['port']
        self.username = config['username']
        self.password = config['password']
        self.camera_name = config['camera_name']
        self.session = None
        self.session_id = None
    
    async def connect(self):
        """Connect to BlueIris server."""
        self.session = aiohttp.ClientSession()
        
        # Authenticate with BlueIris
        await self._authenticate()
        logger.info(f"Connected to BlueIris at {self.host}:{self.port}")
    
    async def disconnect(self):
        """Disconnect from BlueIris server."""
        if self.session:
            await self.session.close()
            logger.info("Disconnected from BlueIris")
    
    async def _authenticate(self):
        """Authenticate with BlueIris server."""
        url = f"http://{self.host}:{self.port}/json"
        
        # First request to get session
        data = {
            "cmd": "login"
        }
        
        async with self.session.post(url, json=data) as response:
            result = await response.json()
            session = result.get('session')
            
            if not session:
                raise Exception("Failed to get session from BlueIris")
        
        # Second request with credentials
        import hashlib
        import hmac
        
        # Create password hash
        password_hash = hashlib.md5(f"{self.username}:{session}:{self.password}".encode()).hexdigest()
        
        data = {
            "cmd": "login",
            "session": session,
            "response": password_hash
        }
        
        async with self.session.post(url, json=data) as response:
            result = await response.json()
            
            if result.get('result') != 'success':
                raise Exception(f"Authentication failed: {result}")
            
            self.session_id = result.get('session')
    
    async def get_latest_image(self) -> Optional[bytes]:
        """Get the latest image from specified camera."""
        try:
            url = f"http://{self.host}:{self.port}/image/{self.camera_name}"
            params = {
                'session': self.session_id,
                'q': 100,  # Quality
                'w': 640,  # Width
                'h': 480   # Height
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.warning(f"Failed to get image: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting image from BlueIris: {e}")
            return None
    
    async def get_camera_list(self) -> Dict[str, Any]:
        """Get list of available cameras."""
        url = f"http://{self.host}:{self.port}/json"
        data = {
            "cmd": "camlist",
            "session": self.session_id
        }
        
        async with self.session.post(url, json=data) as response:
            return await response.json()