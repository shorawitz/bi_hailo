import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize configuration."""
        self.blueiris = config_data.get('blueiris', {})
        self.hailo = config_data.get('hailo', {})
        self.processing = config_data.get('processing', {})
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, creating default")
            cls._create_default_config(config_file)
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(config_data)
    
    @staticmethod
    def _create_default_config(config_path: Path):
        """Create default configuration file."""
        default_config = {
            'blueiris': {
                'host': '192.168.1.100',
                'port': 81,
                'username': 'admin',
                'password': 'password',
                'camera_name': 'cam1'
            },
            'hailo': {
                'model_path': '/path/to/your/model.hef',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.45,
                'input_size': [640, 640]
            },
            'processing': {
                'poll_interval': 1.0,
                'output_dir': 'output',
                'save_images': True,
                'save_detections': True,
                'draw_boxes': True,
                'webhook_url': None
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created default config file: {config_path}")