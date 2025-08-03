import asyncio
import logging
from pathlib import Path
from typing import Optional
import argparse

from src.blueiris_client import BlueIrisClient
from src.hailo_detector import HailoObjectDetector
from src.image_processor import ImageProcessor
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HailoBlueIrisApp:
    def __init__(self, config_path: str):
        """Initialize the Hailo BlueIris detection application."""
        self.config = Config.load_from_file(config_path)
        self.blueiris_client = BlueIrisClient(self.config.blueiris)
        self.hailo_detector = HailoObjectDetector(self.config.hailo)
        self.image_processor = ImageProcessor(self.config.processing)
        self.running = False
    
    async def start(self):
        """Start the detection service."""
        logger.info("Starting Hailo BlueIris detection service...")
        
        try:
            # Initialize Hailo detector
            await self.hailo_detector.initialize()
            
            # Start BlueIris client
            await self.blueiris_client.connect()
            
            self.running = True
            await self._process_loop()
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the detection service."""
        logger.info("Stopping detection service...")
        self.running = False
        
        if hasattr(self, 'hailo_detector'):
            await self.hailo_detector.cleanup()
        
        if hasattr(self, 'blueiris_client'):
            await self.blueiris_client.disconnect()
    
    async def _process_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get image from BlueIris
                image_data = await self.blueiris_client.get_latest_image()
                
                if image_data:
                    # Process image with Hailo
                    detections = await self.hailo_detector.detect_objects(image_data)
                    
                    # Process results
                    await self.image_processor.process_detections(
                        image_data, detections
                    )
                
                # Wait before next iteration
                await asyncio.sleep(self.config.processing.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)

async def main():
    parser = argparse.ArgumentParser(description='Hailo AI BlueIris Object Detection')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path')
    args = parser.parse_args()
    
    app = HailoBlueIrisApp(args.config)
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())