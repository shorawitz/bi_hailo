import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

from .hailo_detector import Detection

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize image processor."""
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.save_images = config.get('save_images', True)
        self.save_detections = config.get('save_detections', True)
        self.draw_boxes = config.get('draw_boxes', True)
        self.webhook_url = config.get('webhook_url')
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load font for drawing
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            self.font = ImageFont.load_default()
    
    async def process_detections(self, image_data: bytes, detections: List[Detection]):
        """Process detection results."""
        if not detections:
            return
        
        logger.info(f"Processing {len(detections)} detections")
        
        # Convert image data to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Draw bounding boxes if enabled
        if self.draw_boxes:
            image = self._draw_detections(image, detections)
        
        # Save processed image
        if self.save_images:
            await self._save_image(image, detections)
        
        # Save detection data
        if self.save_detections:
            await self._save_detection_data(detections)
        
        # Send webhook notification
        if self.webhook_url:
            await self._send_webhook(detections)
    
    def _draw_detections(self, image: Image.Image, detections: List[Detection]) -> Image.Image:
        """Draw bounding boxes and labels on image."""
        draw = ImageDraw.Draw(image)
        
        # Define colors for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors[detection.class_id % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Get text size for background
            bbox = draw.textbbox((0, 0), label, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                         fill=color, outline=color)
            
            # Draw label text
            draw.text((x1 + 2, y1 - text_height - 2), label, 
                     fill=(255, 255, 255), font=self.font)
        
        return image
    
    async def _save_image(self, image: Image.Image, detections: List[Detection]):
        """Save processed image to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        filepath = self.output_dir / filename
        
        # Save image
        image.save(filepath, "JPEG", quality=95)
        logger.info(f"Saved image: {filepath}")
    
    async def _save_detection_data(self, detections: List[Detection]):
        """Save detection data as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert detections to serializable format
        detection_data = {
            "timestamp": datetime.now().isoformat(),
            "detections": [
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": det.bbox
                }
                for det in detections
            ]
        }
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"Saved detection data: {filepath}")
    
    async def _send_webhook(self, detections: List[Detection]):
        """Send webhook notification with detection results."""
        try:
            import aiohttp
            
            payload = {
                "timestamp": datetime.now().isoformat(),
                "detection_count": len(detections),
                "detections": [
                    {
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox": det.bbox
                    }
                    for det in detections
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Webhook notification sent successfully")
                    else:
                        logger.warning(f"Webhook failed: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")