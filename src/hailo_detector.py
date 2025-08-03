import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

try:
    # Import Hailo SDK components
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    from hailo_platform import InputVStreamParams, OutputVStreamParams, FormatType
    HAILO_AVAILABLE = True
except ImportError:
    logger.warning("Hailo SDK not available. Using mock implementation.")
    HAILO_AVAILABLE = False

class Detection:
    def __init__(self, class_id: int, class_name: str, confidence: float, 
                 bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

class HailoObjectDetector:
    def __init__(self, config: Dict[str, Any]):
        """Initialize Hailo object detector."""
        self.model_path = config['model_path']
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.input_size = config.get('input_size', (640, 640))
        
        # COCO class names (adjust based on your model)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.device = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
    
    async def initialize(self):
        """Initialize the Hailo device and model."""
        if not HAILO_AVAILABLE:
            logger.info("Using mock Hailo detector")
            return
        
        try:
            # Initialize Hailo device
            self.device = VDevice()
            
            # Load HEF file
            hef = HEF(self.model_path)
            
            # Configure network group
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(hef, configure_params)[0]
            
            # Create input and output virtual streams
            self.input_vstreams = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            self.output_vstreams = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            
            logger.info("Hailo detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hailo detector: {e}")
            raise
    
    async def detect_objects(self, image_data: bytes) -> List[Detection]:
        """Detect objects in the given image."""
        if not HAILO_AVAILABLE:
            return await self._mock_detect(image_data)
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            detections = await self._run_inference(processed_image, image.size)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for Hailo inference."""
        # Resize image to model input size
        image_resized = image.resize(self.input_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image_resized, dtype=np.float32)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    async def _run_inference(self, image: np.ndarray, original_size: Tuple[int, int]) -> List[Detection]:
        """Run inference on preprocessed image."""
        detections = []
        
        try:
            with InferVStreams(self.network_group, self.input_vstreams, self.output_vstreams) as infer_pipeline:
                # Run inference
                infer_results = infer_pipeline.infer({list(self.input_vstreams.keys())[0]: image})
                
                # Post-process results
                detections = self._post_process_results(infer_results, original_size)
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
        
        return detections
    
    def _post_process_results(self, results: Dict, original_size: Tuple[int, int]) -> List[Detection]:
        """Post-process inference results to extract detections."""
        detections = []
        
        # This is a simplified post-processing example
        # Adjust based on your specific model output format
        for output_name, output_data in results.items():
            # Assuming YOLO-style output: [batch, num_detections, 85]
            # where 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
            
            output_data = output_data[0]  # Remove batch dimension
            
            for detection in output_data:
                # Extract bbox, confidence, and class scores
                x_center, y_center, width, height = detection[:4]
                confidence = detection[4]
                class_scores = detection[5:]
                
                if confidence > self.confidence_threshold:
                    # Find class with highest score
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id] * confidence
                    
                    if class_confidence > self.confidence_threshold:
                        # Convert to absolute coordinates
                        orig_w, orig_h = original_size
                        x1 = int((x_center - width/2) * orig_w)
                        y1 = int((y_center - height/2) * orig_h)
                        x2 = int((x_center + width/2) * orig_w)
                        y2 = int((y_center + height/2) * orig_h)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, orig_w))
                        y1 = max(0, min(y1, orig_h))
                        x2 = max(0, min(x2, orig_w))
                        y2 = max(0, min(y2, orig_h))
                        
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detections.append(Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(class_confidence),
                            bbox=(x1, y1, x2, y2)
                        ))
        
        # Apply Non-Maximum Suppression
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not detections:
            return []
        
        # Convert to format suitable for cv2.dnn.NMSBoxes
        boxes = []
        confidences = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(det.confidence)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        else:
            return []
    
    async def _mock_detect(self, image_data: bytes) -> List[Detection]:
        """Mock detection for testing when Hailo SDK is not available."""
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Return mock detections
        return [
            Detection(0, "person", 0.85, (100, 100, 200, 300)),
            Detection(2, "car", 0.75, (300, 150, 500, 250))
        ]
    
    async def cleanup(self):
        """Clean up resources."""
        if self.device:
            # Clean up Hailo resources
            pass
        logger.info("Hailo detector cleaned up")