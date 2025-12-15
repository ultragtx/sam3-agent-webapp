"""SAM3 Client for segmentation operations"""
import json
import os
from typing import Any, Dict, Optional
from PIL import Image
import torch

from utils.visualization import visualize_masks


class SAM3Client:
    """Client for SAM3 segmentation operations"""
    
    def __init__(self, processor):
        """
        Initialize SAM3 Client
        
        Args:
            processor: SAM3 processor/predictor instance
        """
        self.processor = processor
    
    def _sam3_inference(self, image_path: str, text_prompt: str) -> Dict[str, Any]:
        """
        Run SAM3 inference with text prompt
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for segmentation
        
        Returns:
            Dictionary with segmentation results
        """
        # Load image and ensure RGB format (remove alpha channel if present)
        image = Image.open(image_path).convert('RGB')
        orig_img_w, orig_img_h = image.size
        
        # SAM3 inference - following the demo code pattern
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )
        
        # Extract outputs - output is a dict with 'masks', 'boxes', 'scores'
        boxes = output.get('boxes')
        masks = output.get('masks')
        scores = output.get('scores')
        
        # Handle None values
        if boxes is None:
            boxes = torch.tensor([])
        if masks is None:
            masks = torch.tensor([])
        if scores is None:
            scores = torch.tensor([])
        
        print(f"SAM3 raw output - boxes: {boxes.shape if hasattr(boxes, 'shape') else type(boxes)}, masks: {masks.shape if hasattr(masks, 'shape') else type(masks)}, scores: {scores.shape if hasattr(scores, 'shape') else type(scores)}")
        
        # Normalize boxes to [0, 1]
        if len(boxes) > 0:
            pred_boxes_normalized = [
                [
                    float(box[0] / orig_img_w),
                    float(box[1] / orig_img_h),
                    float((box[2] - box[0]) / orig_img_w),
                    float((box[3] - box[1]) / orig_img_h)
                ]
                for box in boxes.cpu().numpy()
            ]
        else:
            pred_boxes_normalized = []
        
        # Encode masks to RLE
        from utils.rle import rle_encode
        pred_masks_rle = rle_encode(masks.squeeze(1).cpu().numpy()) if len(masks) > 0 else []
        pred_masks_rle = [m['counts'] for m in pred_masks_rle]
        
        outputs = {
            'orig_img_h': orig_img_h,
            'orig_img_w': orig_img_w,
            'pred_boxes': pred_boxes_normalized,
            'pred_masks': pred_masks_rle,
            'pred_scores': scores.tolist() if len(scores) > 0 else []
        }
        
        return outputs
    
    def _remove_overlapping_masks(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove overlapping masks based on IoU threshold"""
        # Import from helpers if available, otherwise skip
        try:
            from utils.mask_overlap_removal import remove_overlapping_masks
            return remove_overlapping_masks(outputs)
        except ImportError:
            print("Warning: mask_overlap_removal not available, skipping")
            return outputs
    
    def segment(
        self,
        image_path: str,
        text_prompt: str,
        output_folder: str = "sam3_output"
    ) -> Dict[str, Any]:
        """
        Perform SAM3 segmentation with text prompt
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for segmentation
            output_folder: Folder to save outputs
        
        Returns:
            Dictionary with results including paths to saved files
        """
        print(f"📞 SAM3 segmenting '{image_path}' with prompt '{text_prompt}'...")
        
        # Create output paths
        text_prompt_safe = text_prompt.replace('/', '_').replace(' ', '_')
        image_name = os.path.basename(image_path).rsplit('.', 1)[0]
        
        os.makedirs(os.path.join(output_folder, image_name), exist_ok=True)
        
        output_json_path = os.path.join(
            output_folder,
            image_name,
            f"{text_prompt_safe}.json"
        )
        output_image_path = os.path.join(
            output_folder,
            image_name,
            f"{text_prompt_safe}.png"
        )
        
        # Run SAM3 inference
        outputs = self._sam3_inference(image_path, text_prompt)
        
        # Remove overlapping masks
        outputs = self._remove_overlapping_masks(outputs)
        
        # Sort by scores (highest first)
        if outputs['pred_scores']:
            score_indices = sorted(
                range(len(outputs['pred_scores'])),
                key=lambda i: outputs['pred_scores'][i],
                reverse=True
            )
            
            outputs['pred_scores'] = [outputs['pred_scores'][i] for i in score_indices]
            outputs['pred_boxes'] = [outputs['pred_boxes'][i] for i in score_indices]
            outputs['pred_masks'] = [outputs['pred_masks'][i] for i in score_indices]
        
        # Filter out invalid RLE masks (too short)
        valid_indices = [i for i, rle in enumerate(outputs['pred_masks']) if len(rle) > 4]
        outputs['pred_masks'] = [outputs['pred_masks'][i] for i in valid_indices]
        outputs['pred_boxes'] = [outputs['pred_boxes'][i] for i in valid_indices]
        outputs['pred_scores'] = [outputs['pred_scores'][i] for i in valid_indices]
        
        # Add metadata
        outputs['original_image_path'] = image_path
        outputs['output_image_path'] = output_image_path
        outputs['text_prompt'] = text_prompt
        
        # Save JSON
        with open(output_json_path, 'w') as f:
            json.dump(outputs, f, indent=4)
        
        print(f"✅ JSON saved to '{output_json_path}'")
        
        # Visualize and save
        print("🔍 Rendering visualization...")
        viz_image = visualize_masks(outputs)
        viz_image.save(output_image_path)
        
        print(f"✅ Visualization saved to '{output_image_path}'")
        
        return {
            'json_path': output_json_path,
            'image_path': output_image_path,
            'num_masks': len(outputs['pred_masks']),
            'outputs': outputs
        }
