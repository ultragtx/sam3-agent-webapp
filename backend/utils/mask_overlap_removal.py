"""Mask overlap removal utilities"""
import numpy as np
from typing import Dict, Any, List


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def remove_overlapping_masks(
    outputs: Dict[str, Any],
    iou_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Remove overlapping masks based on IoU threshold
    
    Args:
        outputs: Dictionary with mask data
        iou_threshold: IoU threshold for considering masks as overlapping
    
    Returns:
        Updated outputs with overlapping masks removed
    """
    try:
        from utils.rle import rle_decode
    except ImportError:
        from ..utils.rle import rle_decode
    
    masks = outputs.get('pred_masks', [])
    boxes = outputs.get('pred_boxes', [])
    scores = outputs.get('pred_scores', [])
    
    if len(masks) <= 1:
        return outputs
    
    # Decode masks
    height = outputs['orig_img_h']
    width = outputs['orig_img_w']
    
    decoded_masks = []
    for rle in masks:
        try:
            mask = rle_decode({'size': [height, width], 'counts': rle})
            decoded_masks.append(mask)
        except:
            # If decoding fails, keep the mask
            decoded_masks.append(np.zeros((height, width), dtype=np.uint8))
    
    # Track masks to keep
    keep_indices = []
    
    # Sort by score (highest first)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    for i in sorted_indices:
        should_keep = True
        
        # Check against already kept masks
        for j in keep_indices:
            iou = compute_iou(decoded_masks[i], decoded_masks[j])
            
            if iou > iou_threshold:
                # Overlapping, don't keep
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(i)
    
    # Sort keep_indices to maintain original order
    keep_indices.sort()
    
    # Filter outputs
    filtered_outputs = {
        'original_image_path': outputs.get('original_image_path'),
        'orig_img_h': outputs['orig_img_h'],
        'orig_img_w': outputs['orig_img_w'],
        'pred_boxes': [boxes[i] for i in keep_indices],
        'pred_masks': [masks[i] for i in keep_indices],
        'pred_scores': [scores[i] for i in keep_indices]
    }
    
    removed_count = len(masks) - len(keep_indices)
    if removed_count > 0:
        print(f"  Removed {removed_count} overlapping masks")
    
    return filtered_outputs
