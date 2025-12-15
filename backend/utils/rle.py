"""RLE encoding/decoding utilities"""
import numpy as np
from typing import List, Dict


def rle_encode(masks: np.ndarray) -> List[Dict[str, any]]:
    """
    Encode binary masks to RLE format
    
    Args:
        masks: Binary masks array of shape (N, H, W)
    
    Returns:
        List of RLE dictionaries
    """
    try:
        from pycocotools import mask as mask_util
        
        # Ensure masks are in correct format
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, :, :]
        
        # Convert to uint8
        masks_uint8 = (masks > 0).astype(np.uint8)
        
        # Encode each mask
        rles = []
        for mask in masks_uint8:
            # Ensure mask is in Fortran order for pycocotools
            mask_fortran = np.asfortranarray(mask)
            rle = mask_util.encode(mask_fortran)
            # Decode bytes to string if necessary
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            rles.append(rle)
        
        return rles
    
    except ImportError:
        print("Warning: pycocotools not available, using fallback RLE encoder")
        return _fallback_rle_encode(masks)


def _fallback_rle_encode(masks: np.ndarray) -> List[Dict[str, any]]:
    """Fallback RLE encoder if pycocotools not available"""
    rles = []
    
    if len(masks.shape) == 2:
        masks = masks[np.newaxis, :, :]
    
    for mask in masks:
        h, w = mask.shape
        # Simple RLE encoding
        flat_mask = mask.flatten()
        
        # Create RLE string
        rle_str = ""
        current_val = 0
        count = 0
        
        for val in flat_mask:
            if val == current_val:
                count += 1
            else:
                rle_str += f"{count},"
                current_val = val
                count = 1
        
        rle_str += f"{count}"
        
        rles.append({
            'size': [h, w],
            'counts': rle_str
        })
    
    return rles


def rle_decode(rle: Dict[str, any]) -> np.ndarray:
    """
    Decode RLE to binary mask
    
    Args:
        rle: RLE dictionary with 'size' and 'counts'
    
    Returns:
        Binary mask array
    """
    try:
        from pycocotools import mask as mask_util
        return mask_util.decode(rle)
    
    except ImportError:
        print("Warning: pycocotools not available, using fallback RLE decoder")
        return _fallback_rle_decode(rle)


def _fallback_rle_decode(rle: Dict[str, any]) -> np.ndarray:
    """Fallback RLE decoder if pycocotools not available"""
    h, w = rle['size']
    counts = rle['counts']
    
    if isinstance(counts, str):
        # Parse simple RLE string
        lengths = [int(x) for x in counts.split(',')]
    else:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Decode
    flat_mask = []
    current_val = 0
    
    for length in lengths:
        flat_mask.extend([current_val] * length)
        current_val = 1 - current_val
    
    # Reshape
    mask = np.array(flat_mask[:h*w], dtype=np.uint8).reshape(h, w)
    return mask
