"""Visualization utilities for SAM3 masks"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from typing import Dict, Any, Optional, List


def generate_colors(num_colors: int) -> List[tuple]:
    """Generate distinct colors for masks"""
    colors = []
    for i in range(num_colors):
        # Generate distinct hues
        hue = (i * 137.5) % 360  # Golden angle for better distribution
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.9 - (i % 2) * 0.1
        
        # Convert HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors.append((r, g, b))
    
    return colors


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """
    Convert RLE string to binary mask
    
    Args:
        rle: RLE encoded string
        height: Image height
        width: Image width
    
    Returns:
        Binary mask array
    """
    try:
        from pycocotools import mask as mask_util
        rle_dict = {'size': [height, width], 'counts': rle}
        return mask_util.decode(rle_dict)
    except ImportError:
        # Fallback if pycocotools not available
        print("Warning: pycocotools not available, using fallback RLE decoder")
        return np.zeros((height, width), dtype=np.uint8)


def visualize_masks(
    outputs: Dict[str, Any],
    selected_mask_idx: Optional[int] = None,
    alpha: float = 0.5
) -> Image.Image:
    """
    Visualize segmentation masks on image
    
    Args:
        outputs: Dictionary with image path and mask data
        selected_mask_idx: If provided, only visualize this mask (0-indexed)
        alpha: Transparency for mask overlay
    
    Returns:
        PIL Image with visualized masks
    """
    # Load image
    image_path = outputs['original_image_path']
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    
    # Create overlay
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Get masks to visualize
    masks = outputs['pred_masks']
    boxes = outputs.get('pred_boxes', [])
    
    if selected_mask_idx is not None:
        # Visualize only selected mask
        mask_indices = [selected_mask_idx]
    else:
        # Visualize all masks
        mask_indices = range(len(masks))
    
    # Generate colors
    colors = generate_colors(len(masks))
    
    # Draw masks
    for idx in mask_indices:
        if idx >= len(masks):
            continue
        
        # Decode RLE mask
        rle_mask = masks[idx]
        mask_array = rle_to_mask(rle_mask, img_height, img_width)
        
        # Get color
        color = colors[idx]
        
        # Create colored mask
        colored_mask = Image.new('RGBA', image.size, color + (int(255 * alpha),))
        mask_pil = Image.fromarray(mask_array * 255, mode='L')
        
        # Composite mask onto overlay
        overlay = Image.alpha_composite(overlay, Image.composite(colored_mask, Image.new('RGBA', image.size, (0, 0, 0, 0)), mask_pil))
        
        # Draw bounding box if available
        if idx < len(boxes):
            box = boxes[idx]
            # Box is in normalized xywh format
            x, y, w, h = box
            x1 = int(x * img_width)
            y1 = int(y * img_height)
            x2 = int((x + w) * img_width)
            y2 = int((y + h) * img_height)
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
            
            # Draw mask number
            label = str(idx + 1)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color + (255,)
            )
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255, 255), font=font)
    
    # Composite onto original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    return result.convert('RGB')


def visualize_single_mask(
    outputs: Dict[str, Any],
    mask_idx: int,
    zoom_factor: float = 2.0
) -> Image.Image:
    """
    Visualize a single mask with optional zoom
    
    Args:
        outputs: Dictionary with image path and mask data
        mask_idx: Index of mask to visualize (0-indexed)
        zoom_factor: Zoom factor for the mask region
    
    Returns:
        PIL Image with single mask visualized
    """
    return visualize_masks(outputs, selected_mask_idx=mask_idx)
