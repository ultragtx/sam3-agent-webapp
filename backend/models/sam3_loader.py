"""
SAM3 Model Loader and Manager
Keeps SAM3 model in GPU memory for fast inference
"""
import os
import torch
from typing import Optional


class SAM3ModelManager:
    """Manages SAM3 model loading and persistence in memory"""
    
    def __init__(self, bpe_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize SAM3 Model Manager
        
        Args:
            bpe_path: Path to BPE vocabulary file (e.g., bpe_simple_vocab_16e6.txt.gz)
            device: Device to load model on (cuda/cpu)
        """
        self.bpe_path = bpe_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load_model(self):
        """Load SAM3 model into memory"""
        if self._loaded:
            print("SAM3 model already loaded")
            return
        
        print(f"Loading SAM3 model with BPE path: {self.bpe_path} on {self.device}...")
        
        try:
            # Import SAM3 modules based on your demo code
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Build SAM3 model
            self.model = build_sam3_image_model(
                bpe_path=self.bpe_path,
            )
            
            # Move to device if needed
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            # Create processor
            self.processor = Sam3Processor(self.model)
            self._loaded = True
            
            print(f"✅ SAM3 model loaded successfully on {self.device}")
        
        except ImportError as e:
            print(f"❌ Failed to import SAM3: {e}")
            print("Please install SAM3 package or add it to Python path")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    def unload_model(self):
        """Unload model from memory"""
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        print("SAM3 model unloaded from memory")
    
    def get_processor(self):
        """Get the SAM3 processor"""
        if not self._loaded:
            raise RuntimeError("SAM3 model not loaded. Call load_model() first.")
        return self.processor
