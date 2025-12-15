"""
Agent Core - Main logic for SAM3 Agent with iterative reasoning
"""
import json
import os
import copy
from typing import Any, Dict, List, Optional
from pathlib import Path

from agent.client_llm import MLLMClient
from agent.client_sam3 import SAM3Client
from utils.visualization import visualize_masks, visualize_single_mask


class AgentCore:
    """SAM3 Agent with iterative MLLM reasoning"""
    
    def __init__(
        self,
        mllm_client: MLLMClient,
        sam3_client: SAM3Client,
        output_dir: str = "./outputs"
    ):
        """
        Initialize Agent Core
        
        Args:
            mllm_client: MLLM client instance
            sam3_client: SAM3 client instance
            output_dir: Output directory for results
        """
        self.mllm_client = mllm_client
        self.sam3_client = sam3_client
        self.output_dir = output_dir
        
        # Load system prompts
        current_dir = Path(__file__).parent.parent
        system_prompt_path = current_dir / "prompts" / "system_prompt.txt"
        iterative_prompt_path = current_dir / "prompts" / "system_prompt_iterative_checking.txt"
        
        if system_prompt_path.exists():
            with open(system_prompt_path, 'r') as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = self._get_default_system_prompt()
        
        if iterative_prompt_path.exists():
            with open(iterative_prompt_path, 'r') as f:
                self.iterative_checking_prompt = f.read().strip()
        else:
            self.iterative_checking_prompt = self._get_default_iterative_prompt()
    
    def _normalize_result_paths(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all paths in result to be relative to output_dir
        
        Args:
            result: Result dictionary with paths
            
        Returns:
            Result with normalized paths
        """
        normalized = result.copy()
        
        # Normalize history paths
        if 'history' in normalized and normalized['history']:
            for round_data in normalized['history']:
                if 'sam3_calls' in round_data:
                    for call in round_data['sam3_calls']:
                        if 'json_path' in call:
                            call['json_path'] = self.sam3_client.normalize_path(call['json_path'])
                        if 'image_path' in call:
                            call['image_path'] = self.sam3_client.normalize_path(call['image_path'])
        
        # Normalize final output paths
        if 'final_output' in normalized and normalized['final_output']:
            if 'json_path' in normalized['final_output']:
                normalized['final_output']['json_path'] = self.sam3_client.normalize_path(
                    normalized['final_output']['json_path']
                )
            if 'image_path' in normalized['final_output']:
                normalized['final_output']['image_path'] = self.sam3_client.normalize_path(
                    normalized['final_output']['image_path']
                )
        
        return normalized
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if file doesn't exist"""
        return """You are a helpful visual-concept grounding assistant capable of leveraging tool calls to ground concepts the user refers to, and providing structured JSON outputs and tool calls.

Available tools:
- segment_phrase: Segment instances of a simple noun phrase
- examine_each_mask: Examine each mask individually 
- select_masks_and_return: Select final masks as output
- report_no_mask: Report that no masks match the query

You must analyze the image carefully, call tools iteratively, and return the masks that best match the user's query."""
    
    def _get_default_iterative_prompt(self) -> str:
        """Get default iterative checking prompt"""
        return """You are examining a specific segmentation mask. Determine if this mask correctly matches the user's query.

Respond with either:
<verdict>Accept</verdict> if the mask is correct
<verdict>Reject</verdict> if the mask is incorrect"""
    
    def run(
        self,
        image_path: str,
        text_prompt: str,
        debug: bool = True,
        max_generations: int = 100,
        event_callback=None
    ) -> Dict[str, Any]:
        """
        Run SAM3 Agent with iterative reasoning
        
        Args:
            image_path: Path to input image
            text_prompt: User's segmentation query
            debug: Enable debug mode
            max_generations: Maximum MLLM generation rounds
            event_callback: Optional callback function for progress events
        
        Returns:
            Dictionary with agent history and final results
        """
        print(f"\n{'='*60}")
        print(f"🤖 Starting SAM3 Agent")
        print(f"📷 Image: {image_path}")
        print(f"💬 Query: {text_prompt}")
        print(f"{'='*60}\n")
        
        # Emit start event
        if event_callback:
            event_callback({
                'type': 'agent_start',
                'data': {
                    'image_path': image_path,
                    'text_prompt': text_prompt
                }
            })
        
        # Initialize state
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": f"The above image is the raw input image. The initial user input query is: '{text_prompt}'."
                    }
                ]
            }
        ]
        
        agent_history = []
        used_text_prompts = set()
        latest_sam3_text_prompt = ""
        latest_output_json_path = ""
        generation_count = 0
        
        # Create output subdirectories
        sam_output_dir = os.path.join(self.output_dir, "sam_outputs")
        agent_output_dir = os.path.join(self.output_dir, "agent_outputs")
        os.makedirs(sam_output_dir, exist_ok=True)
        os.makedirs(agent_output_dir, exist_ok=True)
        
        # Main agent loop
        while generation_count < max_generations:
            print(f"\n{'-'*30} Round {generation_count + 1} {'-'*30}\n")
            
            # Emit round start event
            if event_callback:
                event_callback({
                    'type': 'round_start',
                    'data': {
                        'round': generation_count + 1
                    }
                })
            
            # Call MLLM with streaming
            if event_callback:
                # Stream the response
                if event_callback:
                    event_callback({
                        'type': 'llm_start',
                        'data': {
                            'round': generation_count + 1
                        }
                    })
                
                generated_text = ""
                stream = self.mllm_client.generate(messages, stream=True)
                for chunk in stream:
                    generated_text += chunk
                    if event_callback:
                        event_callback({
                            'type': 'llm_chunk',
                            'data': {
                                'round': generation_count + 1,
                                'chunk': chunk,
                                'accumulated': generated_text
                            }
                        })
                
                if event_callback:
                    event_callback({
                        'type': 'llm_complete',
                        'data': {
                            'round': generation_count + 1,
                            'text': generated_text
                        }
                    })
            else:
                # Non-streaming mode
                generated_text = self.mllm_client.generate(messages)
            
            if generated_text is None:
                print("❌ MLLM returned None, stopping agent")
                break
            
            print(f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n")
            
            # Record in history
            agent_history.append({
                'round': generation_count + 1,
                'messages': copy.deepcopy(messages),
                'generated_text': generated_text,
                'sam3_calls': [],
                'status': 'complete'
            })
            
            # Parse tool call
            if "<tool>" not in generated_text:
                print("⚠️  No tool call found in response")
                break
            
            # Extract tool call JSON
            tool_call_str = generated_text.split("<tool>")[-1].split("</tool>")[0].strip()
            tool_call_str = tool_call_str.replace("}}}", "}}")  # Fix extra braces
            
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("name")
            print(f"🔧 Tool call: {tool_name}")
            
            # Handle different tool calls
            if tool_name == "segment_phrase":
                result = self._handle_segment_phrase(
                    tool_call, messages, generated_text, image_path, text_prompt,
                    used_text_prompts, sam_output_dir, event_callback
                )
                
                if result is None:
                    break
                
                latest_sam3_text_prompt = result['text_prompt']
                latest_output_json_path = result['json_path']
                
                # Add SAM3 call info to current round's history
                if agent_history:
                    agent_history[-1]['sam3_calls'].append({
                        'text_prompt': result['text_prompt'],
                        'num_masks': result['num_masks'],
                        'json_path': result['json_path'],
                        'image_path': result['image_path'],
                        'status': 'complete'
                    })
            
            elif tool_name == "examine_each_mask":
                result = self._handle_examine_each_mask(
                    messages, generated_text, latest_output_json_path,
                    image_path, text_prompt, sam_output_dir
                )
                
                if result:
                    latest_output_json_path = result['json_path']
            
            elif tool_name == "select_masks_and_return":
                final_result = self._handle_select_masks(
                    tool_call, latest_output_json_path, agent_output_dir,
                    image_path, text_prompt
                )
                
                # Add the final visualization to current round's SAM3 calls
                # This shows what the final selection looks like
                if agent_history and final_result:
                    agent_history[-1]['sam3_calls'].append({
                        'text_prompt': f"Final selection: {tool_call['parameters']['final_answer_masks']}",
                        'num_masks': final_result['num_masks'],
                        'json_path': final_result['json_path'],
                        'image_path': final_result['image_path'],
                        'status': 'complete'
                    })
                
                print(f"\n{'='*60}")
                print("✅ Agent completed successfully!")
                print(f"{'='*60}\n")
                
                result = {
                    'status': 'success',
                    'history': agent_history,
                    'final_output': final_result
                }
                
                return self._normalize_result_paths(result)
            
            elif tool_name == "report_no_mask":
                print("📝 Agent reports no valid masks found")
                
                result = {
                    'status': 'no_masks',
                    'history': agent_history,
                    'message': 'No objects match the query'
                }
                
                return self._normalize_result_paths(result)
            
            else:
                print(f"⚠️  Unknown tool: {tool_name}")
                break
            
            generation_count += 1
        
        print(f"\n⚠️  Agent stopped after {generation_count} rounds")
        
        result = {
            'status': 'incomplete',
            'history': agent_history,
            'message': f'Agent stopped after {generation_count} rounds'
        }
        
        return self._normalize_result_paths(result)
    
    def _handle_segment_phrase(
        self, tool_call, messages, generated_text, image_path, text_prompt,
        used_text_prompts, sam_output_dir, event_callback=None
    ):
        """Handle segment_phrase tool call"""
        text_prompt_param = tool_call['parameters']['text_prompt']
        
        # Emit SAM3 start event
        if event_callback:
            event_callback({
                'type': 'sam3_start',
                'data': {
                    'text_prompt': text_prompt_param,
                    'image_path': image_path
                }
            })
        
        # Check for duplicate prompts
        if text_prompt_param in used_text_prompts:
            print(f"⚠️  Prompt '{text_prompt_param}' already used")
            
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text}]
            })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"You have already used '{text_prompt_param}'. Please try a different prompt. Previously used: {list(used_text_prompts)}"
                }]
            })
            return None
        
        # Add to used prompts
        used_text_prompts.add(text_prompt_param)
        
        # Call SAM3
        sam3_result = self.sam3_client.segment(
            image_path=image_path,
            text_prompt=text_prompt_param,
            output_folder=sam_output_dir
        )
        
        num_masks = sam3_result['num_masks']
        
        # Emit SAM3 complete event
        if event_callback:
            event_callback({
                'type': 'sam3_complete',
                'data': {
                    'text_prompt': text_prompt_param,
                    'num_masks': num_masks,
                    'json_path': self.sam3_client.normalize_path(sam3_result['json_path']),
                    'image_path': self.sam3_client.normalize_path(sam3_result['image_path'])
                }
            })
        
        # Add assistant message
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}]
        })
        
        # Add user message with results
        if num_masks == 0:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"The segment_phrase tool generated 0 masks for '{text_prompt_param}'. Try a different prompt."
                }]
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The segment_phrase tool generated {num_masks} masks. Analyze them carefully. Original query: '{text_prompt}'"
                    },
                    {"type": "image", "image": sam3_result['image_path']}
                ]
            })
        
        return {
            'text_prompt': text_prompt_param,
            'json_path': sam3_result['json_path'],
            'image_path': sam3_result['image_path'],
            'num_masks': num_masks
        }
    
    def _handle_examine_each_mask(
        self, messages, generated_text, latest_output_json_path,
        image_path, text_prompt, sam_output_dir
    ):
        """Handle examine_each_mask tool call"""
        # Load current outputs
        with open(latest_output_json_path, 'r') as f:
            current_outputs = json.load(f)
        
        num_masks = len(current_outputs['pred_masks'])
        masks_to_keep = []
        
        print(f"🔍 Examining {num_masks} masks individually...")
        
        # Check each mask
        for i in range(num_masks):
            print(f"  Checking mask {i+1}/{num_masks}...")
            
            # Visualize single mask
            mask_image = visualize_single_mask(current_outputs, i)
            mask_image_path = latest_output_json_path.replace('.json', f'_mask_{i+1}.png')
            mask_image.save(mask_image_path)
            
            # Ask MLLM to verify
            check_messages = [
                {"role": "system", "content": self.iterative_checking_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: '{text_prompt}'"},
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Mask to evaluate:"},
                        {"type": "image", "image": mask_image_path}
                    ]
                }
            ]
            
            verdict_text = self.mllm_client.generate(check_messages)
            
            if verdict_text and "<verdict>" in verdict_text:
                verdict = verdict_text.split("<verdict>")[-1].split("</verdict>")[0].strip()
                
                if "Accept" in verdict:
                    print(f"    ✅ Mask {i+1} accepted")
                    masks_to_keep.append(i)
                else:
                    print(f"    ❌ Mask {i+1} rejected")
        
        # Update outputs with kept masks
        updated_outputs = {
            'original_image_path': current_outputs['original_image_path'],
            'orig_img_h': current_outputs['orig_img_h'],
            'orig_img_w': current_outputs['orig_img_w'],
            'pred_boxes': [current_outputs['pred_boxes'][i] for i in masks_to_keep],
            'pred_scores': [current_outputs['pred_scores'][i] for i in masks_to_keep],
            'pred_masks': [current_outputs['pred_masks'][i] for i in masks_to_keep]
        }
        
        # Save updated outputs
        updated_json_path = latest_output_json_path.replace('.json', f'_filtered.json')
        with open(updated_json_path, 'w') as f:
            json.dump(updated_outputs, f, indent=4)
        
        # Visualize
        updated_viz = visualize_masks(updated_outputs)
        updated_viz_path = updated_json_path.replace('.json', '.png')
        updated_viz.save(updated_viz_path)
        
        # Update messages
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}]
        })
        
        if len(masks_to_keep) == 0:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"All masks were rejected. Try a different prompt. Query: '{text_prompt}'"
                }]
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"After examination, {len(masks_to_keep)} masks remain. Query: '{text_prompt}'"
                    },
                    {"type": "image", "image": updated_viz_path}
                ]
            })
        
        return {'json_path': updated_json_path}
    
    def _handle_select_masks(
        self, tool_call, latest_output_json_path, agent_output_dir,
        image_path, text_prompt
    ):
        """Handle select_masks_and_return tool call"""
        mask_indices = tool_call['parameters']['final_answer_masks']
        
        # Load outputs from JSON
        with open(latest_output_json_path, 'r') as f:
            current_outputs = json.load(f)
        
        # Convert 1-indexed to 0-indexed
        mask_indices_0 = [i - 1 for i in mask_indices if 0 < i <= len(current_outputs['pred_masks'])]
        
        # Create final outputs
        final_outputs = {
            'original_image_path': current_outputs['original_image_path'],
            'orig_img_h': current_outputs['orig_img_h'],
            'orig_img_w': current_outputs['orig_img_w'],
            'pred_boxes': [current_outputs['pred_boxes'][i] for i in mask_indices_0],
            'pred_scores': [current_outputs['pred_scores'][i] for i in mask_indices_0],
            'pred_masks': [current_outputs['pred_masks'][i] for i in mask_indices_0],
            'text_prompt': text_prompt,
            'selected_mask_numbers': mask_indices
        }
        
        # Save final outputs
        image_name = os.path.basename(image_path).rsplit('.', 1)[0]
        final_json_path = os.path.join(agent_output_dir, f"{image_name}_final.json")
        final_image_path = os.path.join(agent_output_dir, f"{image_name}_final.png")
        
        with open(final_json_path, 'w') as f:
            json.dump(final_outputs, f, indent=4)
        
        # Visualize final result
        final_viz = visualize_masks(final_outputs)
        final_viz.save(final_image_path)
        
        print(f"✅ Final output saved:")
        print(f"   JSON: {final_json_path}")
        print(f"   Image: {final_image_path}")
        print(f"   Selected {len(mask_indices)} masks")
        
        return {
            'json_path': final_json_path,
            'image_path': final_image_path,
            'num_masks': len(mask_indices),
            'outputs': final_outputs
        }
