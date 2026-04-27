import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image

# Add G4Splat to path for potential imports
sys.path.append(os.getcwd())

def load_image_transformed(image_path, transform):
    image_source = Image.open(image_path).convert("RGB")
    image_transformed, _ = transform(image_source, None)
    return np.array(image_source), image_transformed

def setup_models(sam_checkpoint, device='cuda'):
    try:
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
        from huggingface_hub import hf_hub_download
        from segment_anything import build_sam, SamPredictor
        import groundingdino.datasets.transforms as T
    except ImportError:
        print("Error: groundingdino or segment_anything not found.")
        print("Please install them using the instructions in 3DRealCar_Toolkit/data_preprocess/README.md")
        sys.exit(1)

    # Load Grounding DINO
    repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=ckpt_filenmae)
    checkpoint = torch.load(cache_file, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.to(device)
    model.eval()

    # Load SAM
    if not os.path.exists(sam_checkpoint):
        print(f"Error: SAM checkpoint not found at {sam_checkpoint}")
        sys.exit(1)
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return model, sam_predictor, transform

def find_max_box(boxes):
    if len(boxes) == 0:
        return -1
    sizes = boxes[:, 2] * boxes[:, 3]
    return torch.argmax(sizes).item()

def run_segmentation(source_path, output_dir, text_prompt, box_threshold, text_threshold, sam_checkpoint, device='cuda'):
    from groundingdino.util.inference import predict
    from groundingdino.util import box_ops
    import pickle

    model, sam_predictor, transform = setup_models(sam_checkpoint, device)
    
    image_paths = sorted(glob(os.path.join(source_path, "images", "*.jpg")) + 
                        glob(os.path.join(source_path, "images", "*.png")))
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in tqdm(image_paths, desc="Segmenting images"):
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_npy = os.path.join(output_dir, f"{name}.npy")
        output_vis = os.path.join(output_dir, f"{name}_vis.jpg")
        output_pkl = os.path.join(output_dir, f"{name}.pkl")
        
        if os.path.exists(output_npy):
            continue
            
        image_source, image_transformed = load_image_transformed(image_path, transform)
        
        boxes, logits, phrases = predict(
            model=model, 
            image=image_transformed, 
            caption=text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
            device=device
        )
        
        if boxes.shape[0] == 0:
            print(f"Warning: No boxes found for {image_path}")
            continue
            
        idx = find_max_box(boxes)
        box = boxes[idx:idx+1]
        
        sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        mask = masks[0, 0].cpu().numpy()
        
        # Save results
        np.save(output_npy, mask)
        
        # Visualization
        mask_vis = Image.fromarray((mask * 255).astype(np.uint8))
        mask_vis.save(os.path.join(output_dir, f"{name}_mask.png"))
        
        # Simple visualization overlay
        vis = image_source.copy()
        vis[mask] = vis[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        Image.fromarray(vis.astype(np.uint8)).save(output_vis)
        
        meta = {
            'image_shape': mask.shape[:2],
            'boxes_xyxy': boxes_xyxy.cpu().numpy(),
            'boxes': box.cpu().numpy(),
            'logits': logits[idx:idx+1].cpu().numpy(),
            'phrases': [phrases[idx]]
        }
        with open(output_pkl, 'wb') as f:
            pickle.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--text_prompt', type=str, default="car mask")
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--sam_checkpoint', type=str, default="resources/models/sam_vit_h_4b8939.pth")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.source_path, "masks", "sam")
        
    run_segmentation(args.source_path, args.output_dir, args.text_prompt, 
                     args.box_threshold, args.text_threshold, args.sam_checkpoint, args.device)
