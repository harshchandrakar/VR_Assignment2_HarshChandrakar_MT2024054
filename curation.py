import pandas as pd
import os
import random
import time
import requests
import re
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ========================
# CONFIGURATION
# ========================
CONFIG = {
    "BASE_DIR": "./abo-images-small/",
    "METADATA_PATH": os.path.join("./csv_files/", "balanced_part2_18000.csv"),
    "LLAVA_VERSIONS": {
        "llava:7b": {
            "num_ctx": 2048,
            "num_gpu": 1,
            "api_endpoint": "http://localhost:11434/api/generate"
        },
        "llava:13b": {
            "num_ctx": 4096,
            "num_gpu": 2,
            "api_endpoint": "http://localhost:11434/api/generate"
        }
    },
    "SELECTED_MODEL": "llava:7b",
    "MIN_QUESTIONS_PER_IMAGE": 2,
    "MAX_QUESTIONS_PER_IMAGE": 5,
    "SAMPLE_PERCENTAGE": 0.001,
    "IMAGE_PROCESSING": {
        "target_size": (336, 336),
        "quality": 75,
        "max_retries": 3,
        "retry_delay": 10
    },
    "VISUAL_FEATURES": {
        "texture": ["smooth", "rough", "glossy", "matte", "textured", "patterned"],
        "logo_present": ["yes", "no"],
        "inscribed_text": {"type": "text"}
    },
    "METADATA_FIELDS": {
        "color": ["red", "blue", "black", "white", "green", "gray", "silver", "yellow"],
        "model_name": {"type": "text"},
        "product_type": {"type": "text"},
        "style": {"type": "text"},
        "material": ["cotton", "leather", "plastic", "nylon", "metal", "wood", "glass"],
        "fabric_type": {"type": "text"},
        "pattern": {"type": "text"},
        "item_shape": {"type": "text"}
    },
    "MAX_VISUALIZATION_SAMPLES": 4,
    "MAX_WORKERS": 1
}

# ========================
# CORE PROCESSING UTILITIES
# ========================
def optimize_image(image_path):
    """Optimize image for LLaVA processing"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(CONFIG["IMAGE_PROCESSING"]["target_size"])
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=CONFIG["IMAGE_PROCESSING"]["quality"])
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return None

def query_llava(prompt, image_b64, current_model):
    """Execute LLaVA query with enhanced timeout"""
    model_config = CONFIG["LLAVA_VERSIONS"][current_model]
    payload = {
        "model": current_model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "num_ctx": model_config["num_ctx"],
            "num_gpu": model_config["num_gpu"]
        }
    }

    for attempt in range(CONFIG["IMAGE_PROCESSING"]["max_retries"]):
        try:
            response = requests.post(
                model_config["api_endpoint"],
                json=payload,
                timeout=45
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            time.sleep(CONFIG["IMAGE_PROCESSING"]["retry_delay"])
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(CONFIG["IMAGE_PROCESSING"]["retry_delay"])
    return ""

# ========================
# ENHANCED ANALYSIS PROMPT
# ========================
ANALYSIS_PROMPT = """Analyze this product image carefully and provide:
1. Brand name (if visible)
2. Primary color (single word)
3. Main material (single word)
4. Surface texture (from: smooth, rough, glossy, matte, textured, patterned)
5. Visible pattern (from: striped, floral, geometric, solid, camouflage, polka-dot)
6. Logo presence (yes/no)
7. Inscribed text (exact wording if present)
8. Product category
9. Model name (if visible)
10. Style (single word)
11. Fabric type (single word)
12. Item shape (single word)
Format strictly as:
brand: <value>, 
color: <value>, 
material: <value>, 
texture: <value>, 
pattern: <value>, 
logo_present: <value>, 
inscribed_text: <value>, 
category: <value>,
model_name: <value>,
style: <value>,
fabric_type: <value>,
item_shape: <value>"""

# ========================
# RESPONSE PROCESSING
# ========================
def parse_llava_response(response):
    """Robust parsing with validation"""
    parsed = {}
    for pair in response.lower().split(','):
        pair = pair.strip()
        if ':' in pair:
            key, value = pair.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == "logo_present":
                value = "yes" if value in ["yes", "present"] else "no"
                
            if key == "inscribed_text":
                value = re.sub(r'[^a-zA-Z0-9 ]', '', value).strip()
                
            parsed[key] = value
    return parsed

def validate_answer(field, value):
    """Validate against allowed values"""
    value = str(value).lower()
    if field in CONFIG["VISUAL_FEATURES"]:
        allowed = CONFIG["VISUAL_FEATURES"][field]
        if isinstance(allowed, list):
            return any(v.lower() == value for v in allowed)
        return True
    
    if field in CONFIG["METADATA_FIELDS"]:
        allowed = CONFIG["METADATA_FIELDS"][field]
        if isinstance(allowed, list):
            return any(v.lower() == value for v in allowed)
        return True
        
    return False

# ========================
# QUESTION GENERATION
# ========================
def generate_visual_questions(image_b64, metadata_row, current_model):
    """Generate questions from visual analysis"""
    llava_response = query_llava(ANALYSIS_PROMPT, image_b64, current_model)
    llava_data = parse_llava_response(llava_response)
    
    qa_pairs = []
    question_templates = {
        "color": "What is the primary color?",
        "model_name": "What is the model name?",
        "product_type": "What type of product is this?",
        "style": "What is the style?",
        "material": "What material is this made of?",
        "fabric_type": "What fabric type is used?",
        "pattern": "What pattern is visible?",
        "item_shape": "What is the item shape?",
        "texture": "How would you describe the texture?",
        "logo_present": "Is there a visible logo?",
        "inscribed_text": "What text is written on the product?"
    }

    for field in question_templates:
        source_data = metadata_row if field in CONFIG["METADATA_FIELDS"] else llava_data
        answer = str(source_data.get(field, '')).strip()
        
        # Use CSV value if available and valid
        if field in CONFIG["METADATA_FIELDS"]:
            csv_value = str(metadata_row.get(field, '')).strip()
            if csv_value and csv_value.lower() not in ['', 'nan', 'none', 'not visible']:
                answer = csv_value
        
        # Skip invalid answers
        if not answer or answer.lower() in ['none', 'not visible', 'n/a', 'nan']:
            continue
            
        if not validate_answer(field, answer):
            continue
            
        qa_pairs.append({
            "Question": question_templates[field],
            "Correct_Answer": answer.split()[0].lower() if field != "inscribed_text" else answer,
            "Image_ID": metadata_row["main_image_id"],
            "Image_Path": metadata_row["image_path"]
        })

    return qa_pairs[:CONFIG["MAX_QUESTIONS_PER_IMAGE"]]

def visualize_qa_samples(df, num_samples):
    """Visualize sample QA pairs for validation"""
    try:
        if len(df) == 0 or 'Image_Path' not in df.columns:
            print("No data available for visualization")
            return
            
        num_samples = min(num_samples, len(df))
        samples = df.sample(n=num_samples)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5*num_samples))
        if num_samples == 1:
            axes = [axes]
            
        for i, (_, row) in enumerate(samples.iterrows()):
            try:
                img_path = os.path.join(CONFIG["BASE_DIR"], "images/small", row["Image_Path"])
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Q: {row['Question']}\nA: {row['Correct_Answer']}")
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading image: {str(e)}", ha='center')
                
        plt.tight_layout()
        plt.savefig("./output_csv/qa_samples.png")
        plt.close()
    except Exception as e:
        print(f"Visualization error: {str(e)}")

# ========================
# MAIN PROCESSING
# ========================
def process_image(idx, row):
    """Image processing pipeline"""
    global processed_count
    try:
        image_path = os.path.join(CONFIG["BASE_DIR"], "images/small", row["image_path"])
        if idx % 10 == 0:
            print(f"--------------- Processing image {idx} ---------------")
            
        if not os.path.exists(image_path):
            return []
            
        image_b64 = optimize_image(image_path)
        if not image_b64:
            return []
            
        return generate_visual_questions(
            image_b64, 
            row, 
            CONFIG["SELECTED_MODEL"]
        )
    except Exception as e:
        print(f"Error processing {row['image_path']}: {str(e)}")
        return []

if __name__ == "__main__":
    # Initialize counter for tracking progress
    processed_count = 0
    
    # Load and prepare metadata
    metadata = pd.read_csv(CONFIG["METADATA_PATH"])
    valid_metadata = metadata[pd.notna(metadata["image_path"])].sample(frac=CONFIG["SAMPLE_PERCENTAGE"])
    print(f"Processing {len(valid_metadata)} images")
    
    # Process images with proper indexing
    # results = []
    # for idx, (_, row) in enumerate(valid_metadata.iterrows()):
    #     results.append(process_image(idx, row))
        
    # Alternative approach using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        results = list(executor.map(
            lambda x: process_image(x[0], x[1][1]), 
            enumerate(valid_metadata.iterrows())
        ))
    
    # Flatten results and save
    all_qa = [qa for sublist in results for qa in sublist]
    df = pd.DataFrame(all_qa)
    
    if not df.empty:
        output_dir = "./output_csv"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f"{output_dir}/enhanced_vqa_dataset_1.csv", index=False)
        print(f"Generated {len(df)} QA pairs")
        # visualize_qa_samples(df, CONFIG["MAX_VISUALIZATION_SAMPLES"])
    else:
        print("No valid QA pairs generated")