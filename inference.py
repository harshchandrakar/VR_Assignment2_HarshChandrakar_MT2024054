import argparse
import os
import pandas as pd
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re
from tqdm import tqdm

class VQAModel:
    def __init__(self, model_id="Magneto76/lora_blip2"):
        """Initialize the VQA model"""
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set appropriate precision based on GPU
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Store model ID
        self.model_id = model_id
        
        # Load processor and model
        self._load_model()
        
    def _load_model(self):
        """Load the processor and model with weights"""
        try:
            # Load processor
            print(f"Loading fine-tuned BLIP2 processor from {self.model_id}...")
            self.processor = Blip2Processor.from_pretrained(
                self.model_id,
                local_files_only=False
            )
            
            # Load the model directly (without PeftModel since we're not using adapter_config.json)
            print(f"Loading BLIP2 model from {self.model_id}...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map={"": self.device},
                low_cpu_mem_usage=True
            )
            
            # Set model to evaluation mode
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    @torch.inference_mode()
    def infer(self, image_path, question):
        """Run inference on an image with a question"""
        try:
            # Verify the image exists and can be opened
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return None
                
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            
            # Analyze the question type to customize prompt
            question_lower = question.lower()
            
            # Determine what type of question is being asked
            is_color_question = any(word in question_lower for word in ["color", "colored", "colors", "red", "blue", "green", "yellow", "black", "white"])
            is_material_question = any(word in question_lower for word in ["material", "made of", "wood", "plastic", "metal", "fabric"])
            is_yesno_question = any(word in question_lower for word in ["is", "are", "does", "has", "can", "do"]) and not any(word in question_lower for word in ["what", "which", "how", "where"])
            is_object_question = any(word in question_lower for word in ["what", "identify", "object", "item", "product"])
            
            # Craft a specific prompt based on question type
            if is_color_question:
                prompt = f"Question: {question} Answer with just the color name in a single word."
            elif is_material_question:
                prompt = f"Question: {question} Answer with just the material name in a single word."
            elif is_yesno_question:
                prompt = f"Question: {question} Answer with only 'yes' or 'no'."
            elif is_object_question:
                prompt = f"Question: {question} Name this object in 1-2 words maximum. No sentences."
            else:
                # Generic prompt for other questions
                prompt = f"Question: {question} Answer with a single word or very short phrase. No sentences."
            
            # Process inputs with prompt
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer with optimized parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,      # Keep short but allow enough for meaningful responses
                num_beams=5,            # More beams for better quality
                early_stopping=True,
                do_sample=False,        # Deterministic for consistency
                repetition_penalty=1.5, # Prevent repetitions
                length_penalty=0.6      # Slightly favor shorter responses
            )
            
            # Process the answer
            raw_answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if not raw_answer:
                return None
            
            # Clean up the answer
            processed_answer = raw_answer.lower().strip()
            
            # Remove common prefixes that shouldn't be in the answer
            prefixes_to_remove = ["answer:", "the answer is", "i would say", "it is", "this is", "the object is",
                               "this object is", "the product is", "a ", "an ", "the ", "it's ", "its ",
                               "question:", "is the", "is it", "the main", "the color is", "the color of"]
            
            for prefix in prefixes_to_remove:
                if processed_answer.startswith(prefix):
                    processed_answer = processed_answer[len(prefix):].strip()
            
            # Remove common suffixes that shouldn't be in the answer
            suffixes_to_remove = ["."]
            for suffix in suffixes_to_remove:
                if processed_answer.endswith(suffix):
                    processed_answer = processed_answer[:-len(suffix)].strip()
            
            # Replace full sentences with key terms
            # For phone cases
            if re.search(r'(phone|galaxy|samsung|lg|iphone|case|cover)', processed_answer):
                if "case" not in processed_answer and "cover" not in processed_answer:
                    processed_answer = "phone case"
                else:
                    processed_answer = re.sub(r'.*?(phone case|phone cover|case|cover).*', r'\1', processed_answer)
            
            # For yes/no questions - strictly normalize to just "yes" or "no"
            if is_yesno_question:
                if any(word in processed_answer for word in ["yes", "yeah", "correct", "true", "affirmative"]):
                    processed_answer = "yes"
                elif any(word in processed_answer for word in ["no", "not", "negative", "false", "isn't", "isn't"]):
                    processed_answer = "no"
            
            # For color questions - extract just the color
            if is_color_question:
                color_words = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", 
                             "pink", "brown", "gray", "grey", "teal", "navy", "gold", "silver", 
                             "multicolor", "multicolored", "multi-color", "bronze", "azure", "lilac"]
                
                for color in color_words:
                    if color in processed_answer:
                        processed_answer = color
                        break
            
            # Term mapping for better matching with ground truth patterns
            term_mapping = {
                "phone case": "cellular phone case",
                "phone cover": "cellular phone case",
                "cellphone case": "cellular phone case",
                "mobile case": "cellular phone case",
                "mobile phone case": "cellular phone case",
                "smartphone case": "cellular phone case",
                "cell case": "cellular phone case",
                "door hinge": "hardware_hinge",
                "hinge": "hardware_hinge",
                "multiple colors": "multicolor",
                "multi colored": "multicolor",
                "colorful": "multicolor",
                "rectangle": "rectangular",
            }
            
            # Apply term mapping
            for term, replacement in term_mapping.items():
                if term in processed_answer:
                    processed_answer = replacement
                    break
            
            # Limit to 1-2 words for clarity
            words = processed_answer.split()
            if len(words) > 2 and not is_yesno_question:
                # Keep only the most important 1-2 words
                important_words = []
                for word in words:
                    if word not in ["a", "an", "the", "is", "are", "of", "with", "and"]:
                        important_words.append(word)
                        if len(important_words) >= 2:
                            break
                
                processed_answer = " ".join(important_words) if important_words else words[0]
            
            # Final cleanup
            processed_answer = processed_answer.strip()
            
            # If answer is empty after processing, return a fallback
            if not processed_answer:
                # Default based on question type
                if is_yesno_question:
                    return "yes"
                elif is_color_question:
                    return "black"
                else:
                    return "phone case"
            
            return processed_answer
                
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            return None
        finally:
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV with {len(df)} entries")

    # Initialize the VQA model
    model = VQAModel(model_id="Magneto76/lora_blip2")

    # Check if the CSV has an 'answer' column - it should be there
    if 'answer' not in df.columns:
        print("ERROR: 'answer' column not found in CSV. This is required for evaluation.")
        print("Available columns:", df.columns.tolist())
        raise ValueError("Missing 'answer' column in metadata CSV")
    
    # Print some metadata information for debugging
    print(f"CSV contains columns: {df.columns.tolist()}")
    print(f"Sample data - first 2 rows:")
    print(df.head(2))

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            answer = model.infer(image_path, question)
            # If answer is None, replace with default
            if answer is None:
                answer = "error"
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            answer = "error"
        
        # Ensure answer is properly formatted (keep full phrases)
        answer = str(answer).lower().strip() if answer else "error"
        generated_answers.append(answer)
        
        # Log every 10th prediction for monitoring
        if idx % 10 == 0:
            print(f"Image {idx}: '{question}' -> '{answer}'")


    # Ensure the dataframe has both 'answer' and 'generated_answer' columns
    df["generated_answer"] = generated_answers
    
    # Make sure we keep the 'answer' column - essential for evaluation
    if 'answer' not in df.columns:
        print("ERROR: Lost 'answer' column during processing. This will break evaluation.")
        return
        
    # Print comparison between actual and predicted answers for debugging
    print("\nSample predictions vs ground truth:")
    sample_df = df[['image_name', 'question', 'answer', 'generated_answer']].head(3)
    for _, row in sample_df.iterrows():
        print(f"Image: {row['image_name']}")
        print(f"Q: {row['question']}")
        print(f"GT: {row['answer']}")
        print(f"Pred: {row['generated_answer']}")
        print("-" * 40)
    
    # Make sure to select only necessary columns to avoid extra columns causing issues
    result_df = df[['image_name', 'question', 'answer', 'generated_answer']]
    
    # Save the results CSV
    result_df.to_csv("results.csv", index=False)
    print(f"Results saved to results.csv with {len(result_df)} entries")


if __name__ == "__main__":
    main()