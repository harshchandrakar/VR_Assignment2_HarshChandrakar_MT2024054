import pandas as pd
import os
import time
from PIL import Image
import matplotlib.pyplot as plt

# ========================
# VISUALIZATION
# ========================
def visualize_qa_samples(df, num_images=None):
    """Display images with their associated questions"""
    if df.empty:
        print("No samples available")
        return
    
    unique_images = df['Image_Path'].unique()
    max_samples = num_images if num_images is not None else CONFIG["MAX_VISUALIZATION_SAMPLES"]
    samples = unique_images[:min(len(unique_images), max_samples)]
    
    for img_path in samples:
        img_questions = df[df['Image_Path'] == img_path]
        
        plt.figure(figsize=(10, 8))
        
        try:
            full_img_path = os.path.join(CONFIG["BASE_DIR"], "images/small", img_path)
            img = Image.open(full_img_path)
            plt.imshow(img)
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Image unavailable: {str(e)}", ha='center', va='center')
            plt.axis('off')
        
        img_id = img_questions.iloc[0]['Image_ID'] if not img_questions.empty else "Unknown"
        plt.title(f"Image ID: {img_id}\nPath: {img_path}", fontsize=12)
        
        qa_text = "\n\n".join([
            f"Q{i+1}: {row['Question']}\nA: {row['Correct_Answer']}"
            for i, (_, row) in enumerate(img_questions.iterrows())
        ])
        
        plt.figtext(0.5, 0.02, qa_text, 
                   ha="center", fontsize=10, family='monospace',
                   bbox={"facecolor":"white", "alpha":0.9, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        plt.show()
        time.sleep(0.5)

if __name__ == "__main__":

    path = "./output_csv/"
    df = pd.read_csv(path)
    visualize_qa_samples(df,4)