# Multimodal Visual Question Answering with Amazon Berkeley Objects Dataset

**Course:** AIM825 Course Project
**Institution:** International Institute of Information Technology, Bangalore
**Team Members:**
* Harsh Kumar Chandrakar (MT2024054)
* Akshay Sharma (MT2024016)
* Shashank Vyas (MT2024141)

## 1. Overview

This project focuses on developing a Visual Question Answering (VQA) system tailored to the Amazon Berkeley Objects (ABO) dataset. The core contributions include the creation of a custom VQA dataset from ABO images with single-word answers, evaluation of a baseline BLIP-2 model, and subsequent fine-tuning of BLIP-2 using Low-Rank Adaptation (LoRA) to enhance its performance on this specific task. The system aims to understand an image and a natural language question about it, generating a concise, single-word answer.

## 2. Key Features

* **Custom VQA Dataset Creation:** A new VQA dataset was generated using the ABO (small variant) images, focusing on questions that elicit single-word answers.
* **Dual-Prompting Strategy:** Utilized a novel approach combining the Gemini-2.0-Flash API (for accuracy) and a locally run LLaVA model (for descriptive questions) to generate diverse and high-quality question-answer pairs.
* **Baseline Model Evaluation:** Evaluated the zero-shot performance of the pre-trained BLIP-2 (`Salesforce/blip2-flan-t5-xl`) model on the curated dataset.
* **LoRA Fine-tuning:** Implemented Parameter-Efficient Fine-Tuning (PEFT) using LoRA to adapt the BLIP-2 model to the custom VQA task, optimizing for performance on limited compute resources (Kaggle GPUs).
* **Comprehensive Evaluation:** Assessed model performance using multiple metrics: Exact Match (EM), Partial Match (PM), BLEU Score, and BERTScore F1.
* **Inference Script:** Developed a script to load the fine-tuned model and perform VQA on new image-question pairs.

## 3. Dataset

### 3.1. Source Dataset
* **Name:** Amazon Berkeley Objects (ABO) Dataset (Small Variant)
* **Contents:** Product listings with multilingual metadata and catalog images. Used for its rich collection of e-commerce product images.

### 3.2. Curated VQA Dataset
* **Filename:** `simplified_vqa_dataset.csv`
* **Format:** CSV file with the following columns:
    * `image_path`: Path or identifier for the image from the ABO dataset.
    * `generated_question`: The natural language question generated about the image.
    * `generated_answer`: The single-word answer generated for the question.
* **Generation Process:**
    * Detailed in the `data_curation.ipynb` notebook.
    * Employed a dual-prompting strategy with Gemini-2.0-Flash API and a local LLaVA model.
    * Focused on generating single-word answers.
* **Data Splitting:** The dataset was split into training, validation, and test sets using the `data_split.py` script. Data balancing techniques were also explored.

## 4. Models

* **Data Curation Models:**
    * Gemini-2.0-Flash API (Google)
    * LLaVA (Locally run)
* **VQA Models:**
    * **Baseline:** BLIP-2 (`Salesforce/blip2-flan-t5-xl`)
    * **Fine-tuned:** BLIP-2 (`Salesforce/blip2-flan-t5-xl`) adapted with LoRA.

## 5. Methodology

### 5.1. Baseline Evaluation
* The pre-trained BLIP-2 model (`Salesforce/blip2-flan-t5-xl`) was evaluated in a zero-shot setting on the test split of the `simplified_vqa_dataset.csv`.
* Notebook: `baseline_blip2.ipynb`

### 5.2. LoRA Fine-tuning
* The BLIP-2 model was fine-tuned using Low-Rank Adaptation (LoRA) to adapt it to the curated VQA task.
* Notebook: `lora-blip2_finetuned.ipynb`
* **LoRA Configuration (PeftConfig):**
    * Rank (`r`): 4
    * Alpha (`lora_alpha`): 16
    * Target Modules: `["q", "v"]` (query and value projection layers)
    * Dropout (`lora_dropout`): 0.05
    * Bias: "none"
    * Task Type: `TaskType.SEQ_2_SEQ_LM`
    * Modules to Save: `["lm_head"]`
* **Training Hyperparameters & Setup:**
    * Base Model Loading: Optimized for memory with `torch_dtype=torch.float16`, `device_map="balanced"`, `offload_folder`, `offload_state_dict=True`, `low_cpu_mem_usage=True`.
    * Optimizer: AdamW
    * Learning Rate: 5e-6
    * Weight Decay: 0.01
    * AdamW Epsilon ($\epsilon$): 1e-8
    * Number of Training Epochs: 2
    * Batch Size (per device, DataLoader): 4
    * Gradient Accumulation Steps: 2 (Effective batch size: 8)
    * Scheduler: Linear schedule with warmup (10 warmup steps)
    * Mixed-Precision Training (fp16): Enabled
    * Training Data: 18,000 samples from the training split.

## 6. Results

The performance of the baseline and LoRA fine-tuned BLIP-2 models was evaluated on the test set:

| Model                    | Exact Match (%) | Partial Match (%) | BLEU Score | BERTScore F1 |
| :----------------------- | :-------------: | :---------------: | :--------: | :----------: |
| Baseline BLIP-2          |      31.45      |       45.98       |   0.4358   |    0.8862    |
| Fine-tuned BLIP-2 (LoRA) |      34.45      |       47.42       |   0.0880   |    0.6766    |

**Discussion:**
LoRA fine-tuning led to improvements in Exact Match and Partial Match, indicating better lexical precision for single-word answers. However, a notable decrease was observed in BLEU Score and BERTScore F1. This suggests that while the fine-tuned model became more adept at predicting the exact single-word answers from our curated dataset, it might have lost some of the baseline model's generality, impacting metrics sensitive to n-gram overlap or broader semantic similarity for answers that are not exact lexical matches.

## 7. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `torch`, `transformers`, `peft`, `pandas`, `nltk`, `scikit-learn`, `Pillow`, `tqdm`, `accelerate`. Ensure CUDA is set up correctly if using GPUs.

## 8. Usage / Running the Code

### 8.1. Data Curation
* Open and run the `notebooks/data_curation.ipynb` notebook.
* This notebook details the process of generating questions and answers using Gemini API and LLaVA, and creating the `simplified_vqa_dataset.csv`.
* Ensure you have API keys set up for Gemini if you are re-running the generation.
* The `scripts/data_split.py` can be used to split the dataset.

### 8.2. Baseline Evaluation
* Open and run the `notebooks/baseline_blip2.ipynb` notebook.
* This will load the pre-trained BLIP-2 model and evaluate its performance on the test set.
* Results (metrics and predictions) are typically saved to files in the `results/` directory.

### 8.3. LoRA Fine-tuning
* Open and run the `notebooks/lora-blip2_finetuned.ipynb` notebook.
* This notebook handles:
    * Loading the base BLIP-2 model.
    * Setting up the LoRA configuration.
    * Training the model on the curated dataset.
    * Saving the fine-tuned LoRA adapter (e.g., to `/kaggle/working/blip2-lora-final` or a local `saved_models/` directory).
* This process requires a GPU and was performed on Kaggle.

### 8.4. Inference
* The `scripts/inference.py` script is used to perform VQA on new images using the fine-tuned model.
* **Example Usage:**
    ```bash
    python scripts/inference.py --image_path "/path/to/your/data" --csv_path "/path/to/your/csv/file""
    ```
    * `--image_path`: Path to the imag folder.
    * `--csv_path`: Path to the metadat.csv.

* The script will load the base model, apply the LoRA adapter, process the input, and print the generated answer.
* The inference script in the fine-tuning notebook (`lora-blip2_finetuned.ipynb`) also contains extensive post-processing logic for refining answers.

## 9. Future Work

(Refer to Section 6 of `report/VR_Final_Project_Report.pdf` for a detailed list)
* Expand dataset with multiple-choice options.
* Experiment with refined LoRA configurations.
* Conduct more in-depth qualitative error analysis.
* Explore alternative base models.
* Adapt the system for multi-word answers.

## 10. References

A full list of references is available in the project report (`report/VR_Final_Project_Report.pdf`). Key references include:
* BLIP-2: Li, J., et al. (2023).
* LoRA: Hu, E. J., et al. (2021).
* Amazon Berkeley Objects (ABO) Dataset.
* Hugging Face PEFT Library.

## 11. Acknowledgements
This project was undertaken as part of the AIM825 course at the International Institute of Information Technology, Bangalore.
