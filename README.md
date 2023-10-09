# GPT-2 Language Model Evaluation
This repository contains the following key components:

- `A`: This script pre-processes your dialogue dataset, converting it into a format suitable for fine-tuning.
- `C`: Use this script to split your pre-processed dataset into training and validation datasets.
- `B`: The main script for fine-tuning your GPT-2 model and evaluating its performance.
- `intents\`: This directory includes examples and sample data used during the evaluation process.
- `README.md`: You are reading it right now! It provides an overview of the repository and its contents.

## Getting Started

1. Clone the Repository:

   ```sh
   git clone https://github.com/ana-bharadwaj/gpt2-evaluation.git
   ```

2. Navigate to the Project Directory:

   ```sh
   cd gpt2-evaluation
   ```

3. Review the Code:

   Take a closer look at the provided Python scripts (`A.py`, `C.py`, and `B.py`) to understand the fine-tuning and evaluation process. Customize these scripts if needed for your specific use case.

4. Prepare Your Data:

   Prepare your dialogue dataset as described in the `A.py` script. Ensure it's in the right format for fine-tuning.

5. Split Your Data:

   If not done already, use the `C.py` script to divide your dataset into training and validation sets.

6. Fine-Tune Your Model:

   Utilize the `B.py` script to fine-tune the GPT-2 model on your data. This script saves the best-performing model and tokenizer.

7. Evaluate the Model:

   Review the evaluation metrics, including evaluation loss and perplexity, to assess your model's performance.


## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers) for providing the transformers library.
- NLP datasets and benchmarks used for evaluation.
