from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union

class SummarizerModel:
    """Base class for text summarization models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        
    def load(self):
        """Load the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.summarizer = pipeline(
            "summarization", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        return self

    def summarize(self, text: str, max_length: int = None, min_length: int = None, 
              ratio: float = 0.3, **kwargs) -> str:
        text = text.strip()
        if not text or len(text.split()) < 20:
            raise ValueError("Input text is too short for summarization.")
        
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < 10:
            raise ValueError("Input has too few tokens after tokenization.")
        
        text_length = len(text.split())
        max_length = max(50, int(text_length * ratio)) if max_length is None else max_length
        min_length = max(30, int(max_length * 0.3)) if min_length is None else min_length
        max_length = min(1024, max_length)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        summary_ids = self.model.generate(
            **inputs, max_length=max_length, min_length=min_length, do_sample=False, **kwargs
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary




class FineTunedSummarizer(SummarizerModel):
    """Summarizer that can be fine-tuned on custom data"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__(model_name)
        
    def load_custom(self, model_path: str):
        """Load a fine-tuned model from local path"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.summarizer = pipeline(
            "summarization", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        return self
    
    def fine_tune(self, train_texts: List[str], train_summaries: List[str], 
                  output_dir: str, epochs: int = 3, batch_size: int = 4):
        """Fine-tune the model on custom data"""
        # This is a simplified version - in practice, you'd need to implement
        # proper dataset preparation, training loop, etc.
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        from datasets import Dataset
        
        # Load model and tokenizer if not already loaded
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Prepare dataset
        train_data = {
            "text": train_texts,
            "summary": train_summaries
        }
        train_dataset = Dataset.from_dict(train_data)
        
        # Tokenize function
        def tokenize_function(examples):
            inputs = self.tokenizer(examples["text"], padding="max_length", truncation=True)
            targets = self.tokenizer(examples["summary"], padding="max_length", truncation=True)
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": targets.input_ids,
            }
        
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Update pipeline with fine-tuned model
        self.summarizer = pipeline(
            "summarization", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        return self


class SummarizerFactory:
    """Factory class to create different summarizers"""
    
    @staticmethod
    def get_summarizer(model_type: str = "bart", model_name: Optional[str] = None) -> SummarizerModel:
        """Create a summarizer based on model type
        
        Args:
            model_type: Type of model to use (bart, t5, pegasus, etc.)
            model_name: Specific model name or path
            
        Returns:
            An initialized summarizer object
        """
        if model_type == "custom" and model_name:
            # Load a custom fine-tuned model
            return FineTunedSummarizer().load_custom(model_name)
            
        # Map model types to default pre-trained models
        model_map = {
            "bart": "facebook/bart-large-cnn",
            "t5": "t5-base",
            "pegasus": "google/pegasus-xsum",
            "led": "allenai/led-base-16384",  # For longer documents
            "distilbart": "sshleifer/distilbart-cnn-12-6"  # Faster, smaller model
        }
        
        # Use provided model_name or default from map
        actual_model = model_name if model_name else model_map.get(model_type, model_map["bart"])
        
        # Create and return the appropriate summarizer
        summarizer = SummarizerModel(actual_model).load()
        return summarizer


def evaluate_summary(original_text: str, summary: str) -> Dict[str, float]:
    """Evaluate the quality of a summary"""
    # This requires additional packages: rouge_score and bert_score
    from rouge_score import rouge_scorer
    
    # ROUGE metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, summary)
    
    # Extract F1 scores
    results = {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }
    
    # Add BERT score if available
    try:
        from bert_score import score
        p, r, f1 = score([summary], [original_text], lang="en")
        results["bert_score"] = f1.item()
    except ImportError:
        pass
    
    return results


# Simple function for backward compatibility
def summarize_text(text: str, model_type: str = "bart", **kwargs) -> str:
    """Summarize text using the specified model type
    
    This function maintains the original API for backward compatibility
    """
    summarizer = SummarizerFactory.get_summarizer(model_type)
    return summarizer.summarize(text, **kwargs)