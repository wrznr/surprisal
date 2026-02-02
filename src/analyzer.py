import torch
import numpy as np
import pandas as pd
import click
import sys
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class SurprisalExtractor:
    def __init__(self, model_name, force_cpu=False):
        self.device = self._get_device(force_cpu)
        
        click.secho(f"Loading model: {model_name}...", fg='cyan')
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            
            # --- DEBUG INFO ---
            # Print limits to help diagnose mismatches
            vocab_limit = self.model.config.vocab_size
            click.secho(f"✓ Model loaded. Vocab Size: {vocab_limit}", fg='green')
            # ------------------
            
        except Exception as e:
            click.secho(f"Error loading model: {e}", fg='red', err=True)
            sys.exit(1)

    def _get_device(self, force_cpu):
        if force_cpu: return "cpu"
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def get_surprisal(self, sentence):
        clean_sentence = str(sentence).strip()
        if not clean_sentence:
            return []

        # 1. Tokenize (get standard Python list of ints)
        inputs = self.tokenizer(clean_sentence, return_tensors=None)
        raw_ids = inputs["input_ids"]
        
        # 2. Determine Safe BOS (Beginning of Sentence) ID
        # We must check against the model's embedding size to prevent IndexError
        max_allowed_id = self.model.config.vocab_size - 1
        
        bos_id = self.tokenizer.bos_token_id
        
        # Fallback 1: Try EOS token if BOS is missing
        if bos_id is None: 
            bos_id = self.tokenizer.eos_token_id
            
        # Fallback 2: Check if the ID is actually valid for this specific model
        # If the tokenizer thinks BOS is ID 52000 but model only has 50257 rows, we must clamp it.
        if bos_id is None or bos_id > max_allowed_id:
            # Use the last valid token in the model as the start signal
            # (This is usually the EOS token anyway in correctly config'd models)
            bos_id = max_allowed_id
            
        # 3. Construct Input
        input_ids_list = [bos_id] + raw_ids
        
        # Final safety check on ALL tokens (just in case raw_ids has garbage)
        input_ids_list = [min(tid, max_allowed_id) for tid in input_ids_list]
        
        input_ids = torch.tensor([input_ids_list]).to(self.device)
        
        # 4. Model Inference
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # 5. Calculate Surprisal
        # Shift logits: Predict next token
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        neg_log_likelihoods = loss_fct(shift_logits, shift_labels)
        surprisals = (neg_log_likelihoods / np.log(2)).cpu().numpy()
        
        # 6. Aggregate Words
        word_data = []
        current_word_ids = []
        current_surprisal = 0.0
        
        for i, score in enumerate(surprisals):
            token_id = shift_labels[i].item()
            token_str = self.tokenizer.decode([token_id])
            
            # New word logic: Start of sentence OR Starts with space
            is_new_word = token_str.startswith(" ") or (i == 0)
            
            if is_new_word:
                if current_word_ids:
                    full_word = self.tokenizer.decode(current_word_ids).strip()
                    word_data.append({"word": full_word, "surprisal": current_surprisal})
                
                current_word_ids = [token_id]
                current_surprisal = score
            else:
                current_word_ids.append(token_id)
                current_surprisal += score

        if current_word_ids:
            full_word = self.tokenizer.decode(current_word_ids).strip()
            word_data.append({"word": full_word, "surprisal": current_surprisal})

        return word_data

@click.command()
@click.option('--sentence', '-s', help='Single German sentence to analyze.')
@click.option('--input-file', '-f', type=click.Path(exists=True), help='Path to .txt or .csv file.')
@click.option('--output-file', '-o', default="surprisal_output.csv", help='Output filename (default: surprisal_output.csv).')
@click.option('--column', '-c', default="sentence", help='Column name if input is CSV (default: "sentence").')
@click.option('--model', '-m', default="dbmdz/german-gpt2", help='Hugging Face model name.')
@click.option('--cpu', is_flag=True, help='Force usage of CPU.')
def main(sentence, input_file, output_file, column, model, cpu):
    """
    Calculate Surprisal for a single sentence OR a file of sentences.
    """
    if not sentence and not input_file:
        sentence = click.prompt("Please enter a sentence")

    extractor = SurprisalExtractor(model_name=model, force_cpu=cpu)

    # --- MODE 1: Single Sentence ---
    if sentence:
        result = extractor.get_surprisal(sentence)
        df = pd.DataFrame(result)
        click.echo("\n" + "="*40)
        click.echo(f"Analysis for: {sentence}")
        click.echo("="*40)
        print(df.to_string(index=False, float_format="%.4f"))
        click.echo("="*40 + "\n")

    # --- MODE 2: Batch Processing (File) ---
    elif input_file:
        click.echo(f"\nProcessing file: {input_file}")
        
        if input_file.endswith('.csv'):
            try:
                df_input = pd.read_csv(input_file)
                if column not in df_input.columns:
                    click.secho(f"Error: Column '{column}' not found. Found: {list(df_input.columns)}", fg='red')
                    return
                sentences = df_input[column].astype(str).tolist()
                ids = df_input['item_id'].tolist() if 'item_id' in df_input.columns else range(1, len(sentences)+1)
            except Exception as e:
                click.secho(f"Error reading CSV: {e}", fg='red')
                return
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            ids = range(1, len(sentences)+1)

        all_results = []
        
        for sent_id, sent in tqdm(zip(ids, sentences), total=len(sentences), desc="Calculating Surprisal"):
            word_data = extractor.get_surprisal(sent)
            
            for i, wd in enumerate(word_data):
                all_results.append({
                    "sentence_id": sent_id,
                    "word_pos": i + 1,
                    "word": wd['word'],
                    "surprisal": wd['surprisal']
                })

        final_df = pd.DataFrame(all_results)
        final_df.to_csv(output_file, index=False)
        
        click.secho(f"\n✓ Success! Processed {len(sentences)} sentences.", fg='green')
        click.secho(f"✓ Saved results to: {output_file}", fg='green')

if __name__ == "__main__":
    main()
