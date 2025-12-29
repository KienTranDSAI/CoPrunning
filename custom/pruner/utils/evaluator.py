"""
Perplexity evaluation for pruned models.

Standalone reimplementation of wanda/lib/eval.py
"""

import torch
import torch.nn as nn
from .dataset_loader import get_loaders


class PerplexityEvaluator:
    """
    Evaluates perplexity of language models on WikiText2.

    Perplexity measures how well a model predicts a test set.
    Lower perplexity = better prediction quality.

    Reference: wanda/lib/eval.py:83-129
    """

    def __init__(self, model, tokenizer):
        """
        Initialize evaluator.

        Args:
            model: Language model to evaluate
            tokenizer: Associated tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, dataset="wikitext2", device=torch.device("cuda:0")):
        """
        Evaluate perplexity on specified dataset.

        Args:
            dataset: Dataset name ('wikitext2' or 'c4')
            device: Device for computation

        Returns:
            Perplexity value (float)
        """
        print(f"Evaluating perplexity on {dataset}")

        # Load test data
        _, testenc = get_loaders(
            dataset, seed=0, seqlen=self.model.seqlen, tokenizer=self.tokenizer
        )

        # Compute perplexity
        with torch.no_grad():
            ppl = self._compute_ppl(testenc, device)

        return ppl

    def _compute_ppl(self, testenc, device):
        """
        Compute perplexity on test data.

        Perplexity = exp(average negative log likelihood)

        Reference: wanda/lib/eval.py:83-129

        Args:
            testenc: Test data with .input_ids attribute
            device: Device for computation

        Returns:
            Perplexity value (float)
        """
        # Get input IDs
        testenc = testenc.input_ids

        # Calculate number of samples
        nsamples = testenc.numel() // self.model.seqlen

        # List to store negative log likelihoods
        nlls = []
        print(f"Processing {nsamples} samples")

        # Process each sample
        for i in range(nsamples):
            if i % 50 == 0:
                print(f"Sample {i}/{nsamples}")

            # Extract sequence
            start = i * self.model.seqlen
            end = (i + 1) * self.model.seqlen
            inputs = testenc[:, start:end].to(device)
            inputs = inputs.reshape(1, self.model.seqlen)

            # Forward pass through model
            lm_logits = self.model(inputs).logits

            # Shift logits and labels for next-token prediction
            # We want to predict token i+1 from tokens 0..i
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

            # Calculate negative log likelihood for this sample
            # loss is per-token, multiply by seqlen to get total NLL
            neg_log_likelihood = loss.float() * self.model.seqlen

            # Append to list
            nlls.append(neg_log_likelihood)

        # Compute perplexity
        # PPL = exp(sum(NLLs) / total_tokens)
        # total_tokens = nsamples * seqlen
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen))

        # Clean up GPU memory
        torch.cuda.empty_cache()

        return ppl.item()
