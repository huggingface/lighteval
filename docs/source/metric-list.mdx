# Metric List

## Automatic metrics for multiple-choice tasks

These metrics use log-likelihood of the different possible targets.
- `loglikelihood_acc`: Fraction of instances where the choice with the best logprob was correct - we recommend using a normalization by length
- `loglikelihood_f1`: Corpus level F1 score of the multichoice selection
- `mcc`: Matthew's correlation coefficient (a measure of agreement between statistical distributions).
- `recall_at_k`: Fraction of instances where the choice with the k-st best logprob or better was correct
- `mrr`: Mean reciprocal rank, a measure of the quality of a ranking of choices ordered by correctness/relevance
- `target_perplexity`: Perplexity of the different choices available.
- `acc_golds_likelihood`: A bit different, it actually checks if the average logprob of a single target is above or below 0.5.
- `multi_f1_numeric`: Loglikelihood F1 score for multiple gold targets.

## Automatic metrics for perplexity and language modeling
These metrics use log-likelihood of prompt.
- `word_perplexity`: Perplexity (log probability of the input) weighted by the number of words of the sequence.
- `byte_perplexity`: Perplexity (log probability of the input) weighted by the number of bytes of the sequence.
- `bits_per_byte`: Average number of bits per byte according to model probabilities.
- `log_prob`: Predicted output's average log probability (input's log prob for language modeling).

## Automatic metrics for generative tasks
These metrics need the model to generate an output. They are therefore slower.
- Base:
    - `exact_match`: Fraction of instances where the prediction matches the gold. Several variations can be made through parametrization:
        - normalization on string pre-comparision on whitespace, articles, capitalization, ....
        - comparing the full string, or only subsets (prefix, suffix, ...)
    - `maj_at_k`: Model majority vote. Samples k generations from the model and assumes the most frequent is the actual prediction.
    - `f1_score`:  Average F1 score in terms of word overlap between the model output and gold (normalisation optional).
    - `f1_score_macro`: Corpus level macro F1 score.
    - `f1_score_macro`: Corpus level micro F1 score.
- Summarization:
    - `rouge`: Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/).
    - `rouge1`: Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on 1-gram overlap.
    - `rouge2`: Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on 2-gram overlap.
    - `rougeL`: Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on longest common subsequence overlap.
    - `rougeLsum`: Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on longest common subsequence overlap.
    - `rouge_t5` (BigBench): Corpus level ROUGE score for all available ROUGE metrics.
    - `faithfulness`: Faithfulness scores based on the SummaC method of [Laban et al. (2022)](https://aclanthology.org/2022.tacl-1.10/).
    - `extractiveness`: Reports, based on [(Grusky et al., 2018)](https://aclanthology.org/N18-1065/):
        - `summarization_coverage`: Extent to which the model-generated summaries are extractive fragments from the source document,
        - `summarization_density`: Extent to which the model-generated summaries are extractive summaries based on the source document,
        - `summarization_compression`: Extent to which the model-generated summaries are compressed relative to the source document.
    - `bert_score`: Reports the average BERTScore precision, recall, and f1 score [(Zhang et al., 2020)](https://openreview.net/pdf?id=SkeHuCVFDr) between model generation and gold summary.
- Translation:
    - `bleu`: Corpus level BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) - uses the sacrebleu implementation.
    - `bleu_1`: Average sample BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) based on 1-gram overlap - uses the nltk implementation.
    - `bleu_4`: Average sample BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) based on 4-gram overlap - uses the nltk implementation.
    - `chrf`: Character n-gram matches f-score.
    - `ter`: Translation edit/error rate.
- Copyright:
    - `copyright`: Reports:
        - `longest_common_prefix_length`: Average length of longest common prefix between model generation and reference,
        - `edit_distance`: Average Levenshtein edit distance between model generation and reference,
        - `edit_similarity`: Average Levenshtein edit similarity (normalized by the length of longer sequence) between model generation and reference.
- Math:
    - Both `exact_match` and `maj_at_k` can be used to evaluate mathematics tasks with math specific normalization to remove and filter latex.

## LLM-as-Judge
- `llm_judge_gpt3p5`: Can be used for any generative task, the model will be scored by a GPT3.5 model using the OpenAI API.
- `llm_judge_llama_3_405b`: Can be used for any generative task, the model will be scored by a Llama 3.405B model using the HuggingFace API.
- `llm_judge_multi_turn_gpt3p5`: Can be used for any generative task, the model will be scored by a GPT3.5 model using the OpenAI API. It is used for multiturn tasks like mt-bench.
- `llm_judge_multi_turn_llama_3_405b`: Can be used for any generative task, the model will be scored by a Llama 3.405B model using the HuggingFace API. It is used for multiturn tasks like mt-bench.
