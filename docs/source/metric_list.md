# Metrics

## Metrics for multiple choice tasks
These metrics use log-likelihood of the different possible targets.
- `loglikelihood_acc` (Harness): Fraction of instances where the choice with the best logprob was correct - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_acc_single_token`)
- `loglikelihood_acc_norm` (Harness): Fraction of instances where the choice with the best logprob, normalized by sequence length, was correct - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_acc_norm_single_token`)
- `loglikelihood_acc_norm_nospace` (Harness): Fraction of instances where the choice with the best logprob, normalized by sequence length, was correct, with the first space ignored
- `loglikelihood_f1` (Harness): Corpus level F1 score of the multichoice selection - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_f1_single_token`)
- `mcc` (Harness): Matthew's correlation coefficient (a measure of agreement between statistical distributions),
- `recall_at_1` (Harness): Fraction of instances where the choice with the best logprob was correct - also exists in a faster version for tasks where the possible choices include only one token per choice (`recall_at_1_single_token`)
- `recall_at_2` (Harness): Fraction of instances where the choice with the 2nd best logprob or better was correct  - also exists in a faster version for tasks where the possible choices include only one token per choice (`recall_at_2_single_token`)
- `mrr` (Harness): Mean reciprocal rank, a measure of the quality of a ranking of choices ordered by correctness/relevance  - also exists in a faster version for tasks where the possible choices include only one token (`mrr_single_token`)
- `target_perplexity` (Harness): Perplexity of the different choices available.
- `acc_golds_likelihood`: (Harness): A bit different, it actually checks if the average logprob of a single target is above or below 0.5
- `multi_f1_numeric`: Loglikelihood F1 score for multiple gold targets

All these metrics also exist in a "single token" version (`loglikelihood_acc_single_token`, `loglikelihood_acc_norm_single_token`, `loglikelihood_f1_single_token`, `mcc_single_token`, `recall@2_single_token` and `mrr_single_token`). When the multichoice option compares only one token (ex: "A" vs "B" vs "C" vs "D", or "yes" vs "no"), using these metrics in the single token version will divide the time spent by the number of choices. Single token evals also include:
- `multi_f1_numeric` (Harness, for CB): computes the f1 score of all possible choices and averages it.

## Metrics for perplexity and language modeling
These metrics use log-likelihood of prompt.
- `word_perplexity` (Harness): Perplexity (log probability of the input) weighted by the number of words of the sequence.
- `byte_perplexity` (Harness): Perplexity (log probability of the input) weighted by the number of bytes of the sequence.
- `bits_per_byte` (HELM): Average number of bits per byte according to model probabilities.
- `log_prob` (HELM): Predicted output's average log probability (input's log prob for language modeling).

## Metrics for generative tasks
These metrics need the model to generate an output. They are therefore slower.
- Base:
    - `perfect_exact_match` (Harness): Fraction of instances where the prediction matches the gold exactly.
    - `exact_match` (HELM): Fraction of instances where the prediction matches the gold with the exception of the border whitespaces (= after a `strip` has been applied to both).
    - `quasi_exact_match` (HELM): Fraction of instances where the normalized prediction matches the normalized gold (normalization done on whitespace, articles, capitalization, ...). Other variations exist, with other normalizers, such as `quasi_exact_match_triviaqa`, which only normalizes the predictions after applying a strip to all sentences.
    - `prefix_exact_match` (HELM): Fraction of instances where the beginning of the prediction matches the gold at the exception of the border whitespaces (= after a `strip` has been applied to both).
    - `prefix_quasi_exact_match` (HELM): Fraction of instances where the normalized beginning of the prediction matches the normalized gold (normalization done on whitespace, articles, capitalization, ...)
    - `exact_match_indicator`: Exact match with some preceding context (before an indicator) removed
    - `f1_score_quasi` (HELM): Average F1 score in terms of word overlap between the model output and gold, with both being normalized first
    - `f1_score`:  Average F1 score in terms of word overlap between the model output and gold without normalisation
    - `f1_score_macro`: Corpus level macro F1 score
    - `f1_score_macro`: Corpus level micro F1 score
    - `maj_at_5` and `maj_at_8`: Model majority vote. Takes n (5 or 8) generations from the model and assumes the most frequent is the actual prediction.
- Summarization:
    - `rouge` (Harness): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/)
    - `rouge1` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on 1-gram overlap.
    - `rouge2` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on 2-gram overlap.
    - `rougeL` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on longest common subsequence overlap.
    - `rougeLsum` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on longest common subsequence overlap.
    - `rouge_t5` (BigBench): Corpus level ROUGE score for all available ROUGE metrics
    - `faithfulness` (HELM): Faithfulness scores based on the SummaC method of [Laban et al. (2022)](https://aclanthology.org/2022.tacl-1.10/).
    - `extractiveness` (HELM): Reports, based on [(Grusky et al., 2018)](https://aclanthology.org/N18-1065/)
        - `summarization_coverage`: Extent to which the model-generated summaries are extractive fragments from the source document,
        - `summarization_density`: Extent to which the model-generated summaries are extractive summaries based on the source document,
        - `summarization_compression`: Extent to which the model-generated summaries are compressed relative to the source document.
    - `bert_score` (HELM): Reports the average BERTScore precision, recall, and f1 score [(Zhang et al., 2020)](https://openreview.net/pdf?id=SkeHuCVFDr) between model generation and gold summary.
    - Translation
    - `bleu`: Corpus level BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) - uses the sacrebleu implementation.
    - `bleu_1` (HELM): Average sample BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) based on 1-gram overlap - uses the nltk implementation.
    - `bleu_4` (HELM): Average sample BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) based on 4-gram overlap - uses the nltk implementation.
    - `chrf` (Harness): Character n-gram matches f-score.
    - `ter` (Harness): Translation edit/error rate.
- Copyright
    - `copyright` (HELM): Reports:
        - `longest_common_prefix_length`: average length of longest common prefix between model generation and reference,
        - `edit_distance`: average Levenshtein edit distance between model generation and reference,
        - `edit_similarity`: average Levenshtein edit similarity (normalized by length of longer sequence) between model generation and reference.
- Math:
    - `quasi_exact_match_math` (HELM): Fraction of instances where the normalized prediction matches the normalized gold (normalization done for math, where latex symbols, units, etc are removed)
    - `maj_at_4_math` (Lighteval): Majority choice evaluation, using the math normalisation for the predictions and gold
    - `quasi_exact_match_gsm8k` (Harness): Fraction of instances where the normalized prediction matches the normalized gold (normalization done for gsm8k, where latex symbols, units, etc are removed)
    - `maj_at_8_gsm8k` (Lighteval): Majority choice evaluation, using the gsm8k normalisation for the predictions and gold
- LLM-as-Judge:
    - `llm_judge_gpt3p5`: Can be used for any generative task, the model will be scored by a GPT3.5 model using the openai API
    - `llm_judge_llama_3_405b`: Can be used for any generative task, the model will be scored by a Llama 3.405B model using the openai API
    - `llm_judge_multi_turn_gpt3p5`: Can be used for any generative task, the model will be scored by a GPT3.5 model using the openai API. It is used for multiturn tasks like mt-bench.
    - `llm_judge_multi_turn_llama_3_405b`: Can be used for any generative task, the model will be scored by a Llama 3.405B model using the openai API. It is used for multiturn tasks like mt-bench.
