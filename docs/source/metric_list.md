# Metrics

- MetricCategory.TARGET_PERPLEXITY
	- acc_golds_likelihood
	- target_perplexity

- MetricCategory.MULTICHOICE_ONE_TOKEN
	- loglikelihood_acc_norm_single_token
	- loglikelihood_acc_single_token
	- loglikelihood_f1_single_token
	- mcc_single_token
	- mrr_single_token
	- multi_f1_numeric
	- recall_at_1_single_token
	- recall_at_2_single_token

- MetricCategory.IGNORED
	- prediction_perplexity

- MetricCategory.PERPLEXITY
	- bits_per_byte
	- byte_perplexity
	- word_perplexity

- MetricCategory.GENERATIVE
	- bert_score
	- bleu
	- bleu_1
	- bleu_4
	- bleurt
	- chrf
	- copyright
	- drop
	- exact_match
	- extractiveness
	- f1_score_quasi
	- f1_score
	- f1_score_macro
	- f1_score_micro
	- faithfulness
	- perfect_exact_match
	- prefix_exact_match
	- prefix_quasi_exact_match
	- quasi_exact_match
	- quasi_exact_match_math
	- quasi_exact_match_triviaqa
	- quasi_exact_match_gsm8k
	- rouge_t5
	- rouge1
	- rouge2
	- rougeL
	- rougeLsum
	- ter

- MetricCategory.GENERATIVE_SAMPLING
	- maj_at_4_math
	- maj_at_5
	- maj_at_8
	- maj_at_8_gsm8k

- MetricCategory.LLM_AS_JUDGE_MULTI_TURN
	- llm_judge_multi_turn_gpt3p5
	- llm_judge_multi_turn_llama_3_405b

- MetricCategory.LLM_AS_JUDGE
	- llm_judge_gpt3p5
	- llm_judge_llama_3_405b

- MetricCategory.MULTICHOICE
	- loglikelihood_acc
	- loglikelihood_acc_norm
	- loglikelihood_acc_norm_nospace
	- loglikelihood_f1
	- mcc
	- mrr
	- recall_at_1
	- recall_at_2
	- truthfulqa_mc_metrics
