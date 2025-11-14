"""
name:
Raft

dataset:
ought/raft

abstract:
The Real-world annotated few-shot (RAFT) meta-benchmark of 11 real-world text
classification tasks.

languages:
english

tags:
classification, reasoning

paper:
https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ca46c1b9512a7a8315fa3c5a946e8265-Abstract-round2.html
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def raft_prompt(line, query_keys, instruction, task_name: str = None):
    query = instruction
    query += "\n".join([f"{key}: {line[key]}" for key in query_keys])
    query += "\nLabel:"
    return Doc(task_name=task_name, query=query, gold_index=0, choices=[str(line["Label"])], instruction=instruction)


def raft_ade_corpus_v2_prompt(line, task_name: str = None):
    instruction = "Label the sentence based on whether it is related to an adverse drug effect (ADE). Details are described below:\nDrugs: Names of drugs and chemicals that include brand names, trivial names, abbreviations and systematic names were annotated. Mentions of drugs or chemicals should strictly be in a therapeutic context. This category does not include the names of metabolites, reaction byproducts, or hospital chemicals (e.g. surgical equipment disinfectants).\nAdverse effect: Mentions of adverse effects include signs, symptoms, diseases, disorders, acquired abnormalities, deficiencies, organ damage or death that strictly occur as a consequence of drug intake.\nPossible labels:\n1. ADE-related\n2. not ADE-related"
    query_keys = ["Sentence"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_banking_77_prompt(line, task_name: str = None):
    instruction = "The following is a banking customer service query. Classify the query into one of the 77 categories available.\nPossible labels:\n1. Refund_not_showing_up\n2. activate_my_card\n3. age_limit\n4. apple_pay_or_google_pay\n5. atm_support\n6. automatic_top_up\n7. balance_not_updated_after_bank_transfer\n8. balance_not_updated_after_cheque_or_cash_deposit\n9. beneficiary_not_allowed\n10. cancel_transfer\n11. card_about_to_expire\n12. card_acceptance\n13. card_arrival\n14. card_delivery_estimate\n15. card_linking\n16. card_not_working\n17. card_payment_fee_charged\n18. card_payment_not_recognised\n19. card_payment_wrong_exchange_rate\n20. card_swallowed\n21. cash_withdrawal_charge\n22. cash_withdrawal_not_recognised\n23. change_pin\n24. compromised_card\n25. contactless_not_working\n26. country_support\n27. declined_card_payment\n28. declined_cash_withdrawal\n29. declined_transfer\n30. direct_debit_payment_not_recognised\n31. disposable_card_limits\n32. edit_personal_details\n33. exchange_charge\n34. exchange_rate\n35. exchange_via_app\n36. extra_charge_on_statement\n37. failed_transfer\n38. fiat_currency_support\n39. get_disposable_virtual_card\n40. get_physical_card\n41. getting_spare_card\n42. getting_virtual_card\n43. lost_or_stolen_card\n44. lost_or_stolen_phone\n45. order_physical_card\n46. passcode_forgotten\n47. pending_card_payment\n48. pending_cash_withdrawal\n49. pending_top_up\n50. pending_transfer\n51. pin_blocked\n52. receiving_money\n53. request_refund\n54. reverted_card_payment?\n55. supported_cards_and_currencies\n56. terminate_account\n57. top_up_by_bank_transfer_charge\n58. top_up_by_card_charge\n59. top_up_by_cash_or_cheque\n60. top_up_failed\n61. top_up_limits\n62. top_up_reverted\n63. topping_up_by_card\n64. transaction_charged_twice\n65. transfer_fee_charged\n66. transfer_into_account\n67. transfer_not_received_by_recipient\n68. transfer_timing\n69. unable_to_verify_identity\n70. verify_my_identity\n71. verify_source_of_funds\n72. verify_top_up\n73. virtual_card_not_working\n74. visa_or_mastercard\n75. why_verify_identity\n76. wrong_amount_of_cash_received\n77. wrong_exchange_rate_for_cash_withdrawal"
    query_keys = ["Query"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_neurips_impact_statement_risks_prompt(line, task_name: str = None):
    instruction = "Label the impact statement based on whether it mentions a harmful application of the research done in the paper. Make sure the statement is sufficient to conclude there are harmful applications of the research being done, not a past risk that this research is solving.\nPossible labels:\n1. doesn't mention a harmful application\n2. mentions a harmful application"
    query_keys = ["Impact statement", "Paper title"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_one_stop_english_prompt(line, task_name: str = None):
    instruction = "The following is an article sourced from The Guardian newspaper, and rewritten by teachers to suit three levels of adult English as Second Language (ESL) learners: elementary, intermediate, and advanced. Predict the level of the article.\nPossible labels:\n1. advanced\n2. elementary\n3. intermediate"
    query_keys = ["Article"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_overruling_prompt(line, task_name: str = None):
    instruction = "In law, an overruling sentence is a statement that nullifies a previous case decision as a precedent, by a constitutionally valid statute or a decision by the same or higher ranking court which establishes a different rule on the point of law involved. Label the sentence based on whether it is overruling or not.\nPossible labels:\n1. not overruling\n2. overruling"
    query_keys = ["Sentence"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_semiconductor_org_types_prompt(line, task_name: str = None):
    instruction = 'The dataset is a list of institutions that have contributed papers to semiconductor conferences in the last 25 years, as catalogued by IEEE and sampled randomly. The goal is to classify the institutions into one of three categories: "university", "company" or "research institute".\nPossible labels:\n1. company\n2. research institute\n3. university'
    query_keys = ["Organization name", "Paper title"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_systematic_review_inclusion_prompt(line, task_name: str = None):
    instruction = "Identify whether this paper should be included in a meta-review which includes the findings of systematic reviews on interventions designed to promote charitable donations.\nIncluded reviews should describe monetary charitable donations, assess any population of participants in any context, and be peer reviewed and written in English.\nThey should not report new data, be non-systematic reviews, consider cause-related marketing or other kinds of prosocial behaviour.\nPossible labels:\n1. included\n2. not included"
    query_keys = ["Title", "Abstract", "Journal"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_tai_safety_research_prompt(line, task_name: str = None):
    instruction = 'Transformative AI (TAI) is defined as AI that precipitates a transition comparable to (or more significant than) the agricultural or industrial revolution. Label a paper as "TAI safety research" if:\n1. The contents of the paper are directly motivated by, and substantively inform, the challenge of ensuring good outcomes for TAI,\n2. There is substantive content on AI safety, not just AI capabilities,\n3. The intended audience is the community of researchers,\n4. It meets a subjective threshold of seriousness/quality,\n5. Peer review is not required.\nPossible labels:\n1. TAI safety research\n2. not TAI safety research'
    query_keys = ["Title", "Abstract Note", "Publication Title", "Item Type", "Publication Year"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_terms_of_service_prompt(line, task_name: str = None):
    instruction = "Label the sentence from a Terms of Service based on whether it is potentially unfair. If it seems clearly unfair, mark it as potentially unfair.\nAccording to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties rights and obligations, to the detriment of the consumer.\nDetails on types of potentially unfair clauses are found below:\nThe jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract. Jurisdiction clauses giving consumers a right to bring disputes in their place of residence were marked as clearly fair, whereas clauses stating that any judicial proceeding takes a residence away were marked as clearly unfair.\nThe choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract. Clauses defining the applicable law as the law of the consumer's country of residence were marked as clearly fair. In every other case, the choice of law clause was considered as potentially unfair.\nThe limitation of liability clause stipulates that the duty to pay damages is limited or excluded, for certain kind of losses, under certain conditions. Clauses that explicitly affirm non-excludable providers' liabilities were marked as clearly fair. Clauses that reduce, limit, or exclude the liability of the service provider were marked as potentially unfair when concerning broad categories of losses or causes of them.\nThe unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clause was always considered as potentially unfair.\nThe unilateral termination clause gives provider the right to suspend and/or terminate the service and/or the contract, and sometimes details the circumstances under which the provider claims to have a right to do so.\nThe contract by using clause stipulates that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them. We always marked such clauses as potentially unfair.\nThe content removal gives the provider a right to modify/delete user's content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so.\nThe arbitration clause requires or allows the parties to resolve their disputes through an arbitration process, before the case could go to court. Clauses stipulating that the arbitration should take place in a state other then the state of consumer's residence or be based on arbiter's discretion were marked as clearly unfair. Clauses defining arbitration as fully optional were marked as clearly fair.\nPossible labels:\n1. not potentially unfair\n2. potentially unfair"
    query_keys = ["Sentence"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_tweet_eval_hate_prompt(line, task_name: str = None):
    instruction = "Label whether the following tweet contains hate speech against either immigrants or women. Hate Speech (HS) is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics.\nPossible labels:\n1. hate speech\n2. not hate speech"
    query_keys = ["Tweet"]
    return raft_prompt(line, query_keys, instruction, task_name)


def raft_twitter_complaints_prompt(line, task_name: str = None):
    instruction = "A complaint presents a state of affairs which breaches the writer\u2019s favorable expectation. Label the tweet text based on whether it contains a complaint.\nPossible labels:\n1. complaint\n2. no complaint"
    query_keys = ["Tweet text"]
    return raft_prompt(line, query_keys, instruction, task_name)


raft_ade_corpus_v2 = LightevalTaskConfig(
    name="raft:ade_corpus_v2",
    prompt_function=raft_ade_corpus_v2_prompt,
    hf_repo="ought/raft",
    hf_subset="ade_corpus_v2",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_banking_77 = LightevalTaskConfig(
    name="raft:banking_77",
    prompt_function=raft_banking_77_prompt,
    hf_repo="ought/raft",
    hf_subset="banking_77",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_neurips_impact_statement_risks = LightevalTaskConfig(
    name="raft:neurips_impact_statement_risks",
    prompt_function=raft_neurips_impact_statement_risks_prompt,
    hf_repo="ought/raft",
    hf_subset="neurips_impact_statement_risks",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_one_stop_english = LightevalTaskConfig(
    name="raft:one_stop_english",
    prompt_function=raft_one_stop_english_prompt,
    hf_repo="ought/raft",
    hf_subset="one_stop_english",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_overruling = LightevalTaskConfig(
    name="raft:overruling",
    prompt_function=raft_overruling_prompt,
    hf_repo="ought/raft",
    hf_subset="overruling",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_semiconductor_org_types = LightevalTaskConfig(
    name="raft:semiconductor_org_types",
    prompt_function=raft_semiconductor_org_types_prompt,
    hf_repo="ought/raft",
    hf_subset="semiconductor_org_types",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_systematic_review_inclusion = LightevalTaskConfig(
    name="raft:systematic_review_inclusion",
    prompt_function=raft_systematic_review_inclusion_prompt,
    hf_repo="ought/raft",
    hf_subset="systematic_review_inclusion",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_tai_safety_research = LightevalTaskConfig(
    name="raft:tai_safety_research",
    prompt_function=raft_tai_safety_research_prompt,
    hf_repo="ought/raft",
    hf_subset="tai_safety_research",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_terms_of_service = LightevalTaskConfig(
    name="raft:terms_of_service",
    prompt_function=raft_terms_of_service_prompt,
    hf_repo="ought/raft",
    hf_subset="terms_of_service",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_tweet_eval_hate = LightevalTaskConfig(
    name="raft:tweet_eval_hate",
    prompt_function=raft_tweet_eval_hate_prompt,
    hf_repo="ought/raft",
    hf_subset="tweet_eval_hate",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

raft_twitter_complaints = LightevalTaskConfig(
    name="raft:twitter_complaints",
    prompt_function=raft_twitter_complaints_prompt,
    hf_repo="ought/raft",
    hf_subset="twitter_complaints",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    raft_ade_corpus_v2,
    raft_banking_77,
    raft_neurips_impact_statement_risks,
    raft_one_stop_english,
    raft_overruling,
    raft_semiconductor_org_types,
    raft_systematic_review_inclusion,
    raft_tai_safety_research,
    raft_terms_of_service,
    raft_tweet_eval_hate,
    raft_twitter_complaints,
]
