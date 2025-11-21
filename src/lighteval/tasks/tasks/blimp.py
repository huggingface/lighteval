"""
name:
Blimp

dataset:
nyu-mll/blimp

abstract:
BLiMP is a challenge set for evaluating what language models (LMs) know
about major grammatical phenomena in English. BLiMP consists of 67
sub-datasets, each containing 1000 minimal pairs isolating specific
contrasts in syntax, morphology, or semantics. The data is automatically
generated according to expert-crafted grammars.

languages:
english

tags:
language-modeling

paper:
https://arxiv.org/abs/1912.00582
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# TODO: Convert to inspect-ai


def blimp_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, query="", choices=[line["sentence_good"], line["sentence_bad"]], gold_index=0)


blimp_adjunct_island = LightevalTaskConfig(
    name="blimp:adjunct_island",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="adjunct_island",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_anaphor_gender_agreement = LightevalTaskConfig(
    name="blimp:anaphor_gender_agreement",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="anaphor_gender_agreement",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_anaphor_number_agreement = LightevalTaskConfig(
    name="blimp:anaphor_number_agreement",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="anaphor_number_agreement",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_animate_subject_passive = LightevalTaskConfig(
    name="blimp:animate_subject_passive",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="animate_subject_passive",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_animate_subject_trans = LightevalTaskConfig(
    name="blimp:animate_subject_trans",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="animate_subject_trans",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_causative = LightevalTaskConfig(
    name="blimp:causative",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="causative",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_complex_NP_island = LightevalTaskConfig(
    name="blimp:complex_NP_island",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="complex_NP_island",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_coordinate_structure_constraint_complex_left_branch = LightevalTaskConfig(
    name="blimp:coordinate_structure_constraint_complex_left_branch",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="coordinate_structure_constraint_complex_left_branch",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_coordinate_structure_constraint_object_extraction = LightevalTaskConfig(
    name="blimp:coordinate_structure_constraint_object_extraction",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="coordinate_structure_constraint_object_extraction",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_1 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_2 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_irregular_1 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_irregular_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_irregular_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_irregular_2 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_irregular_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_irregular_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_with_adj_2 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_with_adj_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_with_adj_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_with_adj_irregular_1 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_with_adj_irregular_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_with_adj_irregular_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_with_adj_irregular_2 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_with_adj_irregular_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_with_adj_irregular_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_determiner_noun_agreement_with_adjective_1 = LightevalTaskConfig(
    name="blimp:determiner_noun_agreement_with_adjective_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="determiner_noun_agreement_with_adjective_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_distractor_agreement_relational_noun = LightevalTaskConfig(
    name="blimp:distractor_agreement_relational_noun",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="distractor_agreement_relational_noun",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_distractor_agreement_relative_clause = LightevalTaskConfig(
    name="blimp:distractor_agreement_relative_clause",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="distractor_agreement_relative_clause",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_drop_argument = LightevalTaskConfig(
    name="blimp:drop_argument",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="drop_argument",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_ellipsis_n_bar_1 = LightevalTaskConfig(
    name="blimp:ellipsis_n_bar_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="ellipsis_n_bar_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_ellipsis_n_bar_2 = LightevalTaskConfig(
    name="blimp:ellipsis_n_bar_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="ellipsis_n_bar_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_existential_there_object_raising = LightevalTaskConfig(
    name="blimp:existential_there_object_raising",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="existential_there_object_raising",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_existential_there_quantifiers_1 = LightevalTaskConfig(
    name="blimp:existential_there_quantifiers_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="existential_there_quantifiers_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_existential_there_quantifiers_2 = LightevalTaskConfig(
    name="blimp:existential_there_quantifiers_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="existential_there_quantifiers_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_existential_there_subject_raising = LightevalTaskConfig(
    name="blimp:existential_there_subject_raising",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="existential_there_subject_raising",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_expletive_it_object_raising = LightevalTaskConfig(
    name="blimp:expletive_it_object_raising",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="expletive_it_object_raising",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_inchoative = LightevalTaskConfig(
    name="blimp:inchoative",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="inchoative",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_intransitive = LightevalTaskConfig(
    name="blimp:intransitive",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="intransitive",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_irregular_past_participle_adjectives = LightevalTaskConfig(
    name="blimp:irregular_past_participle_adjectives",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="irregular_past_participle_adjectives",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_irregular_past_participle_verbs = LightevalTaskConfig(
    name="blimp:irregular_past_participle_verbs",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="irregular_past_participle_verbs",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_irregular_plural_subject_verb_agreement_1 = LightevalTaskConfig(
    name="blimp:irregular_plural_subject_verb_agreement_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="irregular_plural_subject_verb_agreement_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_irregular_plural_subject_verb_agreement_2 = LightevalTaskConfig(
    name="blimp:irregular_plural_subject_verb_agreement_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="irregular_plural_subject_verb_agreement_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_left_branch_island_echo_question = LightevalTaskConfig(
    name="blimp:left_branch_island_echo_question",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="left_branch_island_echo_question",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_left_branch_island_simple_question = LightevalTaskConfig(
    name="blimp:left_branch_island_simple_question",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="left_branch_island_simple_question",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_matrix_question_npi_licensor_present = LightevalTaskConfig(
    name="blimp:matrix_question_npi_licensor_present",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="matrix_question_npi_licensor_present",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_npi_present_1 = LightevalTaskConfig(
    name="blimp:npi_present_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="npi_present_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_npi_present_2 = LightevalTaskConfig(
    name="blimp:npi_present_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="npi_present_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_only_npi_licensor_present = LightevalTaskConfig(
    name="blimp:only_npi_licensor_present",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="only_npi_licensor_present",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_only_npi_scope = LightevalTaskConfig(
    name="blimp:only_npi_scope",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="only_npi_scope",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_passive_1 = LightevalTaskConfig(
    name="blimp:passive_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="passive_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_passive_2 = LightevalTaskConfig(
    name="blimp:passive_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="passive_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_c_command = LightevalTaskConfig(
    name="blimp:principle_A_c_command",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_c_command",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_case_1 = LightevalTaskConfig(
    name="blimp:principle_A_case_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_case_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_case_2 = LightevalTaskConfig(
    name="blimp:principle_A_case_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_case_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_domain_1 = LightevalTaskConfig(
    name="blimp:principle_A_domain_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_domain_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_domain_2 = LightevalTaskConfig(
    name="blimp:principle_A_domain_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_domain_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_domain_3 = LightevalTaskConfig(
    name="blimp:principle_A_domain_3",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_domain_3",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_principle_A_reconstruction = LightevalTaskConfig(
    name="blimp:principle_A_reconstruction",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="principle_A_reconstruction",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_regular_plural_subject_verb_agreement_1 = LightevalTaskConfig(
    name="blimp:regular_plural_subject_verb_agreement_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="regular_plural_subject_verb_agreement_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_regular_plural_subject_verb_agreement_2 = LightevalTaskConfig(
    name="blimp:regular_plural_subject_verb_agreement_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="regular_plural_subject_verb_agreement_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_sentential_negation_npi_licensor_present = LightevalTaskConfig(
    name="blimp:sentential_negation_npi_licensor_present",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="sentential_negation_npi_licensor_present",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_sentential_negation_npi_scope = LightevalTaskConfig(
    name="blimp:sentential_negation_npi_scope",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="sentential_negation_npi_scope",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_sentential_subject_island = LightevalTaskConfig(
    name="blimp:sentential_subject_island",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="sentential_subject_island",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_superlative_quantifiers_1 = LightevalTaskConfig(
    name="blimp:superlative_quantifiers_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="superlative_quantifiers_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_superlative_quantifiers_2 = LightevalTaskConfig(
    name="blimp:superlative_quantifiers_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="superlative_quantifiers_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_tough_vs_raising_1 = LightevalTaskConfig(
    name="blimp:tough_vs_raising_1",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="tough_vs_raising_1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_tough_vs_raising_2 = LightevalTaskConfig(
    name="blimp:tough_vs_raising_2",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="tough_vs_raising_2",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_transitive = LightevalTaskConfig(
    name="blimp:transitive",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="transitive",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_island = LightevalTaskConfig(
    name="blimp:wh_island",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_island",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_questions_object_gap = LightevalTaskConfig(
    name="blimp:wh_questions_object_gap",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_questions_object_gap",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_questions_subject_gap = LightevalTaskConfig(
    name="blimp:wh_questions_subject_gap",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_questions_subject_gap",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_questions_subject_gap_long_distance = LightevalTaskConfig(
    name="blimp:wh_questions_subject_gap_long_distance",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_questions_subject_gap_long_distance",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_vs_that_no_gap = LightevalTaskConfig(
    name="blimp:wh_vs_that_no_gap",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_vs_that_no_gap",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_vs_that_no_gap_long_distance = LightevalTaskConfig(
    name="blimp:wh_vs_that_no_gap_long_distance",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_vs_that_no_gap_long_distance",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_vs_that_with_gap = LightevalTaskConfig(
    name="blimp:wh_vs_that_with_gap",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_vs_that_with_gap",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

blimp_wh_vs_that_with_gap_long_distance = LightevalTaskConfig(
    name="blimp:wh_vs_that_with_gap_long_distance",
    prompt_function=blimp_prompt,
    hf_repo="nyu-mll/blimp",
    hf_subset="wh_vs_that_with_gap_long_distance",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    blimp_adjunct_island,
    blimp_anaphor_gender_agreement,
    blimp_anaphor_number_agreement,
    blimp_animate_subject_passive,
    blimp_animate_subject_trans,
    blimp_causative,
    blimp_complex_NP_island,
    blimp_drop_argument,
    blimp_ellipsis_n_bar_1,
    blimp_ellipsis_n_bar_2,
    blimp_existential_there_object_raising,
    blimp_inchoative,
    blimp_intransitive,
    blimp_irregular_past_participle_adjectives,
    blimp_irregular_past_participle_verbs,
    blimp_only_npi_scope,
    blimp_passive_1,
    blimp_passive_2,
    blimp_principle_A_c_command,
    blimp_principle_A_reconstruction,
    blimp_regular_plural_subject_verb_agreement_1,
    blimp_regular_plural_subject_verb_agreement_2,
    blimp_sentential_negation_npi_licensor_present,
    blimp_sentential_negation_npi_scope,
    blimp_sentential_subject_island,
    blimp_superlative_quantifiers_1,
    blimp_superlative_quantifiers_2,
    blimp_tough_vs_raising_1,
    blimp_tough_vs_raising_2,
    blimp_transitive,
    blimp_wh_island,
    blimp_wh_questions_object_gap,
    blimp_wh_questions_subject_gap,
    blimp_wh_questions_subject_gap_long_distance,
    blimp_wh_vs_that_no_gap,
    blimp_wh_vs_that_no_gap_long_distance,
    blimp_wh_vs_that_with_gap,
    blimp_wh_vs_that_with_gap_long_distance,
]
