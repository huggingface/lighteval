"""
name:
Lextreme

dataset:
lighteval/lextreme

abstract:
LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain

languages:
bulgarian, czech, danish, german, greek, english, spanish, estonian, finnish, french, ga, croatian, hungarian, italian, lithuanian, latvian, mt, dutch, polish, portuguese, romanian, slovak, slovenian, swedish

tags:
classification, legal

paper:
https://arxiv.org/abs/2301.13126
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def lextreme_prompt(line, instruction, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{instruction}\nPassage: {line['input']}\nAnswer: ",
        choices=line["references"],
        gold_index=[line["references"].index(item) for item in line["gold"]],
        instruction=instruction + "\n",
    )


def lextreme_brazilian_court_decisions_judgment_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given the case description "
        "from a decision heard at the State Supreme Court of Alagoas (Brazil). "
        "Predict the judgment of the case "
        "(no: The appeal was denied, "
        "partial: For partially favourable decisions, "
        "yes: For fully favourable decisions)"
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_brazilian_court_decisions_unanimity_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given the case description "
        "from a decision heard at the State Supreme Court of Alagoas (Brazil). "
        "Predict the unanimity of the case (unanimity, not-unanimity, not_determined)"
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_german_argument_mining_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given sentences from German court decisions. "
        "Predict the major component of German Urteilsstil "
        "(conclusion: Overall result, "
        "definition: Abstract legal facts and consequences, "
        "subsumption: Determination sentence / Concrete facts, "
        "other: Anything else)"
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_greek_legal_code_chapter_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a Greek legislative document. "
        "Predict the chapter level category of the "
        "'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to."
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_greek_legal_code_subject_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a Greek legislative document. "
        "Predict the subject level category of the "
        "'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to."
    )

    return lextreme_prompt(line, instruction, task_name)


def lextreme_greek_legal_code_volume_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a Greek legislative document. "
        "Predict the volume level category of the "
        "'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to."
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_swiss_judgment_prediction_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given the facts description "
        "from a decision heard at the Swiss Federal Supreme Court. "
        "Predict the judgment of the case (approval or dismissal)"
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_online_terms_of_service_unfairness_levels_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a sentence "
        "from a Terms of Service (ToS) document. "
        "Predict the unfairness level of the sentence (potentially_unfair, clearly_unfair, clearly_fair, untagged)"
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_online_terms_of_service_clause_topics_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a sentence "
        "from a Terms of Service (ToS) document. "
        "Predict the clause topics of the sentence "
        "(0: Arbitration, "
        "1: Unilateral change, "
        "2: Content removal, "
        "3: Jurisdiction, "
        "4: Choice of law, "
        "5: Limitation of liability, "
        "6: Unilateral termination, "
        "7: Contract by using, "
        "8: Privacy included)"
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_covid19_emergency_event_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a sentence from a European legislative document. "
        "Predict the applicable measurements against COVID-19 "
        "(0: State of Emergency, "
        "1: Restrictions of fundamental rights and civil liberties, "
        "2: Restrictions of daily liberties, "
        "3: Closures / lockdown, "
        "4: Suspension of international cooperation and commitments, "
        "5: Police mobilization, "
        "6: Army mobilization, "
        "7: Government oversight)"
    )

    return lextreme_prompt(line, instruction, task_name)


def lextreme_multi_eurlex_level_1_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a document from an EU law. Predict the level 1 concept in the EUROVOC taxonomy."
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_multi_eurlex_level_2_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a document from an EU law. Predict the level 2 concept in the EUROVOC taxonomy."
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_multi_eurlex_level_3_prompt(line, task_name: str = None):
    instruction = (
        "In this task, you are given a document from an EU law. Predict the level 3 concept in the EUROVOC taxonomy."
    )
    return lextreme_prompt(line, instruction, task_name)


def lextreme_greek_legal_ner_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a Greek legal document. Predict the named entities."
    return lextreme_prompt(line, instruction, task_name)


def lextreme_legalnero_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a legal text. Predict the named entities of legal interest."
    return lextreme_prompt(line, instruction, task_name)


def lextreme_lener_br_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a Brazilian legal text. Predict the named entities."
    return lextreme_prompt(line, instruction, task_name)


def lextreme_mapa_coarse_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a legal text. Predict the coarse-grained labels."
    return lextreme_prompt(line, instruction, task_name)


def lextreme_mapa_fine_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a legal text. Predict the fine-grained labels."
    return lextreme_prompt(line, instruction, task_name)


lextreme_brazilian_court_decisions_judgment = LightevalTaskConfig(
    name="lextreme:brazilian_court_decisions_judgment",
    prompt_function=lextreme_brazilian_court_decisions_judgment_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="brazilian_court_decisions_judgment",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_brazilian_court_decisions_unanimity = LightevalTaskConfig(
    name="lextreme:brazilian_court_decisions_unanimity",
    prompt_function=lextreme_brazilian_court_decisions_unanimity_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="brazilian_court_decisions_unanimity",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_covid19_emergency_event = LightevalTaskConfig(
    name="lextreme:covid19_emergency_event",
    prompt_function=lextreme_covid19_emergency_event_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="covid19_emergency_event",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_german_argument_mining = LightevalTaskConfig(
    name="lextreme:german_argument_mining",
    prompt_function=lextreme_german_argument_mining_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="german_argument_mining",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_greek_legal_code_chapter = LightevalTaskConfig(
    name="lextreme:greek_legal_code_chapter",
    prompt_function=lextreme_greek_legal_code_chapter_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="greek_legal_code_chapter",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_greek_legal_code_subject = LightevalTaskConfig(
    name="lextreme:greek_legal_code_subject",
    prompt_function=lextreme_greek_legal_code_subject_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="greek_legal_code_subject",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_greek_legal_code_volume = LightevalTaskConfig(
    name="lextreme:greek_legal_code_volume",
    prompt_function=lextreme_greek_legal_code_volume_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="greek_legal_code_volume",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_greek_legal_ner = LightevalTaskConfig(
    name="lextreme:greek_legal_ner",
    prompt_function=lextreme_greek_legal_ner_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="greek_legal_ner",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=430,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_legalnero = LightevalTaskConfig(
    name="lextreme:legalnero",
    prompt_function=lextreme_legalnero_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="legalnero",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=788,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_lener_br = LightevalTaskConfig(
    name="lextreme:lener_br",
    prompt_function=lextreme_lener_br_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="lener_br",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=338,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_mapa_coarse = LightevalTaskConfig(
    name="lextreme:mapa_coarse",
    prompt_function=lextreme_mapa_coarse_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="mapa_coarse",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=274,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_mapa_fine = LightevalTaskConfig(
    name="lextreme:mapa_fine",
    prompt_function=lextreme_mapa_fine_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="mapa_fine",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=274,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_multi_eurlex_level_1 = LightevalTaskConfig(
    name="lextreme:multi_eurlex_level_1",
    prompt_function=lextreme_multi_eurlex_level_1_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="multi_eurlex_level_1",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_multi_eurlex_level_2 = LightevalTaskConfig(
    name="lextreme:multi_eurlex_level_2",
    prompt_function=lextreme_multi_eurlex_level_2_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="multi_eurlex_level_2",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_multi_eurlex_level_3 = LightevalTaskConfig(
    name="lextreme:multi_eurlex_level_3",
    prompt_function=lextreme_multi_eurlex_level_3_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="multi_eurlex_level_3",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_online_terms_of_service_clause_topics = LightevalTaskConfig(
    name="lextreme:online_terms_of_service_clause_topics",
    prompt_function=lextreme_online_terms_of_service_clause_topics_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="online_terms_of_service_clause_topics",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_online_terms_of_service_unfairness_levels = LightevalTaskConfig(
    name="lextreme:online_terms_of_service_unfairness_levels",
    prompt_function=lextreme_online_terms_of_service_unfairness_levels_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="online_terms_of_service_unfairness_levels",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lextreme_swiss_judgment_prediction = LightevalTaskConfig(
    name="lextreme:swiss_judgment_prediction",
    prompt_function=lextreme_swiss_judgment_prediction_prompt,
    hf_repo="lighteval/lextreme",
    hf_subset="swiss_judgment_prediction",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    lextreme_brazilian_court_decisions_judgment,
    lextreme_brazilian_court_decisions_unanimity,
    lextreme_covid19_emergency_event,
    lextreme_german_argument_mining,
    lextreme_greek_legal_code_chapter,
    lextreme_greek_legal_code_subject,
    lextreme_greek_legal_code_volume,
    lextreme_greek_legal_ner,
    lextreme_legalnero,
    lextreme_lener_br,
    lextreme_mapa_coarse,
    lextreme_mapa_fine,
    lextreme_multi_eurlex_level_1,
    lextreme_multi_eurlex_level_2,
    lextreme_multi_eurlex_level_3,
    lextreme_online_terms_of_service_clause_topics,
    lextreme_online_terms_of_service_unfairness_levels,
    lextreme_swiss_judgment_prediction,
]
