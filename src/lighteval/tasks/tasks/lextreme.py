"""
abstract:
LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain

languages:
bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv

tags:
classification, legal

paper:
https://arxiv.org/abs/2301.13126
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


lextreme_brazilian_court_decisions_judgment = LightevalTaskConfig(
    name="lextreme:brazilian_court_decisions_judgment",
    suite=["lighteval"],
    prompt_function=prompt.lextreme_brazilian_court_decisions_judgment,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_brazilian_court_decisions_unanimity,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_covid19_emergency_event,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_german_argument_mining,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_greek_legal_code_chapter,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_greek_legal_code_subject,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_greek_legal_code_volume,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_greek_legal_ner,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_legalnero,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_lener_br,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_mapa_coarse,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_mapa_fine,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_multi_eurlex_level_1,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_multi_eurlex_level_2,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_multi_eurlex_level_3,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_online_terms_of_service_clause_topics,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_online_terms_of_service_unfairness_levels,
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
    suite=["lighteval"],
    prompt_function=prompt.lextreme_swiss_judgment_prediction,
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
