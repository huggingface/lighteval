"""
name:
Wikifact

dataset:
lighteval/wikifact

abstract:
Extensively test factual knowledge.

languages:
english

tags:
factuality, knowledge

paper:
https://aclanthology.org/D19-1250/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def wikifact_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, query=f"{line['question']} ", gold_index=0, choices=[line["references"]])


wikifact_applies_to_jurisdiction = LightevalTaskConfig(
    name="wikifact:applies_to_jurisdiction",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="applies_to_jurisdiction",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_atomic_number = LightevalTaskConfig(
    name="wikifact:atomic_number",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="atomic_number",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_author = LightevalTaskConfig(
    name="wikifact:author",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="author",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_award_received = LightevalTaskConfig(
    name="wikifact:award_received",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="award_received",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_basic_form_of_government = LightevalTaskConfig(
    name="wikifact:basic_form_of_government",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="basic_form_of_government",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_capital = LightevalTaskConfig(
    name="wikifact:capital",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="capital",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_capital_of = LightevalTaskConfig(
    name="wikifact:capital_of",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="capital_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_central_bank = LightevalTaskConfig(
    name="wikifact:central_bank",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="central_bank",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_composer = LightevalTaskConfig(
    name="wikifact:composer",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="composer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_continent = LightevalTaskConfig(
    name="wikifact:continent",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="continent",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_country = LightevalTaskConfig(
    name="wikifact:country",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="country",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_country_of_citizenship = LightevalTaskConfig(
    name="wikifact:country_of_citizenship",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="country_of_citizenship",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_country_of_origin = LightevalTaskConfig(
    name="wikifact:country_of_origin",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="country_of_origin",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_creator = LightevalTaskConfig(
    name="wikifact:creator",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="creator",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_currency = LightevalTaskConfig(
    name="wikifact:currency",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="currency",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_defendant = LightevalTaskConfig(
    name="wikifact:defendant",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="defendant",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_developer = LightevalTaskConfig(
    name="wikifact:developer",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="developer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_diplomatic_relation = LightevalTaskConfig(
    name="wikifact:diplomatic_relation",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="diplomatic_relation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_director = LightevalTaskConfig(
    name="wikifact:director",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="director",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_discoverer_or_inventor = LightevalTaskConfig(
    name="wikifact:discoverer_or_inventor",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="discoverer_or_inventor",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_drug_or_therapy_used_for_treatment = LightevalTaskConfig(
    name="wikifact:drug_or_therapy_used_for_treatment",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="drug_or_therapy_used_for_treatment",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_educated_at = LightevalTaskConfig(
    name="wikifact:educated_at",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="educated_at",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_electron_configuration = LightevalTaskConfig(
    name="wikifact:electron_configuration",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="electron_configuration",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_employer = LightevalTaskConfig(
    name="wikifact:employer",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="employer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_field_of_work = LightevalTaskConfig(
    name="wikifact:field_of_work",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="field_of_work",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_file_extension = LightevalTaskConfig(
    name="wikifact:file_extension",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="file_extension",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_genetic_association = LightevalTaskConfig(
    name="wikifact:genetic_association",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="genetic_association",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_genre = LightevalTaskConfig(
    name="wikifact:genre",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="genre",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_has_part = LightevalTaskConfig(
    name="wikifact:has_part",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="has_part",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_head_of_government = LightevalTaskConfig(
    name="wikifact:head_of_government",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="head_of_government",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_head_of_state = LightevalTaskConfig(
    name="wikifact:head_of_state",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="head_of_state",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_headquarters_location = LightevalTaskConfig(
    name="wikifact:headquarters_location",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="headquarters_location",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_industry = LightevalTaskConfig(
    name="wikifact:industry",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="industry",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_influenced_by = LightevalTaskConfig(
    name="wikifact:influenced_by",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="influenced_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_instance_of = LightevalTaskConfig(
    name="wikifact:instance_of",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="instance_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_instrument = LightevalTaskConfig(
    name="wikifact:instrument",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="instrument",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_language_of_work_or_name = LightevalTaskConfig(
    name="wikifact:language_of_work_or_name",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="language_of_work_or_name",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_languages_spoken_written_or_signed = LightevalTaskConfig(
    name="wikifact:languages_spoken_written_or_signed",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="languages_spoken_written_or_signed",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_laws_applied = LightevalTaskConfig(
    name="wikifact:laws_applied",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="laws_applied",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_located_in_the_administrative_territorial_entity = LightevalTaskConfig(
    name="wikifact:located_in_the_administrative_territorial_entity",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="located_in_the_administrative_territorial_entity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_location = LightevalTaskConfig(
    name="wikifact:location",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="location",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_location_of_discovery = LightevalTaskConfig(
    name="wikifact:location_of_discovery",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="location_of_discovery",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_location_of_formation = LightevalTaskConfig(
    name="wikifact:location_of_formation",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="location_of_formation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_majority_opinion_by = LightevalTaskConfig(
    name="wikifact:majority_opinion_by",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="majority_opinion_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_manufacturer = LightevalTaskConfig(
    name="wikifact:manufacturer",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="manufacturer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_measured_physical_quantity = LightevalTaskConfig(
    name="wikifact:measured_physical_quantity",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="measured_physical_quantity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_medical_condition_treated = LightevalTaskConfig(
    name="wikifact:medical_condition_treated",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="medical_condition_treated",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_member_of = LightevalTaskConfig(
    name="wikifact:member_of",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="member_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_member_of_political_party = LightevalTaskConfig(
    name="wikifact:member_of_political_party",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="member_of_political_party",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_member_of_sports_team = LightevalTaskConfig(
    name="wikifact:member_of_sports_team",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="member_of_sports_team",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_movement = LightevalTaskConfig(
    name="wikifact:movement",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="movement",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_named_after = LightevalTaskConfig(
    name="wikifact:named_after",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="named_after",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_native_language = LightevalTaskConfig(
    name="wikifact:native_language",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="native_language",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_number_of_processor_cores = LightevalTaskConfig(
    name="wikifact:number_of_processor_cores",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="number_of_processor_cores",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_occupation = LightevalTaskConfig(
    name="wikifact:occupation",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="occupation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_office_held_by_head_of_government = LightevalTaskConfig(
    name="wikifact:office_held_by_head_of_government",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="office_held_by_head_of_government",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_office_held_by_head_of_state = LightevalTaskConfig(
    name="wikifact:office_held_by_head_of_state",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="office_held_by_head_of_state",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_official_language = LightevalTaskConfig(
    name="wikifact:official_language",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="official_language",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_operating_system = LightevalTaskConfig(
    name="wikifact:operating_system",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="operating_system",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_original_language_of_film_or_TV_show = LightevalTaskConfig(
    name="wikifact:original_language_of_film_or_TV_show",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="original_language_of_film_or_TV_show",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_original_network = LightevalTaskConfig(
    name="wikifact:original_network",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="original_network",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_overrules = LightevalTaskConfig(
    name="wikifact:overrules",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="overrules",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_owned_by = LightevalTaskConfig(
    name="wikifact:owned_by",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="owned_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_part_of = LightevalTaskConfig(
    name="wikifact:part_of",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="part_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_participating_team = LightevalTaskConfig(
    name="wikifact:participating_team",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="participating_team",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_place_of_birth = LightevalTaskConfig(
    name="wikifact:place_of_birth",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="place_of_birth",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_place_of_death = LightevalTaskConfig(
    name="wikifact:place_of_death",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="place_of_death",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_plaintiff = LightevalTaskConfig(
    name="wikifact:plaintiff",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="plaintiff",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_position_held = LightevalTaskConfig(
    name="wikifact:position_held",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="position_held",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_position_played_on_team = LightevalTaskConfig(
    name="wikifact:position_played_on_team",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="position_played_on_team",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_programming_language = LightevalTaskConfig(
    name="wikifact:programming_language",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="programming_language",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_recommended_unit_of_measurement = LightevalTaskConfig(
    name="wikifact:recommended_unit_of_measurement",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="recommended_unit_of_measurement",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_record_label = LightevalTaskConfig(
    name="wikifact:record_label",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="record_label",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_religion = LightevalTaskConfig(
    name="wikifact:religion",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="religion",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_repealed_by = LightevalTaskConfig(
    name="wikifact:repealed_by",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="repealed_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_shares_border_with = LightevalTaskConfig(
    name="wikifact:shares_border_with",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="shares_border_with",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_solved_by = LightevalTaskConfig(
    name="wikifact:solved_by",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="solved_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_statement_describes = LightevalTaskConfig(
    name="wikifact:statement_describes",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="statement_describes",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_stock_exchange = LightevalTaskConfig(
    name="wikifact:stock_exchange",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="stock_exchange",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_subclass_of = LightevalTaskConfig(
    name="wikifact:subclass_of",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="subclass_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_subsidiary = LightevalTaskConfig(
    name="wikifact:subsidiary",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="subsidiary",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_symptoms_and_signs = LightevalTaskConfig(
    name="wikifact:symptoms_and_signs",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="symptoms_and_signs",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_therapeutic_area = LightevalTaskConfig(
    name="wikifact:therapeutic_area",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="therapeutic_area",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_time_of_discovery_or_invention = LightevalTaskConfig(
    name="wikifact:time_of_discovery_or_invention",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="time_of_discovery_or_invention",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_twinned_administrative_body = LightevalTaskConfig(
    name="wikifact:twinned_administrative_body",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="twinned_administrative_body",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

wikifact_work_location = LightevalTaskConfig(
    name="wikifact:work_location",
    prompt_function=wikifact_prompt,
    hf_repo="lighteval/wikifact",
    hf_subset="work_location",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    wikifact_applies_to_jurisdiction,
    wikifact_atomic_number,
    wikifact_author,
    wikifact_employer,
    wikifact_field_of_work,
    wikifact_file_extension,
    wikifact_genetic_association,
    wikifact_instrument,
    wikifact_language_of_work_or_name,
    wikifact_languages_spoken_written_or_signed,
    wikifact_laws_applied,
    wikifact_located_in_the_administrative_territorial_entity,
    wikifact_location,
    wikifact_location_of_discovery,
    wikifact_location_of_formation,
    wikifact_member_of,
    wikifact_member_of_political_party,
    wikifact_member_of_sports_team,
    wikifact_movement,
    wikifact_headquarters_location,
    wikifact_industry,
    wikifact_named_after,
    wikifact_native_language,
    wikifact_number_of_processor_cores,
    wikifact_occupation,
    wikifact_original_language_of_film_or_TV_show,
    wikifact_original_network,
    wikifact_overrules,
    wikifact_owned_by,
    wikifact_part_of,
    wikifact_participating_team,
    wikifact_place_of_birth,
    wikifact_place_of_death,
    wikifact_position_played_on_team,
    wikifact_programming_language,
    wikifact_recommended_unit_of_measurement,
    wikifact_record_label,
    wikifact_religion,
    wikifact_repealed_by,
    wikifact_shares_border_with,
    wikifact_solved_by,
    wikifact_statement_describes,
    wikifact_stock_exchange,
    wikifact_subclass_of,
    wikifact_subsidiary,
    wikifact_symptoms_and_signs,
    wikifact_therapeutic_area,
    wikifact_time_of_discovery_or_invention,
    wikifact_twinned_administrative_body,
    wikifact_work_location,
]
