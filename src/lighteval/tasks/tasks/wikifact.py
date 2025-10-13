# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import helm_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig


wikifact_applies_to_jurisdiction_helm = LightevalTaskConfig(
    name="wikifact:applies_to_jurisdiction",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="applies_to_jurisdiction",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_atomic_number_helm = LightevalTaskConfig(
    name="wikifact:atomic_number",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="atomic_number",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_author_helm = LightevalTaskConfig(
    name="wikifact:author",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="author",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_award_received_helm = LightevalTaskConfig(
    name="wikifact:award_received",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="award_received",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_basic_form_of_government_helm = LightevalTaskConfig(
    name="wikifact:basic_form_of_government",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="basic_form_of_government",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_capital_helm = LightevalTaskConfig(
    name="wikifact:capital",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="capital",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_capital_of_helm = LightevalTaskConfig(
    name="wikifact:capital_of",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="capital_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_central_bank_helm = LightevalTaskConfig(
    name="wikifact:central_bank",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="central_bank",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_composer_helm = LightevalTaskConfig(
    name="wikifact:composer",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="composer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_continent_helm = LightevalTaskConfig(
    name="wikifact:continent",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="continent",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_country_helm = LightevalTaskConfig(
    name="wikifact:country",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="country",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_country_of_citizenship_helm = LightevalTaskConfig(
    name="wikifact:country_of_citizenship",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="country_of_citizenship",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_country_of_origin_helm = LightevalTaskConfig(
    name="wikifact:country_of_origin",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="country_of_origin",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_creator_helm = LightevalTaskConfig(
    name="wikifact:creator",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="creator",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_currency_helm = LightevalTaskConfig(
    name="wikifact:currency",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="currency",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_defendant_helm = LightevalTaskConfig(
    name="wikifact:defendant",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="defendant",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_developer_helm = LightevalTaskConfig(
    name="wikifact:developer",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="developer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_diplomatic_relation_helm = LightevalTaskConfig(
    name="wikifact:diplomatic_relation",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="diplomatic_relation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_director_helm = LightevalTaskConfig(
    name="wikifact:director",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="director",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_discoverer_or_inventor_helm = LightevalTaskConfig(
    name="wikifact:discoverer_or_inventor",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="discoverer_or_inventor",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_drug_or_therapy_used_for_treatment_helm = LightevalTaskConfig(
    name="wikifact:drug_or_therapy_used_for_treatment",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="drug_or_therapy_used_for_treatment",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_educated_at_helm = LightevalTaskConfig(
    name="wikifact:educated_at",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="educated_at",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_electron_configuration_helm = LightevalTaskConfig(
    name="wikifact:electron_configuration",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="electron_configuration",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_employer_helm = LightevalTaskConfig(
    name="wikifact:employer",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="employer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_field_of_work_helm = LightevalTaskConfig(
    name="wikifact:field_of_work",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="field_of_work",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_file_extension_helm = LightevalTaskConfig(
    name="wikifact:file_extension",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="file_extension",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_genetic_association_helm = LightevalTaskConfig(
    name="wikifact:genetic_association",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="genetic_association",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_genre_helm = LightevalTaskConfig(
    name="wikifact:genre",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="genre",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_has_part_helm = LightevalTaskConfig(
    name="wikifact:has_part",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="has_part",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_head_of_government_helm = LightevalTaskConfig(
    name="wikifact:head_of_government",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="head_of_government",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_head_of_state_helm = LightevalTaskConfig(
    name="wikifact:head_of_state",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="head_of_state",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_headquarters_location_helm = LightevalTaskConfig(
    name="wikifact:headquarters_location",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="headquarters_location",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_industry_helm = LightevalTaskConfig(
    name="wikifact:industry",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="industry",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_influenced_by_helm = LightevalTaskConfig(
    name="wikifact:influenced_by",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="influenced_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_instance_of_helm = LightevalTaskConfig(
    name="wikifact:instance_of",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="instance_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_instrument_helm = LightevalTaskConfig(
    name="wikifact:instrument",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="instrument",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_language_of_work_or_name_helm = LightevalTaskConfig(
    name="wikifact:language_of_work_or_name",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="language_of_work_or_name",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_languages_spoken_written_or_signed_helm = LightevalTaskConfig(
    name="wikifact:languages_spoken_written_or_signed",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="languages_spoken_written_or_signed",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_laws_applied_helm = LightevalTaskConfig(
    name="wikifact:laws_applied",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="laws_applied",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_located_in_the_administrative_territorial_entity_helm = LightevalTaskConfig(
    name="wikifact:located_in_the_administrative_territorial_entity",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="located_in_the_administrative_territorial_entity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_location_helm = LightevalTaskConfig(
    name="wikifact:location",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="location",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_location_of_discovery_helm = LightevalTaskConfig(
    name="wikifact:location_of_discovery",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="location_of_discovery",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_location_of_formation_helm = LightevalTaskConfig(
    name="wikifact:location_of_formation",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="location_of_formation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_majority_opinion_by_helm = LightevalTaskConfig(
    name="wikifact:majority_opinion_by",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="majority_opinion_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_manufacturer_helm = LightevalTaskConfig(
    name="wikifact:manufacturer",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="manufacturer",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_measured_physical_quantity_helm = LightevalTaskConfig(
    name="wikifact:measured_physical_quantity",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="measured_physical_quantity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_medical_condition_treated_helm = LightevalTaskConfig(
    name="wikifact:medical_condition_treated",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="medical_condition_treated",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_member_of_helm = LightevalTaskConfig(
    name="wikifact:member_of",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="member_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_member_of_political_party_helm = LightevalTaskConfig(
    name="wikifact:member_of_political_party",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="member_of_political_party",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_member_of_sports_team_helm = LightevalTaskConfig(
    name="wikifact:member_of_sports_team",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="member_of_sports_team",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_movement_helm = LightevalTaskConfig(
    name="wikifact:movement",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="movement",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_named_after_helm = LightevalTaskConfig(
    name="wikifact:named_after",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="named_after",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_native_language_helm = LightevalTaskConfig(
    name="wikifact:native_language",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="native_language",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_number_of_processor_cores_helm = LightevalTaskConfig(
    name="wikifact:number_of_processor_cores",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="number_of_processor_cores",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_occupation_helm = LightevalTaskConfig(
    name="wikifact:occupation",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="occupation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_office_held_by_head_of_government_helm = LightevalTaskConfig(
    name="wikifact:office_held_by_head_of_government",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="office_held_by_head_of_government",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_office_held_by_head_of_state_helm = LightevalTaskConfig(
    name="wikifact:office_held_by_head_of_state",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="office_held_by_head_of_state",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_official_language_helm = LightevalTaskConfig(
    name="wikifact:official_language",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="official_language",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_operating_system_helm = LightevalTaskConfig(
    name="wikifact:operating_system",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="operating_system",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_original_language_of_film_or_TV_show_helm = LightevalTaskConfig(
    name="wikifact:original_language_of_film_or_TV_show",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="original_language_of_film_or_TV_show",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_original_network_helm = LightevalTaskConfig(
    name="wikifact:original_network",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="original_network",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_overrules_helm = LightevalTaskConfig(
    name="wikifact:overrules",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="overrules",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_owned_by_helm = LightevalTaskConfig(
    name="wikifact:owned_by",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="owned_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_part_of_helm = LightevalTaskConfig(
    name="wikifact:part_of",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="part_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_participating_team_helm = LightevalTaskConfig(
    name="wikifact:participating_team",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="participating_team",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_place_of_birth_helm = LightevalTaskConfig(
    name="wikifact:place_of_birth",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="place_of_birth",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_place_of_death_helm = LightevalTaskConfig(
    name="wikifact:place_of_death",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="place_of_death",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_plaintiff_helm = LightevalTaskConfig(
    name="wikifact:plaintiff",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="plaintiff",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_position_held_helm = LightevalTaskConfig(
    name="wikifact:position_held",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="position_held",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_position_played_on_team_helm = LightevalTaskConfig(
    name="wikifact:position_played_on_team",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="position_played_on_team",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_programming_language_helm = LightevalTaskConfig(
    name="wikifact:programming_language",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="programming_language",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_recommended_unit_of_measurement_helm = LightevalTaskConfig(
    name="wikifact:recommended_unit_of_measurement",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="recommended_unit_of_measurement",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_record_label_helm = LightevalTaskConfig(
    name="wikifact:record_label",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="record_label",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_religion_helm = LightevalTaskConfig(
    name="wikifact:religion",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="religion",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_repealed_by_helm = LightevalTaskConfig(
    name="wikifact:repealed_by",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="repealed_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_shares_border_with_helm = LightevalTaskConfig(
    name="wikifact:shares_border_with",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="shares_border_with",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_solved_by_helm = LightevalTaskConfig(
    name="wikifact:solved_by",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="solved_by",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_statement_describes_helm = LightevalTaskConfig(
    name="wikifact:statement_describes",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="statement_describes",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_stock_exchange_helm = LightevalTaskConfig(
    name="wikifact:stock_exchange",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="stock_exchange",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_subclass_of_helm = LightevalTaskConfig(
    name="wikifact:subclass_of",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="subclass_of",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_subsidiary_helm = LightevalTaskConfig(
    name="wikifact:subsidiary",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="subsidiary",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_symptoms_and_signs_helm = LightevalTaskConfig(
    name="wikifact:symptoms_and_signs",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="symptoms_and_signs",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_therapeutic_area_helm = LightevalTaskConfig(
    name="wikifact:therapeutic_area",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="therapeutic_area",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_time_of_discovery_or_invention_helm = LightevalTaskConfig(
    name="wikifact:time_of_discovery_or_invention",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="time_of_discovery_or_invention",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_twinned_administrative_body_helm = LightevalTaskConfig(
    name="wikifact:twinned_administrative_body",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="twinned_administrative_body",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

wikifact_work_location_helm = LightevalTaskConfig(
    name="wikifact:work_location",
    suite=["helm"],
    prompt_function=prompt.wikifact,
    hf_repo="lighteval/wikifact",
    hf_subset="work_location",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)
