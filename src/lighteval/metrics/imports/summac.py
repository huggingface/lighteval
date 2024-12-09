# noqa: CPY001
# Source: https://github.com/tingofurro/summac
# Apache 2 license
###############################################

import json
import logging
import os
import time

import nltk
import numpy as np
import torch
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)

# GPU-related business


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi")
    memory_available = [int(x.split()[2]) + 5 * i for i, x in enumerate(open("tmp_smi", "r").readlines())]
    os.remove("tmp_smi")
    return np.argmax(memory_available)


def any_gpu_with_space(gb_needed):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi")
    memory_available = [float(x.split()[2]) / 1024.0 for i, x in enumerate(open("tmp_smi", "r").readlines())]
    os.remove("tmp_smi")
    return any(mem >= gb_needed for mem in memory_available)


def wait_free_gpu(gb_needed):
    while not any_gpu_with_space(gb_needed):
        time.sleep(30)


def select_freer_gpu():
    freer_gpu = str(get_freer_gpu())
    logger.info("Will use GPU: %s" % (freer_gpu))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" + freer_gpu
    return freer_gpu


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0:  # Leftovers
        yield batch


model_map = {
    "snli-base": {"model_card": "boychaboy/SNLI_roberta-base", "entailment_idx": 0, "contradiction_idx": 2},
    "snli-large": {"model_card": "boychaboy/SNLI_roberta-large", "entailment_idx": 0, "contradiction_idx": 2},
    "mnli-base": {"model_card": "microsoft/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli": {"model_card": "roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {
        "model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "entailment_idx": 0,
        "contradiction_idx": 2,
    },
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc": {"model_card": "tals/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-only": {"model_card": "tals/albert-xlarge-vitaminc", "entailment_idx": 0, "contradiction_idx": 1},
    # "decomp": 0,
}


def card_to_name(card):
    card2name = {v["model_card"]: k for k, v in model_map.items()}
    if card in card2name:
        return card2name[card]
    return card


def name_to_card(name):
    if name in model_map:
        return model_map[name]["model_card"]
    return name


def get_neutral_idx(ent_idx, con_idx):
    return list({0, 1, 2} - {ent_idx, con_idx})[0]


class SummaCImager:
    def __init__(
        self, model_name="mnli", granularity="paragraph", use_cache=True, max_doc_sents=100, device="cuda", **kwargs
    ):
        self.grans = granularity.split("-")

        assert (
            all(gran in ["paragraph", "sentence", "document", "2sents", "mixed"] for gran in self.grans)
            and len(self.grans) <= 2
        ), "Unrecognized `granularity` %s" % (granularity)
        assert model_name in model_map.keys(), "Unrecognized model name: `%s`" % (model_name)

        self.model_name = model_name
        if model_name != "decomp":
            self.model_card = name_to_card(model_name)
            self.entailment_idx = model_map[model_name]["entailment_idx"]
            self.contradiction_idx = model_map[model_name]["contradiction_idx"]
            self.neutral_idx = get_neutral_idx(self.entailment_idx, self.contradiction_idx)

        self.granularity = granularity
        self.use_cache = use_cache
        self.cache_folder = "/export/share/plaban/summac_cache/"

        self.max_doc_sents = max_doc_sents
        self.max_input_length = 500
        self.device = device
        self.cache = {}
        self.model = None  # Lazy loader

    def load_nli(self):
        if self.model_name == "decomp":
            from allennlp.predictors.predictor import Predictor

            self.model = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz",
                cuda_device=0,
            )

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_card).eval()
            self.model.to(self.device).half()

    def split_sentences(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def split_2sents(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        two_sents = [" ".join(sentences[i : (i + 2)]) for i in range(len(sentences))]
        return two_sents

    def split_paragraphs(self, text):
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    def split_text(self, text, granularity="sentence"):
        if granularity == "document":
            return [text]
        elif granularity == "paragraph":
            return self.split_paragraphs(text)
        elif granularity == "sentence":
            return self.split_sentences(text)
        elif granularity == "2sents":
            return self.split_2sents(text)
        elif granularity == "mixed":
            return self.split_sentences(text) + self.split_paragraphs(text)

    def build_image(self, original, generated):
        cache_key = (original, generated)
        if self.use_cache and cache_key in self.cache:
            cached_image = self.cache[cache_key]
            cached_image = cached_image[:, : self.max_doc_sents, :]
            return cached_image

        if len(self.grans) == 1:
            gran_doc, gran_sum = self.grans[0], self.grans[0]
        else:
            gran_doc, gran_sum = self.grans[0], self.grans[1]

        original_chunks = self.split_text(original, granularity=gran_doc)[: self.max_doc_sents]
        generated_chunks = self.split_text(generated, granularity=gran_sum)

        N_ori = len(original_chunks)
        N_gen = len(generated_chunks)

        if N_ori == 0 or N_gen == 0:
            return np.zeros((3, 1, 1))
        # assert (N_ori > 0 and N_gen > 0), "One of the inputs has no chunks"

        image = np.zeros((3, N_ori, N_gen))

        if self.model is None:
            self.load_nli()

        dataset = [
            {"premise": original_chunks[i], "hypothesis": generated_chunks[j], "doc_i": i, "gen_i": j}
            for i in range(N_ori)
            for j in range(N_gen)
        ]
        for batch in batcher(dataset, batch_size=20):
            if self.model_name == "decomp":
                batch_evids, batch_conts, batch_neuts = [], [], []
                batch_json = [{"premise": d["premise"], "hypothesis": d["hypothesis"]} for d in batch]
                model_outs = self.model.predict_batch_json(batch_json)
                for out in model_outs:
                    probs = out["label_probs"]
                    batch_evids.append(probs[0])
                    batch_conts.append(probs[1])
                    batch_neuts.append(probs[2])

            else:
                batch_prems = [b["premise"] for b in batch]
                batch_hypos = [b["hypothesis"] for b in batch]
                batch_tokens = self.tokenizer.batch_encode_plus(
                    list(zip(batch_prems, batch_hypos)),
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                    truncation_strategy="only_first",
                )
                batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
                with torch.no_grad():
                    model_outputs = self.model(**batch_tokens)

                batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
                batch_evids = batch_probs[:, self.entailment_idx].tolist()
                batch_conts = batch_probs[:, self.contradiction_idx].tolist()
                batch_neuts = batch_probs[:, self.neutral_idx].tolist()

            for b, evid, cont, neut in zip(batch, batch_evids, batch_conts, batch_neuts):
                image[0, b["doc_i"], b["gen_i"]] = evid
                image[1, b["doc_i"], b["gen_i"]] = cont
                image[2, b["doc_i"], b["gen_i"]] = neut

        if self.use_cache:
            self.cache[cache_key] = image
        return image

    def get_cache_file(self):
        return os.path.join(self.cache_folder, "cache_%s_%s.json" % (self.model_name, self.granularity))

    def save_cache(self):
        cache_cp = {"[///]".join(k): v.tolist() for k, v in self.cache.items()}
        with open(self.get_cache_file(), "w") as f:
            json.dump(cache_cp, f)

    def load_cache(self):
        cache_file = self.get_cache_file()
        if os.path.isfile(cache_file):
            with open(cache_file, "r") as f:
                cache_cp = json.load(f)
                self.cache = {tuple(k.split("[///]")): np.array(v) for k, v in cache_cp.items()}


class SummaCZS:
    def __init__(
        self,
        model_name="mnli",
        granularity="paragraph",
        op1="max",
        op2="mean",
        use_ent=True,
        use_con=True,
        imager_load_cache=True,
        device="cuda",
        **kwargs,
    ):
        assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
        assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"

        self.imager = SummaCImager(model_name=model_name, granularity=granularity, device=device, **kwargs)
        if imager_load_cache:
            self.imager.load_cache()
        self.op2 = op2
        self.op1 = op1
        self.use_ent = use_ent
        self.use_con = use_con

    def save_imager_cache(self):
        self.imager.save_cache()

    def score_one(self, original, generated):
        image = self.imager.build_image(original, generated)

        ent_scores = np.max(image[0], axis=0)
        co_scores = np.max(image[1], axis=0)
        if self.op1 == "mean":
            ent_scores = np.mean(image[0], axis=0)
            co_scores = np.mean(image[1], axis=0)
        elif self.op1 == "min":
            ent_scores = np.min(image[0], axis=0)
            co_scores = np.min(image[1], axis=0)

        if self.use_ent and self.use_con:
            scores = ent_scores - co_scores
        elif self.use_ent:
            scores = ent_scores
        elif self.use_con:
            scores = 1.0 - co_scores

        final_score = np.mean(scores)
        if self.op2 == "min":
            final_score = np.min(scores)
        elif self.op2 == "max":
            final_score = np.max(scores)

        return {"score": final_score, "image": image}

    def score(self, sources, generateds, **kwargs):
        output = {"scores": [], "images": []}
        for source, gen in zip(sources, generateds):
            score = self.score_one(source, gen)
            output["scores"].append(score["score"])
            output["images"].append(score["image"])
        return output
