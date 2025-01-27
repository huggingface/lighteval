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

# ruff: noqa: F405, F403, F401
"""
This module contains task configurations and prompt functions for evaluating
LLM models on Swiss legal datasets. Each task is defined using the
`LightevalTaskConfig` class with its respective prompt function. The tasks
cover a variety of benchmarks, including: translation of laws, court decisions
and press releases.

Author: Joel Niklaus
"""
import importlib.metadata as importlib_metadata
import logging
import os
import statistics
from dataclasses import dataclass

import nltk
import requests
import torch
from comet import download_model, load_from_checkpoint
from gemba import get_gemba_scores
from nltk import word_tokenize
from nltk.translate import meteor_score
from packaging import version
from sacrebleu import sentence_bleu, sentence_chrf, sentence_ter
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lighteval.metrics.imports.bert_scorer import BERTScorer
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import BertScore, JudgeLLM
from lighteval.metrics.normalizations import remove_braces, remove_braces_and_strip
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.extended.mix_eval.main import process_judge_response_freeform_gpt
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Try to optimize CUDA operations
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
    # Enable TF32 for faster matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable tensor cores if available
    if torch.cuda.get_device_capability()[0] >= 7:
        # This will speed up GPU inference, e.g., for COMET and BLEURT
        torch.set_float32_matmul_precision("medium")

# CUSTOM METRICS

SYSTEM = {
    "basic": "Act as a Judge specializing in the evaluation of translations of Swiss legal documents. Your task is to assess the accuracy, clarity, and fidelity of the model's translation to the golden translation, while considering the nuances of legal language.",
    "detailed": "You are a senior legal translator and quality assurance specialist with over 20 years of experience in Swiss law, certified by the Swiss Sworn Translators Association (Association suisse des traducteurs-jurés, ASTJ). You possess native-level proficiency in all Swiss national languages (German, French, Italian, and Romansh) as well as English, enabling precise evaluation of legal nuances across all linguistic combinations. Your task is to evaluate machine-translated legal texts for accuracy, clarity and fidelity to Swiss legal standards analyzing the subtle complexities of legal language. You excel at identifying even minor discrepancies and calibrating evaluation scores appropriately to reflect the severity of each error.",
}

USER = {
    "basic": """
You will be provided with a source text, its golden translation, and the model's translation. Your task is to judge how correct the model's translation is based on the golden translation, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). You should first briefly give your reasoning process regarding how the model's translation conforms to or contradicts the golden translation, and then give the correctness score. The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples.
""",
    "detailed": """
INPUT FORMAT:
Source Text: [Original text in source language]
Golden Translation: [Reference professional translation]
Model Translation: [Machine-generated translation to be evaluated]


EVALUATION DIMENSIONS:
Accuracy: Semantic equivalence, correct legal terminology, and preservation of legal meaning.
Clarity: Logical flow, appropriate legal register, and unambiguous expression.
Fidelity: Adherence to Swiss legal conventions, jurisdiction-specific terminology, and formal register.


SCORING RUBRIC:
1.0: Perfect translation
0.7-0.9: Minor issues only
0.4-0.6: Significant but non-critical errors
0.1-0.3: Major errors affecting legal meaning
0.0: Completely incorrect


EVALUATION GUIDELINES:
Stylistic differences should not impact accuracy significantly unless they alter the legal meaning.
Untranslated Latin terms (e.g., prima facie) are not considered errors, but they should still be assessed for appropriate use within the context of the answer.
Terminology should be used consistently throughout the text.
Consider both explicit and implicit legal meanings.
Consider jurisdiction-specific legal terminology.
Flag any ambiguities, omissions or additions that affect legal meaning.


REQUIRED OUTPUT FORMAT:
Your response should be in plain text with the following sections:
Reasoning: Analyze how the model's translation aligns with or differs from the golden translation, focusing on significant legal and linguistic aspects.
Examples: Identify specific terms, phrases, or sections in the model's answer that were correct or incorrect, with explanations.
Score: End with exactly this format: \"The correctness score: [[score]]\"
The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples.""",
}

FEW_SHOT = {
    "diverse": """Example 1:
Source Text:
```A contract is void if its terms are impossible, unlawful or immoral. However, where the defect pertains only to certain terms of a contract, those terms alone are void unless there is cause to assume that the contract would not have been concluded without them.```

Golden Translation:
```Il contratto che ha per oggetto una cosa impossibile o contraria alle leggi od ai buoni costumi è nullo. Se il contratto è viziato solo in alcune parti, queste soltanto sono nulle, ove non si debba ammettere che senza la parte nulla esso non sarebbe stato conchiuso.```

Model’s Translation:
```Il contratto è nullo se le sue clausole sono impossibili, illecite o immorali. Tuttavia, quando il vizio riguarda solo determinate clausole del contratto, solo queste sono nulle, salvo che vi sia motivo di ritenere che il contratto non sarebbe stato concluso senza di esse.```

Your Judgment: The model’s translation aligns well with the golden translation in terms of accuracy, clarity, and fidelity to the source text. However, there are minor stylistic differences. For example, the golden translation uses “conchiuso” an older and more formal term, while the model opts for “concluso” which is modern. Similarly, the golden translation uses the idiomatic phrase “contraria alle leggi od ai buoni costumi” whereas the model employs the more literal “illecite o immorali”. The correctness score: [[0.9]]


Example 2:
Source Text:
```Art. 13 Abs. 2, Art. 36 Abs. 1 BV; Art. 141 Abs. 2 StPO; Verwertbarkeit von polizeilichen Aufzeichnungen der automatischen Fahrzeugfahndung und Verkehrsüberwachung (AFV).
Die Erhebung und die Aufbewahrung von Aufzeichnungen der AFV stellen einen Eingriff in die Grundrechte der Betroffenen dar, insbesondere in das Recht auf Privatsphäre, das den Anspruch auf informationelle Selbstbestimmung miteinschliesst (E. 3.1). Für die AFV besteht im Kanton Thurgau keine hinreichend bestimmte gesetzliche Grundlage. Der mit der Überwachung verbundene Eingriff in die Privatsphäre verstösst daher gegen Art. 13 Abs. 2 i.V.m. Art. 36 Abs. 1 BV (E. 3.2 und 3.3).
Stellt die Polizei im Rahmen ihrer präventiven Kontrolltätigkeit strafbare Handlungen fest, ermittelt sie nach Art. 306 ff. StPO. Die Frage, ob die mangels gesetzlicher Grundlage rechtswidrig erhobenen Beweismittel im Strafprozess verwertbar sind, ist nach Art. 141 Abs. 2 StPO zu prüfen (Präzisierung der Rechtsprechung; E. 4.1 und 4.2). Verwertbarkeit im vorliegenden Fall verneint (E. 4.3).```

Golden Translation:
```Art. 13 al. 2, art. 36 al. 1 Cst.; art. 141 al. 2 CPP; exploitabilité d'enregistrements réalisés par la police au moyen du Système de recherche automatisée de véhicules et surveillance du trafic (RVS).
La réalisation et le stockage d'enregistrements par la RVS constituent une atteinte aux droits fondamentaux des personnes concernées, singulièrement à la protection de leur sphère privée, qui englobe le droit à l'autodétermination en matière de données (consid. 3.1). Le recours à la RVS ne repose sur aucune base légale suffisamment précise dans le canton de Thurgovie. L'atteinte à la sphère privée inhérente à cette surveillance viole partant l'art. 13 al. 2 en corrélation avec l'art. 36 al. 1 Cst. (consid. 3.2 et 3.3).
La constatation d'infractions par la police dans le cadre de son activité de contrôle préventive, relève de l'investigation au sens des art. 306 ss CPP. L'exploitabilité dans le procès pénal de preuves recueillies de manière illicite faute de base légale doit être examinée à l'aune de l'art. 141 al. 2 CPP (précision de la jurisprudence; consid. 4.1 et 4.2). Exploitabilité niée en l'espèce (consid. 4.3).```

Model’s Translation:
```Art. 13 al. 2, Art. 36 al. 1 Cst.; Art. 141 al. 2 CPP; Admissibilité des enregistrements de la poursuite automatisée des véhicules et de la surveillance de la circulation (AFV).
La collecte et la conservation des enregistrements de l'AFV constituent une ingérence dans les droits fondamentaux des personnes concernées, en particulier dans le droit à la vie privée, qui inclut le droit à l'autodétermination informationnelle (consid. 3.1). Dans le canton de Thurgovie, il n'existe pas de base légale suffisamment déterminée pour l'AFV. Par conséquent, l'ingérence dans la vie privée associée à la surveillance viole donc les art. 13 al. 2 combiné avec l'art. 36 al. 1 Cst (consid. 3.2 et 3.3).
Si la police relève des infractions pénales dans le cadre de ses activités de contrôle préventif, elle enquête conformément aux art. 306 et suivants CPP. La question de savoir si les preuves illégalement recueillies peuvent être utilisées dans la procédure pénale est examinée conformément à l'art. 141 al. 2 CPP (précision de la jurisprudence; consid. 4.1 et 4.2). Admissibilité dans le cas présent niée (consid. 4.3).```

Your Judgment: The model’s translation mostly aligns with the golden translation but diverges when it comes to accuracy and fidelity to Swiss legal terminology. For instance, the term “exploitabilité” which is closer to the Swiss provision is replaced in the model’s translation with “admissibilité”. Similarly, “ingérence” is used instead of “atteinte”, although “atteinte” is commonly used in Swiss law to discuss a violation of fundamental rights. Also, the term "recherche automatisée de véhicules et surveillance du trafic (RVS)" used by the golden translation is more established than "poursuite automatisée des véhicules et de la surveillance de la circulation (AFV)" in the model’s translation. The model’s translation is almost complete, but omits a critical point in one sentence: that the evidence was unlawfully obtained due to lack of a sufficiently clear legal basis. This omission impacts the completeness. The correctness score: [[0.7]]


Example 3:
Source Text:
```Yoko Ono est propriétaire de la montre de John Lennon – rejet du recours d'un collectionneur contre un arrêt rendu par la Cour de justice genevoise

Le Tribunal fédéral rejette le recours déposé par un collectionneur contre l'arrêt de la Cour de justice genevoise par lequel celle-ci confirmait que Yoko Ono est propriétaire de la montre qu'elle avait offerte à John Lennon en 1980, deux mois avant qu'il ne soit assassiné. Le collectionneur, qui a remis la montre à une maison de vente aux enchères genevoise en 2014 afin d'en faire estimer la valeur, a quant à lui revendiqué la propriété de ladite montre.

En 1980, Yoko Ono a acquis à New York une montre de marque Patek Philippe. Elle y a fait graver au dos l'inscription « (JUST LIKE) STARTING OVER LOVE YOKO 10·9·1980 N.Y.C » et l'a offerte à son époux, John Lennon, le 9 octobre 1980 pour son 40e anniversaire. Le 8 décembre 1980, John Lennon a été assassiné à New York. La montre a été répertoriée dans l'inventaire successoral et conservée dans une pièce de l'appartement de Yoko Ono à New York. Par la suite, la montre s'est retrouvée aux mains d'un homme qui avait été le chauffeur privé de Yoko Ono de 1995 à 2006. Un autre possesseur intermédiaire l'a remise à une maison de vente aux enchères allemande, où elle a été acquise par un collectionneur en 2014. Ce dernier l'a remise la même année à une maison de vente aux enchères genevoise afin d'en faire estimer la valeur, ce dont a été informée Yoko Ono. Cette dernière n'avait jusqu'alors pas eu conscience du fait que la montre n'était plus en sa possession. En 2018, le collectionneur a formé à Genève une action visant à constater sa qualité de propriétaire, action à laquelle Yoko Ono s'est opposée. En 2022, le tribunal de première instance genevois a constaté que Yoko Ono était la seule et unique propriétaire de la montre, ce que la Cour de justice du canton de Genève, statuant sur appel du collectionneur, a confirmé en 2023.

Le Tribunal fédéral rejette le recours déposé par le collectionneur contre cet arrêt. Il n'est tout d'abord pas contesté que la propriété de la montre a été acquise par succession par Yoko Ono après le décès de John Lennon. C'est en outre sans arbitraire que la Cour de justice genevoise a retenu que la montre avait été volée par l'ancien chauffeur et que, à l'inverse, aucun élément ne permettait de démontrer que Yoko Ono aurait eu l'intention de faire donation au chauffeur d'une chose si particulière que la montre, gravée d'une inscription, qu'elle avait offerte à John Lennon deux mois avant son décès. Dès lors qu'il s'agit d'une chose volée, le collectionneur, aujourd'hui recourant, ne pouvait pas acquérir la propriété de la montre par un mode originaire d'acquisition lorsqu'il l'a achetée en Allemagne en 2014 ; selon le droit allemand applicable en la matière, cela vaut indépendamment du fait que l'acquéreur était ou non de bonne foi quant à l'origine de la chose.```

Golden Translation:
```Yoko Ono ist Eigentümerin der Uhr von John Lennon – Beschwerde von Sammler gegen Genfer Urteil abgewiesen

Das Bundesgericht weist die Beschwerde eines Sammlers gegen das Urteil des Genfer Kantonsgerichts ab, mit dem Yoko Ono als Eigentümerin der Uhr bestätigt wurde, die sie John Lennon 1980 zwei Monate vor seiner Ermordung geschenkt hat. Der Sammler hatte die Uhr 2014 zur Schätzung bei einem Auktionshaus in Genf eingereicht und seinerseits Eigentümerschaft an der Uhr geltend gemacht.

Yoko Ono hatte 1980 in New York eine Uhr der Marke Patek Philippe gekauft. Sie liess auf der Rückseite die Gravur "(JUST LIKE) STARTING OVER LOVE YOKO 10·9·1980 N.Y.C" anbringen und schenkte sie ihrem Ehemann John Lennon am 9. Oktober 1980 zum 40. Geburtstag. Am 8. Dezember 1980 wurde John Lennon in New York ermordet. Die Uhr wurde ins Erbschaftsinventar aufgenommen und in einem Zimmer der Wohnung von Yoko Ono in New York aufbewahrt. Sie gelangte von dort in die Hände eines Mannes, der von 1995 bis 2006 Privatchauffeur von Yoko Ono gewesen war. Ein weiterer Zwischenbesitzer brachte die Uhr in ein deutsches Auktionshaus, wo sie 2014 von einem Sammler erworben wurde. Dieser reichte die Uhr im gleichen Jahr bei einem Auktionshaus in Genf zur Schätzung ihres Wertes ein. Davon erfuhr Yoko Ono, die bis dahin keine Kenntnis davon gehabt hatte, dass sich die Uhr nicht mehr in ihrem Besitz befand. Der Sammler erhob 2018 in Genf eine Klage auf Feststellung seiner Eigentümerschaft, der sich Yoko Ono widersetzte. Das erstinstanzliche Genfer Gericht stellte 2022 fest, dass Yoko Ono die alleinige Eigentümerin der Uhr sei, was auf Berufung des Sammlers 2023 vom Kantonsgericht des Kantons Genf bestätigt wurde.

Das Bundesgericht weist die dagegen erhobene Beschwerde des Sammlers ab. Unbestritten ist zunächst, dass das Eigentum an der Uhr nach dem Tod von John Lennon durch Erbschaft an Yoko Ono übergegangen ist. Sodann durfte das Genfer Kantonsgericht willkürfrei davon ausgehen, dass die Uhr vom ehemaligen Chauffeur gestohlen wurde und umgekehrt nichts dafür spricht, dass Yoko Ono dem Chauffeur dieses einzigartige Stück, welches sie John Lennon zwei Monate vor seinem Tod mit einer Inschrift versehen überreicht hatte, hätte schenken wollen. Da es sich bei der Uhr um eine gestohlene Sache handelt, konnte der Sammler und heutige Beschwerdeführer bei ihrem Erwerb 2014 in Deutschland nicht unmittelbar zum Eigentümer werden; dies gilt gemäss dem diesbezüglich anwendbaren deutschen Recht unabhängig davon, ob der Erwerber in Bezug auf die Herkunft der Sache in gutem Glauben war.```

Model’s Translation:
```Hier, der Bundesgerichtshof hat den Antrag des Käufer gegen den Urteil des Zivilgerichts in Zürich abgelehnt, das das Zivilgericht in Zürich 2023 bestätigt hatte, dass Yoko Ono die Uhr als Geschenk 1980 an John Lennon gegeben hat, bevor dieser am 8. Dezember 1980 ermordet wurde. Der Käufer, der die Uhr 2014 in eine Geneßische Auktion verkauft hat, behauptet, die Uhr sei ihm verliehen worden, und nicht von Yoko Ono.

Im Jahr 1980 hat Yoko Ono in New York eine Uhr von der Marke Patek Philippe gekauft. Sie hat auf der Rückseite der Uhr die Inschrift "JUST LIKE STARTING OVER LOVE YOKO 10·9·1980 N.Y.C." gemalt und sie an ihren Ehemann, John Lennon, am 9. Oktober 1980 als Geburtstag gegeben. Am 8. Dezember 1980 wurde Lennon ermordet in New York. Die Uhr wurde in den Erbeverzeichnis des Lennon-Mädchens aufbewahrt und in einem Zimmer in Yoko Onos New Yorker Wohnung.

Danach wurde die Uhr von einem Mann, der als Wagenkellner von Yoko Ono zwischen 1995 und 2006 gearbeitet hat, in die Hände gebracht. Ein weiterer mittlerer Eigentümer hat sie 2014 in eine deutsche Auktion verkauft, wo sie einem Käufer in Zürich übergeben wurde. Dieser Käufer hat die Uhr 2014 in eine Geneßische Auktion verkauft, um sie zu valieren, und Yoko Ono war von diesem Moment an nicht mehr bewusst, dass die Uhr nicht mehr in ihrer Hand war.

Im Jahr 2018 hat der Käufer in Zürich eine Anzeige gegen Yoko Ono eröffnet, um die Qualität des Eigentums der Uhr zu bestätigen, was Yoko Ono ansprach. Im Jahr 2022 hat der Zivilgericht in Zürich bestätigt, dass Yoko Ono die einzige und letzte Eigentümerin der Uhr ist, was der Zivilgerichtshof im Jahr 2023 bestätigt hat.

Der Bundesgerichtshof hat den Antrag des Käufer gegen dieses Urteil abgelehnt. Er ist zunächst nicht bezweifelt, dass Yoko Ono die Uhr nach dem Tod von John Lennon durch Erbvergabe gekauft hat. Es ist auch nicht bezweifelt, dass die Zivilgerichtsverfassung in Zürich die Uhr als gestohlen angenommen hat, da niemand einen Beweis dafür gab, dass Yoko Ono die Uhr als Geschenk an den Wagenkellner hatte. Deshalb kann der Käufer nicht durch eine rechtliche Handlung die Uhr von Yoko Ono erwerben, wenn er die Uhr in Deutschland gekauft hat.```

Your Judgment: The model’s translation diverges significantly from the golden translation in accuracy, clarity, and fidelity. Critical legal terminology is mistranslated, omitted, and distorted. For instance, the courts are misidentified (“Zivilgerichtsverfassung”, “Zivilgericht”, “Bundesgerichtshof”). The model’s translation has several grammatical errors, such as “Geneßische Auktion”, “Erbvergabe”, “Wagenkellner” and “zu valieren”. The model also omits the explanation that, under German law, stolen property cannot be acquired in good faith. The correctness score: [[0.2]]
""",
    "single": """Example 1:
Source Text:
```Le contrat est nul s’il a pour objet une chose impossible, illicite ou contraire aux moeurs. Si le contrat n’est vicié que dans certaines de ses clauses, ces clauses sont seules frappées de nullité, à moins qu’il n’y ait lieu d’admettre que le contrat n’aurait pas été conclu sans elles.```

Golden Translation:
```Ein Vertrag, der einen unmöglichen oder widerrechtlichen Inhalt hat oder gegen die guten Sitten verstösst, ist nichtig. Betrifft aber der Mangel bloss einzelne Teile des Vertrages, so sind nur diese nichtig, sobald nicht anzunehmen ist, dass er ohne den nichtigen Teil überhaupt nicht geschlossen worden wäre.```

Model’s Translation:
```Der Vertrag ist nichtig, wenn er einen unmöglichen, widerrechtlichen oder sittenwidrigen Inhalt hat. Betrifft der Mangel bloß einzelne Teile des Vertrages, so sind nur diese nichtig, sobald nicht anzunehmen ist, dass er ohne den nichtigen Teil überhaupt nicht geschlossen worden wäre.```

Your Judgment: The model’s translation aligns well with the golden translation in terms of accuracy and clarity. However, minor stylistic differences exist. For example, the golden translation uses “gegen die guten Sitten verstösst” which is more idiomatic, while the model opts for the more literal “sittenwidrigen Inhalt hat” Similarly, “Ein Vertrag” in the golden translation better reflects generalized legal phrasing than the model’s literal “Der Vertrag”. The correctness score: [[0.9]]


Example 2:
Source Text:
```Art. 13 al. 2, art. 36 al. 1 Cst.; art. 141 al. 2 CPP; exploitabilité d'enregistrements réalisés par la police au moyen du Système de recherche automatisée de véhicules et surveillance du trafic (RVS).
La réalisation et le stockage d'enregistrements par la RVS constituent une atteinte aux droits fondamentaux des personnes concernées, singulièrement à la protection de leur sphère privée, qui englobe le droit à l'autodétermination en matière de données (consid. 3.1). Le recours à la RVS ne repose sur aucune base légale suffisamment précise dans le canton de Thurgovie. L'atteinte à la sphère privée inhérente à cette surveillance viole partant l'art. 13 al. 2 en corrélation avec l'art. 36 al. 1 Cst. (consid. 3.2 et 3.3).
La constatation d'infractions par la police dans le cadre de son activité de contrôle préventive, relève de l'investigation au sens des art. 306 ss CPP. L'exploitabilité dans le procès pénal de preuves recueillies de manière illicite faute de base légale doit être examinée à l'aune de l'art. 141 al. 2 CPP (précision de la jurisprudence; consid. 4.1 et 4.2). Exploitabilité niée en l'espèce (consid. 4.3).```

Golden Translation:
```Art. 13 Abs. 2, Art. 36 Abs. 1 BV; Art. 141 Abs. 2 StPO; Verwertbarkeit von polizeilichen Aufzeichnungen der automatischen Fahrzeugfahndung und Verkehrsüberwachung (AFV).
Die Erhebung und die Aufbewahrung von Aufzeichnungen der AFV stellen einen Eingriff in die Grundrechte der Betroffenen dar, insbesondere in das Recht auf Privatsphäre, das den Anspruch auf informationelle Selbstbestimmung miteinschliesst (E. 3.1). Für die AFV besteht im Kanton Thurgau keine hinreichend bestimmte gesetzliche Grundlage. Der mit der Überwachung verbundene Eingriff in die Privatsphäre verstösst daher gegen Art. 13 Abs. 2 i.V.m. Art. 36 Abs. 1 BV (E. 3.2 und 3.3).
Stellt die Polizei im Rahmen ihrer präventiven Kontrolltätigkeit strafbare Handlungen fest, ermittelt sie nach Art. 306 ff. StPO. Die Frage, ob die mangels gesetzlicher Grundlage rechtswidrig erhobenen Beweismittel im Strafprozess verwertbar sind, ist nach Art. 141 Abs. 2 StPO zu prüfen (Präzisierung der Rechtsprechung; E. 4.1 und 4.2). Verwertbarkeit im vorliegenden Fall verneint (E. 4.3).```

Model’s Translation:
```Art. 13 Abs. 2, Art. 36 Abs. 1 BV; Art. 141 Abs. 2 StPO; Verwertbarkeit von Aufzeichnungen, die durch die Polizei mit dem System zur automatisierten Fahrzeugerfassung und Verkehrsüberwachung (RVS) durchgeführt wurden.
Die Durchführung und Speicherung von Aufzeichnungen durch das RVS stellt einen Eingriff in die Grundrechte der betroffenen Personen dar, insbesondere in den Schutz ihrer Privatsphäre, die das Recht auf Selbstbestimmung in Bezug auf Daten umfasst (E. 3.1). Der Einsatz von RVS stützt sich im Kanton Thurgau auf keine ausreichend präzise gesetzliche Grundlage. Die in dieser Überwachung liegende Beeinträchtigung der Privatsphäre verstößt daher gegen Art. 13 Abs. 2 in Verbindung mit Art. 36 Abs. 1 BV (E. 3.2 und 3.3).
Die Aufdeckung von Straftaten durch die Polizei im Rahmen ihrer präventiven Kontrolltätigkeit fällt unter die Ermittlung im Sinne der Art. 306 ff. StPO. Die Verwertbarkeit im Strafverfahren von Beweisen, die auf illegale Weise erhoben wurden, muss anhand von Art. 141 Abs. 2 StPO geprüft werden (Klarstellung der Rechtsprechung; E. 4.1 und 4.2). Verwertbarkeit in diesem Fall verneint (E. 4.3).```

Your Judgment: The model’s translation is accurate overall but omits a critical point in the second-to-last sentence: the evidence was unlawfully obtained due to a lack of legal basis. Additionally, its fidelity to Swiss legal terminology is limited. For example, the model uses "Klarstellung der Rechtsprechung" instead of the more appropriate "Präzisierung der Rechtsprechung" and "nicht ausreichend präzise" rather than the common "hinreichend bestimmt" It also consistently uses the French abbreviation "RVS" instead of the German "automatische Fahrzeugfahndung und Verkehrsüberwachung (AFV)" Lastly, "Recht auf Selbstbestimmung in Bezug auf Daten" is overly literal compared to the idiomatic "Anspruch auf informationelle Selbstbestimmung". The correctness score: [[0.6]]


Example 3:
Source Text:
```Yoko Ono est propriétaire de la montre de John Lennon – rejet du recours d'un collectionneur contre un arrêt rendu par la Cour de justice genevoise

Le Tribunal fédéral rejette le recours déposé par un collectionneur contre l'arrêt de la Cour de justice genevoise par lequel celle-ci confirmait que Yoko Ono est propriétaire de la montre qu'elle avait offerte à John Lennon en 1980, deux mois avant qu'il ne soit assassiné. Le collectionneur, qui a remis la montre à une maison de vente aux enchères genevoise en 2014 afin d'en faire estimer la valeur, a quant à lui revendiqué la propriété de ladite montre.

En 1980, Yoko Ono a acquis à New York une montre de marque Patek Philippe. Elle y a fait graver au dos l'inscription « (JUST LIKE) STARTING OVER LOVE YOKO 10·9·1980 N.Y.C » et l'a offerte à son époux, John Lennon, le 9 octobre 1980 pour son 40e anniversaire. Le 8 décembre 1980, John Lennon a été assassiné à New York. La montre a été répertoriée dans l'inventaire successoral et conservée dans une pièce de l'appartement de Yoko Ono à New York. Par la suite, la montre s'est retrouvée aux mains d'un homme qui avait été le chauffeur privé de Yoko Ono de 1995 à 2006. Un autre possesseur intermédiaire l'a remise à une maison de vente aux enchères allemande, où elle a été acquise par un collectionneur en 2014. Ce dernier l'a remise la même année à une maison de vente aux enchères genevoise afin d'en faire estimer la valeur, ce dont a été informée Yoko Ono. Cette dernière n'avait jusqu'alors pas eu conscience du fait que la montre n'était plus en sa possession. En 2018, le collectionneur a formé à Genève une action visant à constater sa qualité de propriétaire, action à laquelle Yoko Ono s'est opposée. En 2022, le tribunal de première instance genevois a constaté que Yoko Ono était la seule et unique propriétaire de la montre, ce que la Cour de justice du canton de Genève, statuant sur appel du collectionneur, a confirmé en 2023.

Le Tribunal fédéral rejette le recours déposé par le collectionneur contre cet arrêt. Il n'est tout d'abord pas contesté que la propriété de la montre a été acquise par succession par Yoko Ono après le décès de John Lennon. C'est en outre sans arbitraire que la Cour de justice genevoise a retenu que la montre avait été volée par l'ancien chauffeur et que, à l'inverse, aucun élément ne permettait de démontrer que Yoko Ono aurait eu l'intention de faire donation au chauffeur d'une chose si particulière que la montre, gravée d'une inscription, qu'elle avait offerte à John Lennon deux mois avant son décès. Dès lors qu'il s'agit d'une chose volée, le collectionneur, aujourd'hui recourant, ne pouvait pas acquérir la propriété de la montre par un mode originaire d'acquisition lorsqu'il l'a achetée en Allemagne en 2014 ; selon le droit allemand applicable en la matière, cela vaut indépendamment du fait que l'acquéreur était ou non de bonne foi quant à l'origine de la chose.```

Golden Translation:
```Yoko Ono ist Eigentümerin der Uhr von John Lennon – Beschwerde von Sammler gegen Genfer Urteil abgewiesen

Das Bundesgericht weist die Beschwerde eines Sammlers gegen das Urteil des Genfer Kantonsgerichts ab, mit dem Yoko Ono als Eigentümerin der Uhr bestätigt wurde, die sie John Lennon 1980 zwei Monate vor seiner Ermordung geschenkt hat. Der Sammler hatte die Uhr 2014 zur Schätzung bei einem Auktionshaus in Genf eingereicht und seinerseits Eigentümerschaft an der Uhr geltend gemacht.

Yoko Ono hatte 1980 in New York eine Uhr der Marke Patek Philippe gekauft. Sie liess auf der Rückseite die Gravur "(JUST LIKE) STARTING OVER LOVE YOKO 10·9·1980 N.Y.C" anbringen und schenkte sie ihrem Ehemann John Lennon am 9. Oktober 1980 zum 40. Geburtstag. Am 8. Dezember 1980 wurde John Lennon in New York ermordet. Die Uhr wurde ins Erbschaftsinventar aufgenommen und in einem Zimmer der Wohnung von Yoko Ono in New York aufbewahrt. Sie gelangte von dort in die Hände eines Mannes, der von 1995 bis 2006 Privatchauffeur von Yoko Ono gewesen war. Ein weiterer Zwischenbesitzer brachte die Uhr in ein deutsches Auktionshaus, wo sie 2014 von einem Sammler erworben wurde. Dieser reichte die Uhr im gleichen Jahr bei einem Auktionshaus in Genf zur Schätzung ihres Wertes ein. Davon erfuhr Yoko Ono, die bis dahin keine Kenntnis davon gehabt hatte, dass sich die Uhr nicht mehr in ihrem Besitz befand. Der Sammler erhob 2018 in Genf eine Klage auf Feststellung seiner Eigentümerschaft, der sich Yoko Ono widersetzte. Das erstinstanzliche Genfer Gericht stellte 2022 fest, dass Yoko Ono die alleinige Eigentümerin der Uhr sei, was auf Berufung des Sammlers 2023 vom Kantonsgericht des Kantons Genf bestätigt wurde.

Das Bundesgericht weist die dagegen erhobene Beschwerde des Sammlers ab. Unbestritten ist zunächst, dass das Eigentum an der Uhr nach dem Tod von John Lennon durch Erbschaft an Yoko Ono übergegangen ist. Sodann durfte das Genfer Kantonsgericht willkürfrei davon ausgehen, dass die Uhr vom ehemaligen Chauffeur gestohlen wurde und umgekehrt nichts dafür spricht, dass Yoko Ono dem Chauffeur dieses einzigartige Stück, welches sie John Lennon zwei Monate vor seinem Tod mit einer Inschrift versehen überreicht hatte, hätte schenken wollen. Da es sich bei der Uhr um eine gestohlene Sache handelt, konnte der Sammler und heutige Beschwerdeführer bei ihrem Erwerb 2014 in Deutschland nicht unmittelbar zum Eigentümer werden; dies gilt gemäss dem diesbezüglich anwendbaren deutschen Recht unabhängig davon, ob der Erwerber in Bezug auf die Herkunft der Sache in gutem Glauben war.```

Model’s Translation:
```Hier, der Bundesgerichtshof hat den Antrag des Käufer gegen den Urteil des Zivilgerichts in Zürich abgelehnt, das das Zivilgericht in Zürich 2023 bestätigt hatte, dass Yoko Ono die Uhr als Geschenk 1980 an John Lennon gegeben hat, bevor dieser am 8. Dezember 1980 ermordet wurde. Der Käufer, der die Uhr 2014 in eine Geneßische Auktion verkauft hat, behauptet, die Uhr sei ihm verliehen worden, und nicht von Yoko Ono.

Im Jahr 1980 hat Yoko Ono in New York eine Uhr von der Marke Patek Philippe gekauft. Sie hat auf der Rückseite der Uhr die Inschrift "JUST LIKE STARTING OVER LOVE YOKO 10·9·1980 N.Y.C." gemalt und sie an ihren Ehemann, John Lennon, am 9. Oktober 1980 als Geburtstag gegeben. Am 8. Dezember 1980 wurde Lennon ermordet in New York. Die Uhr wurde in den Erbeverzeichnis des Lennon-Mädchens aufbewahrt und in einem Zimmer in Yoko Onos New Yorker Wohnung.

Danach wurde die Uhr von einem Mann, der als Wagenkellner von Yoko Ono zwischen 1995 und 2006 gearbeitet hat, in die Hände gebracht. Ein weiterer mittlerer Eigentümer hat sie 2014 in eine deutsche Auktion verkauft, wo sie einem Käufer in Zürich übergeben wurde. Dieser Käufer hat die Uhr 2014 in eine Geneßische Auktion verkauft, um sie zu valieren, und Yoko Ono war von diesem Moment an nicht mehr bewusst, dass die Uhr nicht mehr in ihrer Hand war.

Im Jahr 2018 hat der Käufer in Zürich eine Anzeige gegen Yoko Ono eröffnet, um die Qualität des Eigentums der Uhr zu bestätigen, was Yoko Ono ansprach. Im Jahr 2022 hat der Zivilgericht in Zürich bestätigt, dass Yoko Ono die einzige und letzte Eigentümerin der Uhr ist, was der Zivilgerichtshof im Jahr 2023 bestätigt hat.

Der Bundesgerichtshof hat den Antrag des Käufer gegen dieses Urteil abgelehnt. Er ist zunächst nicht bezweifelt, dass Yoko Ono die Uhr nach dem Tod von John Lennon durch Erbvergabe gekauft hat. Es ist auch nicht bezweifelt, dass die Zivilgerichtsverfassung in Zürich die Uhr als gestohlen angenommen hat, da niemand einen Beweis dafür gab, dass Yoko Ono die Uhr als Geschenk an den Wagenkellner hatte. Deshalb kann der Käufer nicht durch eine rechtliche Handlung die Uhr von Yoko Ono erwerben, wenn er die Uhr in Deutschland gekauft hat.```

Your Judgment: The model’s translation diverges significantly from the golden translation in accuracy, clarity, and fidelity. Critical legal terminology is mistranslated, omitted, and distorted. For instance, the courts are misidentified (“Zivilgerichtsverfassung”, “Zivilgericht”, “Bundesgerichtshof”). The model’s translation has several grammatical errors, such as “Geneßische Auktion”, “Erbvergabe”, “Wagenkellner” and “zu valieren”. The model also omits the explanation that, under German law, stolen property cannot be acquired in good faith. The correctness score: [[0.2]]
""",
}


class JudgeSwissLegalTranslation(JudgeLLM):
    def compute(
        self,
        sample_ids: list[str],
        responses: list,
        formatted_docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        logger.info(f"Judging {len(formatted_docs)} samples with {self.short_judge_name}...")
        questions = [formatted_doc.specific["source"] for formatted_doc in formatted_docs]
        options = [formatted_doc.choices for formatted_doc in formatted_docs]
        golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]
        predictions = [response[0].result for response in responses]

        scores, _, judgements = self.judge.evaluate_answer_batch(questions, predictions, options, golds)
        # Exclude the messages (user prompt) because they are too long
        return [
            {
                self.short_judge_name: score * 100,
                f"{self.short_judge_name}_judgment": judgment,
            }
            for score, judgment in zip(scores, judgements)
        ]


def get_swiss_legal_translation_judge(
    judge_model_name: str = "openai/gpt-4o-2024-11-20",
    short_judge_name: str = "slt_judge_gpt-4o",
    backend: str = "litellm",
    system_style: str = "basic",  # "basic" or "detailed"
    few_shot_style: str = "diverse",  # "diverse" or "single"
):
    def swiss_legal_translation_judge(question, options, answer, gold):
        system_prompt = SYSTEM[system_style]
        user = USER[system_style]
        few_shot_examples = FEW_SHOT[few_shot_style]
        instruction = f"""Judge the below case, give the brief reasoning process and the correctness score.


Source Text:
```{question}```

Golden Translation:
```{gold}```

Model's Translation:
```{answer}```

Your Judgment:
"""
        user_prompt = user + few_shot_examples + instruction

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return SampleLevelMetricGrouping(
        metric_name=[short_judge_name],
        higher_is_better={short_judge_name: True},
        category=MetricCategory.LLM_AS_JUDGE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=JudgeSwissLegalTranslation(
            judge_model_name=judge_model_name,
            template=swiss_legal_translation_judge,
            process_judge_response=process_judge_response_freeform_gpt,
            judge_backend=backend,
            short_judge_name=short_judge_name,
        ).compute,
        corpus_level_fn={short_judge_name: statistics.mean},
    )


class GEMBA:
    def __init__(self, method: str = "GEMBA-MQM_norm", model: str = "gpt-4o"):
        self.method = method
        self.model = model
        self.name = f"{method.split('_')[0]}_{model}"

    def compute(
        self,
        sample_ids: list[str],
        responses: list,
        formatted_docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        logger.info(f"Judging {len(formatted_docs)} samples with {self.name}...")
        source_langs = [formatted_doc.specific["source_lang"] for formatted_doc in formatted_docs]
        target_langs = [formatted_doc.specific["target_lang"] for formatted_doc in formatted_docs]
        # There should be only one language each in the batch
        assert len(set(source_langs)) == len(set(target_langs)) == 1
        sources = [formatted_doc.specific["source"] for formatted_doc in formatted_docs]
        predictions = [response[0].result for response in responses]

        answers, errors = get_gemba_scores(
            sources, predictions, source_langs[0], target_langs[0], method=self.method, model=self.model
        )

        # Handle cases where errors might be nan
        formatted_errors = []
        for error in errors:
            if isinstance(error, dict):
                # Convert defaultdict to dic
                formatted_errors.append([{key: value} for key, value in error.items()])
            else:
                formatted_errors.append([{"error": ["No error details available"]}])

        return [{self.name: answer, f"{self.name}_errors": error} for answer, error in zip(answers, formatted_errors)]


def get_gemba_judge(method: str = "GEMBA-MQM_norm", model: str = "gpt-4o"):
    name = f"{method.split('_')[0]}_{model}"
    return SampleLevelMetricGrouping(
        metric_name=[name],
        higher_is_better={name: True},
        category=MetricCategory.LLM_AS_JUDGE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=GEMBA(method=method, model=model).compute,
        corpus_level_fn={name: statistics.mean},
    )


def get_bert_score(language: str, num_layers: int = 24, model_type: str = "xlm-roberta-large", device: str = "cpu"):
    if device == "mps":
        raise ValueError("MPS is not supported for BERTScore")
    logger.info(
        f"Loading BERTScore with lang={language}, num_layers={num_layers}, model_type={model_type}, and device={device}..."
    )
    score = BertScore(normalize_gold=remove_braces, normalize_pred=remove_braces_and_strip)
    score.bert_scorer = BERTScorer(
        # We could download the files from here and set the baseline_path ourselves:
        # https://github.com/Tiiiger/bert_score/tree/master/bert_score/rescale_baseline
        model_type=model_type,
        lang=language,  # Needs to be set if rescale_with_baseline is True
        num_layers=num_layers,  # Needs to be set if rescale_with_baseline is True
        rescale_with_baseline=True,
        baseline_path=None,
        device=device,
    )

    if language == "rm":
        language = "it"
        logger.warning("There is no BERTScore baseline file for Rumantsch, using Italian instead.")

    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(score.bert_scorer.baseline_path), exist_ok=True)

    # Download the baseline file if it doesn't exist
    if not os.path.exists(score.bert_scorer.baseline_path):
        raw_url = f"https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/{language}/{model_type}.tsv"
        logger.info(f"Downloading BERTScore baseline file from {raw_url}")
        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(score.bert_scorer.baseline_path, "wb") as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download baseline file from {raw_url}")

    return SampleLevelMetricGrouping(
        metric_name=["BERTScore-P", "BERTScore-R", "BERTScore-F"],
        higher_is_better={
            "BERTScore-P": True,
            "BERTScore-R": True,
            "BERTScore-F": True,
        },
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=lambda *args, **kwargs: {k: v * 100 for k, v in score.compute(*args, **kwargs).items()},
        corpus_level_fn={
            "BERTScore-P": statistics.mean,
            "BERTScore-R": statistics.mean,
            "BERTScore-F": statistics.mean,
        },
    )


class BLEURT:
    def __init__(
        self,
        model_size: str = "tiny",
        seq_len: int = 512,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Creates a BLEURT scorer based on the model size (tiny, base, large) and sequence length (128, 512)."""
        assert model_size in [
            "tiny",
            "base",
            "large",
        ], "Model size must be either tiny, base, or large"
        assert seq_len in [128, 512], "Sequence length must be either 128 or 512"
        if device == "mps":
            raise ValueError("MPS is not supported for BLEURT")

        self.metric_name = f"bleurt_{model_size}"
        self.model_size = model_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

        # Lazy loading
        self.tokenizer = None
        self.model = None

    def _ensure_initialized(self):
        """Lazy initialization of model and tokenizer"""
        if self.tokenizer is None:
            logger.info(f"Loading BLEURT tokenizer {self.metric_name} lazily...")
            self.tokenizer = AutoTokenizer.from_pretrained(f"Elron/bleurt-{self.model_size}-{self.seq_len}")

        if self.model is None:
            logger.info(f"Loading BLEURT model {self.metric_name} lazily...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                f"Elron/bleurt-{self.model_size}-{self.seq_len}"
            )
            self.model = self.model.to(self.device)
            self.model.eval()

    def _process_batch(self, references: list[str], candidates: list[str]) -> list[float]:
        """Process a batch of references and candidates"""
        # Clean and prepare inputs
        references = [str(ref).strip() for ref in references]
        candidates = [str(cand).strip() for cand in candidates]

        # Tokenize
        inputs = self.tokenizer(
            references,
            candidates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_len,
        )

        # Log warning if any sequences were truncated
        if any(len(encoding) == self.seq_len for encoding in inputs["input_ids"]):
            logger.warning(f"Some inputs were truncated to max_length={self.seq_len} in BLEURT scoring")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)[0]

        return outputs.squeeze().cpu().tolist()

    def compute(
        self,
        sample_ids: list[str],
        responses: list,
        formatted_docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        """Compute BLEURT scores for a batch of translations"""
        self._ensure_initialized()

        logger.info(f"Scoring {len(formatted_docs)} samples with {self.metric_name}...")

        # Get references and predictions
        references = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]
        predictions = [response[0].result for response in responses]

        # Process in batches
        all_scores = []
        for i in tqdm(
            range(0, len(references), self.batch_size),
            desc=f"Processing batches of size {self.batch_size} with {self.metric_name}",
        ):
            batch_refs = references[i : i + self.batch_size]
            batch_preds = predictions[i : i + self.batch_size]
            try:
                scores = self._process_batch(batch_refs, batch_preds)
                all_scores.extend(scores if isinstance(scores, list) else [scores])
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                # Use minimum score for failed batches
                all_scores.extend([-1.0] * len(batch_refs))

        return [{self.metric_name: score * 100} for score in all_scores]


def get_bleurt(
    model_size: str = "tiny",
    seq_len: int = 512,
    batch_size: int = 32,
    device: str = "cpu",
):
    logger.info(
        f"Loading BLEURT with model_size={model_size}, seq_len={seq_len}, batch_size={batch_size}, and device={device}..."
    )
    name = f"bleurt_{model_size}"
    return SampleLevelMetricGrouping(
        metric_name=[name],
        higher_is_better={name: True},
        category=MetricCategory.LLM_AS_JUDGE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=BLEURT(model_size=model_size, seq_len=seq_len, batch_size=batch_size, device=device).compute,
        corpus_level_fn={name: statistics.mean},
    )


class COMET:
    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 8,
        gpus: int = 1,
        accelerator: str = "cpu",
    ):
        if accelerator == "mps":
            raise ValueError("MPS is not supported for COMET")

        self.metric_name = model_name.split("/")[-1]
        self.model = None  # Lazy loading of the model
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.accelerator = accelerator

    def compute(
        self,
        sample_ids: list[str],
        responses: list,
        formatted_docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        # Only load the model here to save memory and time
        if self.model is None:
            logger.info(f"Loading COMET model {self.model_name} lazily...")
            self.model = load_from_checkpoint(download_model(self.model_name))

        logger.info(f"Scoring {len(formatted_docs)} samples with {self.metric_name}...")
        golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]
        predictions = [response[0].result for response in responses]
        sources = [kwargs["formatted_doc"].specific["source"] for kwargs["formatted_doc"] in formatted_docs]

        data = [
            {"src": src, "mt": pred if isinstance(pred, str) else pred[0], "ref": gold}
            for src, pred, gold in zip(sources, predictions, golds)
        ]
        model_output = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self.gpus,
            accelerator=self.accelerator,
        )

        return [{self.metric_name: score * 100} for score in model_output["scores"]]


def get_comet(
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
    device: str = "cpu",
):
    logger.info(
        f"Loading COMET with model_name={model_name}, batch_size={batch_size}, gpus={gpus}, and device={device}..."
    )
    name = model_name.split("/")[-1]
    return SampleLevelMetricGrouping(
        metric_name=[name],
        higher_is_better={name: True},
        category=MetricCategory.LLM_AS_JUDGE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=COMET(
            model_name=model_name,
            batch_size=batch_size,
            gpus=gpus,
            accelerator=device,
        ).compute,
        corpus_level_fn={name: statistics.mean},
    )


class METEOR:
    def __init__(self, alpha=0.9, beta=3, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
        assert NLTK_VERSION >= version.Version("3.9.0"), "NLTK version must be >= 3.9.0"
        nltk.download("punkt_tab")
        nltk.download("wordnet")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        if isinstance(golds[0], list):  # multiple references
            scores = [
                meteor_score.meteor_score(
                    [word_tokenize(ref) for ref in refs],
                    word_tokenize(pred),
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                for refs, pred in zip(golds, predictions)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(
                    word_tokenize(ref),
                    word_tokenize(pred),
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                for ref, pred in zip(golds, predictions)
            ]

        return statistics.mean(scores) * 100


def get_meteor():
    return SampleLevelMetric(
        metric_name="meteor",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=METEOR().compute,
        corpus_level_fn=statistics.mean,
    )


class BLEU:
    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """
        Compute BLEU score for a list of predictions against their references.

        Args:
            golds: List of reference strings
            predictions: List of prediction strings

        Returns:
            Mean BLEU score scaled to 0-100
        """
        scores = []
        for ref, pred in zip(golds, predictions):
            scores.append(sentence_bleu(pred, [ref]).score)

        return statistics.mean(scores) * 100


def get_bleu_sentence():
    return SampleLevelMetric(
        metric_name="bleu_sentence",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=BLEU().compute,
        corpus_level_fn=statistics.mean,
    )


class CHRF:
    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """
        Compute chrF score for a list of predictions against their references.
        """
        scores = []
        for ref, pred in zip(golds, predictions):
            scores.append(sentence_chrf(pred, [ref]).score)

        return statistics.mean(scores) * 100


def get_chrf_sentence():
    return SampleLevelMetric(
        metric_name="chrf_sentence",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=CHRF().compute,
        corpus_level_fn=statistics.mean,
    )


class TER:
    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """
        Compute TER score for a list of predictions against their references.
        """
        scores = []
        for ref, pred in zip(golds, predictions):
            scores.append(sentence_ter(pred, [ref]).score)

        return statistics.mean(scores) * 100


def get_ter_sentence():
    return SampleLevelMetric(
        metric_name="ter_sentence",
        higher_is_better=False,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        sample_level_fn=TER().compute,
        corpus_level_fn=statistics.mean,
    )


# EVALS WITH SUBSET


def create_translation_pairs(langs_list: list) -> list[tuple]:
    """
    Create all possible translation pairs from a given list of languages.

    Args:
    langs_list (list): A list of languages.

    Returns:
    lang_pairs_list (list): A list of tuples representing a translation pair.
    """
    lang_pairs_list = []
    for i, lang1 in enumerate(langs_list):
        for lang2 in langs_list[i + 1 :]:
            lang_pairs_list.append((lang1, lang2))
            lang_pairs_list.append((lang2, lang1))
    return lang_pairs_list


@dataclass
class LevelConfig:
    name: str
    text_col_name: str
    metadata_cols: list[str]
    generation_size: int
    stop_sequence: list[str]  # just "\n" leads to problems for anthropic models, maybe we need a special case there


@dataclass
class DatasetConfig:
    name: str
    hf_repo: str
    languages: list[str]
    subsets: dict[str, LevelConfig]

    def __post_init__(self):
        self.translation_pairs = create_translation_pairs(self.languages)


# Translation of Swiss Federal Supreme Court Decision Summaries on three levels: the entire decision, the regeste level and the text level.
SwissDecisionSummaryTranslations = DatasetConfig(
    name="sdst",
    hf_repo="joelniklaus/SwissDecisionSummaryTranslations",
    languages=["de", "fr", "it"],
    subsets={
        "bge_level": LevelConfig(
            name="bge_level",
            text_col_name="bgeText",
            metadata_cols=["bge"],
            generation_size=2048,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "regeste_level": LevelConfig(
            name="regeste_level",
            text_col_name="regesteText",
            metadata_cols=["bge"],
            generation_size=512,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "text_level": LevelConfig(
            name="text_level",
            text_col_name="text",
            metadata_cols=["bge"],
            generation_size=256,
            stop_sequence=["</s>", ".\n", "\n"],
        ),
    },
)

# Translation of Swiss Federal Laws on three levels: the entire law, the article level and the paragraph level.
SwissLawTranslations = DatasetConfig(
    name="slt",
    hf_repo="joelniklaus/SwissLawTranslations",
    languages=["de", "fr", "it", "rm", "en"],
    subsets={
        "law_level": LevelConfig(
            name="law_level",
            text_col_name="lawText",
            metadata_cols=["rsNr"],
            generation_size=16384,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "article_level": LevelConfig(
            name="article_level",
            text_col_name="artText",
            metadata_cols=["rsNr"],
            generation_size=1024,
            stop_sequence=["</s>", ".\n\n", "\n\n"],
        ),
        "paragraph_level": LevelConfig(
            name="paragraph_level",
            text_col_name="parText",
            metadata_cols=["rsNr"],
            generation_size=256,
            stop_sequence=["</s>", ".\n", "\n"],
        ),
    },
)

# Translation of Swiss Federal Supreme Court Press Releases on one level: the entire press release.
SwissSupremeCourtPressReleaseTranslations = DatasetConfig(
    name="sscprt",
    hf_repo="joelniklaus/SwissSupremeCourtPressReleaseTranslations",
    languages=["de", "fr", "it"],
    subsets={
        "press_release": LevelConfig(
            name="press_release",
            text_col_name="text",
            metadata_cols=["filename"],
            generation_size=1024,
            stop_sequence=["</s>"],
        )
    },
)


def create_prompt_fn(level_config: LevelConfig, source_lang: str, target_lang: str):
    """
    Create a prompt function for a given level configuration.
    """
    text_col = level_config.text_col_name
    src_text_col = f"{source_lang}_{text_col}"
    target_text_col = f"{target_lang}_{text_col}"

    def prompt_fn(line: dict, task_name: str = None):
        # Following Template A from https://github.com/huggingface/lighteval/pull/389#issuecomment-2471580177
        custom_query = f"{source_lang.upper()}: {line[src_text_col]}\n{target_lang.upper()}: "

        return Doc(
            task_name=task_name,
            query=custom_query,
            choices=[str(line[target_text_col])],
            gold_index=0,
            specific={
                **{col: line[col] for col in level_config.metadata_cols},
                "question": custom_query,
                "source": line[src_text_col],
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )

    return prompt_fn


JUDGE_MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18",
    "gpt-4o": "openai/gpt-4o-2024-11-20",
    "gemini-1-5-flash": "gemini/gemini-1.5-flash-002",
    "gemini-1-5-pro": "gemini/gemini-1.5-pro-002",
    # The Claude models do not follow the required output format.
    # "claude-3-5-haiku": "anthropic/claude-3-5-haiku-20241022",
    # "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet-20241022",
}

LEXICAL_METRICS = [
    "bleu",
    "chrf",
    "bleu_sentence",
    "chrf_sentence",
    "ter_sentence",
    "meteor",
]
GPU_METRICS = [
    "bert_score",
    "bleurt_large",
    "xcomet_xxl",
]
API_METRICS = [
    "gemba_mqm_gpt_4o",
    "slt_judge_gpt_4o",
]
JUDGE_METRICS = [
    f"slt_judge_{judge_model}-{system_style}-{few_shot_style}".replace("-", "_")
    for judge_model in JUDGE_MODELS
    for system_style in ["basic", "detailed"]
    for few_shot_style in ["diverse", "single"]
]

metrics_to_evaluate = ["judge"]

METRICS_TO_USE = []
if metrics_to_evaluate == ["debug"]:
    METRICS_TO_USE = ["bleu"]
elif "lexical" in metrics_to_evaluate:
    METRICS_TO_USE += LEXICAL_METRICS
elif "gpu" in metrics_to_evaluate:
    METRICS_TO_USE += GPU_METRICS
elif "api" in metrics_to_evaluate:
    METRICS_TO_USE += API_METRICS
elif "judge" in metrics_to_evaluate:
    METRICS_TO_USE += JUDGE_METRICS
else:
    METRICS_TO_USE = LEXICAL_METRICS + GPU_METRICS + API_METRICS
logger.info(f"Available metrics: {METRICS_TO_USE}")

METRICS = {}


def init_lexical_metric(metric_name: str):
    # Corpus level metrics
    if metric_name == "bleu":
        METRICS["bleu"] = Metrics.bleu
    if metric_name == "chrf":
        METRICS["chrf"] = Metrics.chrf
    if metric_name == "ter":
        # TER often hangs for a while and takes more than 10 minutes to compute
        METRICS["ter"] = Metrics.ter
    # Sample level metrics
    if metric_name == "bleu_sentence":
        METRICS["bleu_sentence"] = get_bleu_sentence()
    if metric_name == "chrf_sentence":
        METRICS["chrf_sentence"] = get_chrf_sentence()
    if metric_name == "ter_sentence":
        METRICS["ter_sentence"] = get_ter_sentence()
    if metric_name == "meteor":
        METRICS["meteor"] = get_meteor()


def init_model_based_metric(metric_name: str):
    if metric_name == "bert_score":
        METRICS["bert_score"] = {  # Create BERTScore metrics for each language
            lang: get_bert_score(language=lang, model_type="xlm-roberta-large", device=device)
            for lang in ["de", "fr", "it", "rm", "en"]
        }
    if metric_name == "bleurt_tiny":
        METRICS["bleurt_tiny"] = get_bleurt(model_size="tiny", seq_len=512, batch_size=256, device=device)
    if metric_name == "bleurt_base":
        METRICS["bleurt_base"] = get_bleurt(model_size="base", seq_len=512, batch_size=256, device=device)
    if metric_name == "bleurt_large":
        METRICS["bleurt_large"] = get_bleurt(model_size="large", seq_len=512, batch_size=256, device=device)
    # There are also reference-free models (e.g., Unbabel/wmt22-cometkiwi-da), but since we have reference gold labels, we use the reference-based models.
    if metric_name == "wmt22-comet-da":
        METRICS["wmt22-comet-da"] = get_comet(
            model_name="Unbabel/wmt22-comet-da", batch_size=64, gpus=1, device=device
        )
    if metric_name == "xcomet_xl":
        METRICS["xcomet_xl"] = get_comet(model_name="Unbabel/XCOMET-XL", batch_size=32, gpus=1, device=device)
    if metric_name == "xcomet_xxl":
        METRICS["xcomet_xxl"] = get_comet(model_name="Unbabel/XCOMET-XXL", batch_size=16, gpus=1, device=device)


def init_llm_judge_metric(metric_name: str):
    if metric_name == "gemba_mqm_gpt_4o":
        METRICS["gemba_mqm_gpt_4o"] = get_gemba_judge(method="GEMBA-MQM_norm", model="gpt-4o")

    if metric_name == "slt_judge_gpt_4o":
        METRICS["slt_judge_gpt_4o"] = get_swiss_legal_translation_judge(
            judge_model_name="openai/gpt-4o-2024-11-20",
            short_judge_name="slt_judge_gpt-4o",
        )

    # Check all the judge metric combinations
    for judge_model in JUDGE_MODELS:
        for system_style in ["basic", "detailed"]:
            for few_shot_style in ["diverse", "single"]:
                short_judge_name = f"slt_judge_{judge_model}-{system_style}-{few_shot_style}"
                judge_metric_name = short_judge_name.replace("-", "_")
                if metric_name == judge_metric_name:
                    METRICS[metric_name] = get_swiss_legal_translation_judge(
                        judge_model_name=JUDGE_MODELS[judge_model],
                        short_judge_name=short_judge_name,
                        system_style=system_style,
                        few_shot_style=few_shot_style,
                    )
                    break


def init_metric(metric_name: str):
    # Only load the metric once
    if metric_name in METRICS:
        logger.debug(f"Metric {metric_name} already initialized")
        return

    # ===== Lexical metrics =====
    init_lexical_metric(metric_name)
    # ===== Model-based metrics =====
    init_model_based_metric(metric_name)
    # ===== LLM Judge metrics =====
    init_llm_judge_metric(metric_name)


# Additionally we could consider adding the following open source judge models:
# flowaicom/Flow-Judge-v0.1, prometheus-eval/prometheus-7b-v2.0
# However, these are only fine-tuned on English data and we need multilingual support.


def get_metrics(METRICS_TO_USE, target_lang: str, generation_size: int):
    metrics = []
    for metric in METRICS_TO_USE:
        # These metrics are sentence level metrics and we only want to use them for generation sizes up to 512.
        short_metrics = [
            "bleu_sentence",
            "chrf_sentence",
            "ter_sentence",
            "bert_score",
            "bleurt_tiny",
            "bleurt_base",
            "bleurt_large",
            "wmt22-comet-da",
            "xcomet_xl",
            "xcomet_xxl",
        ]
        if generation_size > 512 and metric in short_metrics:
            logger.debug(
                f"Skipping {metric} for generation size {generation_size} because the maximum supported sequence length is 512."
            )
            continue

        init_metric(metric)
        if metric == "bert_score":
            # Add only the BERTScore for the target language
            metrics.append(METRICS["bert_score"][target_lang])
        else:
            metrics.append(METRICS[metric])
    return metrics


class TranslationTask(LightevalTaskConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        level_name: str,
        source_lang: str,
        target_lang: str,
    ):
        level_config = dataset_config.subsets[level_name]
        super().__init__(
            name=f"{dataset_config.name}-{level_name}:{source_lang}-{target_lang}",
            suite=["community"],
            prompt_function=create_prompt_fn(level_config, source_lang, target_lang),
            hf_repo=dataset_config.hf_repo,
            hf_subset=level_name,
            hf_filter=None,
            hf_avail_splits=["train", "validation", "test"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=level_config.generation_size,
            metric=get_metrics(METRICS_TO_USE, target_lang, level_config.generation_size),
            stop_sequence=level_config.stop_sequence,
            trust_dataset=True,
            # Remove the target language in the beginning if it exists: e.g., FR: {translation}
            # Is only applied to the generative metrics, but also there seems not to be invoked, maybe not passed through?
            # output_regex=f"(?:{target_lang.upper()}:\s*?)?(.*)",
        )


# STORE YOUR EVALS

# list of all the subsets to use for this eval
DATASETS = [
    SwissDecisionSummaryTranslations,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
]

TASKS_TABLE = [
    TranslationTask(
        dataset_config=dataset,
        level_name=subset,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    for dataset in DATASETS
    for subset in dataset.subsets
    for source_lang, target_lang in dataset.translation_pairs
]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
