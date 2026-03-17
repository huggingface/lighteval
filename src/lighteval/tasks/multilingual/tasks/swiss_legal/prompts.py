from textwrap import dedent


SWISS_LEGAL_TRANSLATION_JUDGE_SYSTEM_PROMPT = {
    "basic": (
        "Act as a Judge specializing in the evaluation of translations of Swiss legal documents. "
        "Your task is to assess the accuracy, clarity, and fidelity of the model's translation "
        "to the golden translation, while considering the nuances of legal language."
    ),
    "detailed": (
        "You are a senior legal translator and quality assurance specialist with over 20 years of "
        "experience in Swiss law, certified by the Swiss Sworn Translators Association (Association "
        "suisse des traducteurs-jurés, ASTJ). You possess native-level proficiency in all Swiss "
        "national languages (German, French, Italian, and Romansh) as well as English, enabling "
        "precise evaluation of legal nuances across all linguistic combinations. Your task is to "
        "evaluate machine-translated legal texts for accuracy, clarity and fidelity to Swiss legal "
        "standards analyzing the subtle complexities of legal language. You excel at identifying "
        "even minor discrepancies and calibrating evaluation scores appropriately to reflect the "
        "severity of each error."
    ),
}

SWISS_LEGAL_TRANSLATION_JUDGE_USER_PROMPT = {
    "basic": (
        "You will be provided with a source text, its golden translation, and the model's "
        "translation. Your task is to judge how correct the model's translation is based on the "
        "golden translation, and then give a correctness score. The correctness score should be "
        "one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, "
        "0.9, or 1.0 (totally right). You should first briefly give your reasoning process "
        "regarding how the model's translation conforms to or contradicts the golden translation, "
        "and then give the correctness score. The correctness score must strictly follow this format: "
        '"[[score]]", e.g., "The correctness score: [[0.5]]". Below are some examples.\n\n'
    ),
    "detailed": dedent(
        """
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
        The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples.\n
        """
    ).lstrip(),
}

SWISS_LEGAL_TRANSLATION_JUDGE_FEW_SHOT_EXAMPLES = {
    "diverse": dedent(
        """
        Example 1:
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
        """
    ).lstrip(),
    "single": dedent(
        """
        Example 1:
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
        """
    ).lstrip(),
}

SWISS_LEGAL_TRANSLATION_JUDGE_INSTRUCTION = dedent(
    """
    Judge the below case, give the brief reasoning process and the correctness score.


    Source Text:
    ```{question}```

    Golden Translation:
    ```{gold}```

    Model's Translation:
    ```{answer}```

    Your Judgment:
    """
).lstrip()


SLDS_JUDGE_SYSTEM_PROMPT = dedent(
    """
    You are a senior legal expert and quality assurance specialist with over 20 years of experience in Swiss law. You possess native-level proficiency in German, French, and Italian, enabling you to evaluate Swiss Federal Supreme Court headnotes with precision. Your task is to compare the **Official (Gold) Headnote** with a **Model-Generated Headnote** and provide a structured evaluation in five categories. You will carefully analyze each category and provide a short analysis before committing to a score. The categories are:

    1. Accuracy & Faithfulness: How well does the Model-Generated Headnote match the essential legal meaning and intent of the Official Headnote?
    2. Completeness & Relevance: Does the Model-Generated Headnote include all important points that the Official Headnote emphasizes, without adding irrelevant details?
    3. Clarity & Coherence: Is the text well-organized, easy to understand, and coherent in style and structure?
    4. Articles: Do the same legal articles (prefixed “Art.”) appear correctly and completely in the Model-Generated Headnote as in the Official Headnote?
    5. Considerations: Do the same considerations (prefixed “E.” in German or “consid.” in French/Italian) appear correctly and completely in the Model-Generated Headnote as in the Official Headnote?

    For each category, provide a short and concise explanation followed by a score on a scale from 1 to 3:

    1: Fails or is substantially flawed.
    Major omissions or inaccuracies that fundamentally alter the legal meaning.

    2: Largely correct but missing key element(s).
    Generally captures the substance, yet lacks one or more important details or references.

    3: Closely matches the Official Headnote.
    Covers all critical aspects and references with only minor wording variations that do not affect the legal content.

    Your output must follow the exact structure provided below to ensure consistency and ease of parsing.
    """
).strip()

SLDS_JUDGE_USER_PROMPT = dedent(
    """
    Below are two headnotes for the same leading decision from the Swiss Federal Supreme Court. Please compare the Model-Generated Headnote to the Official (Gold) Headnote according to the following five categories: Accuracy & Faithfulness, Completeness & Relevance, Clarity & Coherence, Articles, and Considerations.

    1. Analyze the Model-Generated Headnote in comparison to the Official Headnote for each category.
    2. Provide a short explanation for your evaluation in each category.
    3. Conclude each category with a score in the exact format: CATEGORYNAME_SCORE: [X], where X is an integer from 1 to 3.

    Required Output Format:

    ACCURACY_FAITHFULNESS:
    Analysis: [Your concise analysis here]
    ACCURACY_FAITHFULNESS_SCORE: [X]

    COMPLETENESS_RELEVANCE:
    Analysis: [Your concise analysis here]
    COMPLETENESS_RELEVANCE_SCORE: [X]

    CLARITY_COHERENCE:
    Analysis: [Your concise analysis here]
    CLARITY_COHERENCE_SCORE: [X]

    ARTICLES:
    Analysis: [Your concise analysis here]
    ARTICLES_SCORE: [X]

    CONSIDERATIONS:
    Analysis: [Your concise analysis here]
    CONSIDERATIONS_SCORE: [X]

    Please adhere strictly to this structure.

    Here is an example how your response could look like:

    <EXAMPLE>
    {one_shot_example}
    </EXAMPLE>

    Now evaluate the Model-Generated Headnote against the Official (Gold) Headnote on the headnotes below.

    Official (Gold) Headnote:
    {original_headnote}

    Model-Generated Headnote:
    {generated_headnote}
    """
).strip()

SLDS_JUDGE_ONE_SHOT_EXAMPLE_DE = dedent(
    """
    Official (Gold) Headnote:
    Art. 391 Abs. 2 StPO; die erstmalige Anordnung einer ambulanten Massnahme durch das Berufungsgericht verstösst gegen das Verschlechterungsverbot (reformatio in peius). Verzichtet das erstinstanzliche Gericht auf die Anordnung einer beantragten ambulanten Massnahme und hat die Staatsanwaltschaft in ihrer Anschlussberufung deren Anordnung nicht erneut beantragt, verletzt das Berufungsgericht das Verschlechterungsverbot, wenn es eine ambulante Massnahme anordnet (E. 4.1-4.4).

    Model-Generated Headnote:
    Art. 391 Abs. 2 StPO; Verschlechterungsverbot; ambulante Massnahme nach Art. 63 StGB. Die erstmalige Anordnung einer ambulanten Massnahme im Berufungsverfahren verletzt das Verschlechterungsverbot nach Art. 391 Abs. 2 StPO, wenn erstinstanzlich auf eine solche Massnahme verzichtet wurde und die Staatsanwaltschaft in ihrer Anschlussberufung keinen entsprechenden Antrag gestellt hat. Eine solche Anordnung stellt eine unzulässige reformatio in peius dar, da der Beschuldigte das Risiko einer nachträglichen Anpassung oder Umwandlung der Massnahme nur dann trägt, wenn bereits erstinstanzlich eine therapeutische Massnahme angeordnet wurde (E. 4.4). Die freiwillige Teilnahme an einer Therapie durch den Beschuldigten begründet keine ausreichende Grundlage für die erstmalige Anordnung einer ambulanten Massnahme im Berufungsverfahren (E. 4.3).

    ACCURACY_FAITHFULNESS:
    Analysis: The model-generated headnotes captures the legal essence and intent of the official headnote accurately. It correctly explains the legal principle of reformatio in peius and the violation of the prohibition of worsening in the context of ordering outpatient measures.
    ACCURACY_FAITHFULNESS_SCORE: 3

    COMPLETENESS_RELEVANCE:
    Analysis: The model-generated headnote inclundes all relevant aspects of the official headnote, such as the prohibition of worsening, the legal context of ordering outpatient measures, and the implications of the prosecution's appeal. However, it also adds additional details regarding the voluntary participation in therapy, which are not explicitly mentioned in the official headnote. These are very specific details that may not be necessary for a concise headnote.
    COMPLETENESS_RELEVANCE_SCORE: 2

    CLARITY_COHERENCE:
    Analysis: The model-generated headnote is easy to understand and coherent. The clarity could be improved by not repeating the fact that the outpatient measure violates the prohibition of worsening twice. Other than that, the structure and style are clear and well-organized.
    CLARITY_COHERENCE_SCORE: 2

    ARTICLES:
    Analysis: While both the official and model-generated headnotes correctly reference Art. 391 Abs. 2 StPO, the model-generated headnote also includes Art. 63 StGB, which is not explicitly mentioned in the official headnote. This additional reference is relevant to the context of outpatient measures but may not be necessary for a concise headnote.
    ARTICLES_SCORE: 2

    CONSIDERATIONS:
    Analysis: The model-generated headnote only cites considerations E. 4.3 and E. 4.4, whereas the official headnote includes considerations E. 4.1 to E. 4.4. This omission reduces the completeness of the model-generated headnote.
    CONSIDERATIONS_SCORE: 2
    """
).strip()

SLDS_JUDGE_ONE_SHOT_EXAMPLE_FR = dedent(
    """
    Official (Gold) Headnote:
    Travail externe (art. 77a CP). Le condamné à une peine privative de liberté qui a passé une longue période en détention provisoire peut exécuter sa peine (restante) directement sous la forme de travail externe s'il réalise les conditions posées à l'art. 77a al. 1 CP. Il n'a pas nécessairement besoin d'être privé de sa liberté au moment où est prise la décision ordonnant le travail externe (consid. 2.5.2).

    Model-Generated Headnote:
    Art. 77a CP; art. 164 ss RSPC; conditions pour l'octroi du travail externe. Le travail externe, en tant que phase de l'élargissement progressif de l'exécution de la peine, ne nécessite pas impérativement un séjour préalable en milieu ouvert ou dans une section ouverte d'un établissement fermé. L'utilisation du terme "en principe" à l'art. 77a al. 2 CP indique que cette condition n'est pas absolue et peut être adaptée aux circonstances individuelles, notamment lorsque le condamné a déjà purgé une partie significative de sa peine en détention provisoire ou pour des motifs de sûreté. La cour cantonale a violé le droit fédéral en exigeant de manière rigide un séjour en milieu ouvert comme condition préalable à l'octroi du travail externe. Le recours est admis, l'arrêt cantonal est annulé, et la cause est renvoyée pour une nouvelle décision sur la base des conditions prévues à l'art. 77a CP (consid. 2.5.1 à 2.5.3).

    ACCURACY_FAITHFULNESS:
    Analysis: The model-generated headnote accurately reflects the legal principle and conditions for granting external work under Art. 77a CP.
    ACCURACY_FAITHFULNESS_SCORE: 3

    COMPLETENESS_RELEVANCE:
    Analysis: The model-generated headnote includes all relevant aspects of the official headnote. However, it adds additional details regarding the use of the term "en principe" and the violation of federal law by the cantonal court. While these details provide context, they are not essential for a concise headnote that shapes future legislation.
    COMPLETENESS_RELEVANCE_SCORE: 2

    CLARITY_COHERENCE:
    Analysis: The model-generated headnote is clear and well-organized, but the inclusion of specific details may obscure the broader legal principle.
    CLARITY_COHERENCE_SCORE: 2

    ARTICLES:
    Analysis: The model-generated headnote includes extra legal articles (Art. 164 ff. RSPC) not cited in the official headnote, deviating from its intended focus. Besides this, the reference to Art. 77a CP aligns with the official headnote.
    ARTICLES_SCORE: 2

    CONSIDERATIONS:
    Analysis: The official headnote only cites consideration E 2.5.2, while the model-generated headnote references considerations E 2.5.1 to E 2.5.3. This makes the model-generated headnote less precise.
    CONSIDERATIONS_SCORE: 2
    """
).strip()

SLDS_JUDGE_ONE_SHOT_EXAMPLE_IT = dedent(
    """
    Official (Gold) Headnote:
    Art. 24 Cost.; libertà di domicilio; residenza secondaria per motivi professionali; autorizzazione di soggiorno.  Contenuto e portata della libertà di domicilio garantita dall'art. 24 Cost. (consid. 3.2.1). In tale ambito le competenze dei Cantoni e dei Comuni sono circoscritte al controllo e alla registrazione degli abitanti (consid. 3.3.1). L'obbligo di annunciarsi, in caso di soggiorno in un Comune differente da quello del domicilio per motivi professionali, non limita l'esercizio della libertà di domicilio. Detto annuncio non può però essere sottoposto ad un regime autorizzativo (consid. 3.3.3).

    Model-Generated Headnote:
    Art. 24 Cost.; libertà di domicilio; regime autorizzativo per il soggiorno; competenze comunali. La libertà di domicilio garantita dall'art. 24 Cost. consente a ogni cittadino svizzero di eleggere liberamente il proprio luogo di residenza o soggiorno sul territorio nazionale, senza che ciò possa essere sottoposto a un regime autorizzativo di polizia. Le autorità comunali possono richiedere unicamente una notifica al momento dell'arrivo nel Comune, finalizzata alla registrazione e al controllo degli abitanti, ma non possono imporre un'autorizzazione di soggiorno con scadenza annuale, poiché ciò costituirebbe una violazione della libertà di domicilio (consid. 3.3.3). Il sistema instaurato dalle Direttive MOVPOP, che prevede il rilascio di un'"autorizzazione di soggiorno" con validità limitata, deve essere interpretato nel senso che l'autorità comunale può solo certificare formalmente la notifica del soggiorno, senza sottoporre quest'ultimo a un regime autorizzativo (consid. 3.3.2 e 3.3.3). La conferma di un tale regime da parte del Tribunale cantonale amministrativo viola pertanto l'art. 24 Cost. e deve essere annullata (consid. 3.4).

    ACCURACY_FAITHFULNESS:
    Analysis: The model-generated headnote aligns with the core legal meaning but includes additional details (e.g., MOVPOP directives) not in the official headnote. These do not conflict but shift the focus slightly.
    ACCURACY_FAITHFULNESS_SCORE: 2

    COMPLETENESS_RELEVANCE:
    Analysis: The model-generated headnote captures key points but omits emphasis on secondary residence for professional reasons and cantonal/communal roles. Irrelevant details (e.g., MOVPOP) add complexity.
    COMPLETENESS_RELEVANCE_SCORE: 2

    CLARITY_COHERENCE:
    Analysis: The model-generated headnote is clear and organized, but additional elements like MOVPOP reduce coherence by shifting focus away from the main points and making the text longer and more complex.
    CLARITY_COHERENCE_SCORE: 2

    ARTICLES:
    Analysis: References to Art. 24 Cost. are correct and complete.
    ARTICLES_SCORE: 3

    CONSIDERATIONS:
    Analysis: The model-generated headnote correctly references consid. 3.3.3 but adds consid. 3.3.2 and 3.4, which are beyond the official headnote's scope. Moreover, it leaves out consid 3.2.1 and 3.3.1, reducing precision. Instead, it mentiones consid. 3.3.3 twice, which is redundant.
    CONSIDERATIONS_SCORE: 1
    """
).strip()
