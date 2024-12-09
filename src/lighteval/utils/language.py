# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class Language(Enum):
    ENGLISH = "eng"
    SPANISH = "spa"
    PORTUGUESE = "por"
    ITALIAN = "ita"
    FRENCH = "fra"
    ROMANIAN = "ron"
    GERMAN = "deu"
    LATIN = "lat"
    CZECH = "ces"
    DANISH = "dan"
    FINNISH = "fin"
    GREEK = "ell"
    NORWEGIAN = "nor"
    POLISH = "pol"
    RUSSIAN = "rus"
    SLOVENIAN = "slv"
    SWEDISH = "swe"
    TURKISH = "tur"
    DUTCH = "nld"
    CHINESE = "zho"
    JAPANESE = "jpn"
    VIETNAMESE = "vie"
    INDONESIAN = "ind"
    PERSIAN = "fas"
    KOREAN = "kor"
    ARABIC = "ara"
    THAI = "tha"
    HINDI = "hin"
    BENGALI = "ben"
    TAMIL = "tam"
    HUNGARIAN = "hun"
    UKRAINIAN = "ukr"
    SLOVAK = "slk"
    BULGARIAN = "bul"
    CATALAN = "cat"
    CROATIAN = "hrv"
    SERBIAN = "srp"
    LITHUANIAN = "lit"
    ESTONIAN = "est"
    HEBREW = "heb"
    LATVIAN = "lav"
    SERBOCROATIAN = "hbs"  # Deprecated
    ALBANIAN = "sqi"
    AZERBAIJANI = "aze"
    ICELANDIC = "isl"
    MACEDONIAN = "mkd"
    GEORGIAN = "kat"
    GALICIAN = "glg"
    ARMENIAN = "hye"
    BASQUE = "eus"
    SWAHILI = "swa"
    MALAY = "msa"
    TAGALOG = "tgl"
    JAVANESE = "jav"
    PUNJABI = "pan"
    BIHARI = "bih"  # Deprecated
    GUJARATI = "guj"
    YORUBA = "yor"
    MARATHI = "mar"
    URDU = "urd"
    AMHARIC = "amh"
    TELUGU = "tel"
    HAITIAN = "hti"
    MALAYALAM = "mal"
    KANNADA = "kan"
    NEPALI = "nep"
    KAZAKH = "kaz"
    BELARUSIAN = "bel"
    BURMESE = "mya"
    ESPERANTO = "epo"
    UZBEK = "uzb"
    KHMER = "khm"
    TAJIK = "tgk"
    WELSH = "cym"
    NORWEGIAN_NYNORSK = "nno"
    BOSNIAN = "bos"
    SINHALA = "sin"
    TATAR = "tat"
    AFRIKAANS = "afr"
    ORIYA = "ori"
    KIRGHIZ = "kir"
    IRISH = "gle"
    OCCITAN = "oci"
    KURDISH = "kur"
    LAO = "lao"
    LUXEMBOURGISH = "ltz"
    BASHKIR = "bak"
    WESTERN_FRISIAN = "fry"
    PASHTO = "pus"
    MALTESE = "mlt"
    BRETON = "bre"
    ASSAMESE = "asm"
    MALAGASY = "mlg"
    DIVEHI = "div"
    YIDDISH = "yid"
    SOMALI = "som"
    SANSKRIT = "san"
    SINDHI = "snd"
    QUECHUA = "que"
    TURKMEN = "tuk"
    SOUTH_AZERBAIJANI = "azb"
    SORANI = "ckb"
    CEBUANO = "ceb"
    WAR = "war"
    SHAN = "shn"
    UDMURT = "udm"
    ZULU = "zul"


# This mapping was created for beleble, it converts iso_639_3 individual codes to iso_639_3 macro codes
# However it requires iso639-lang package and I don't see a point installing it just for this mapping
# Code to generate:
# ````
# from langcodes import Language
# from iso639 import Lang

# dst = get_dataset_config_names("facebook/belebele")
# output = {}
# for i in dst:
#     lang_old = Lang(Language.get(i).language)
#     lang = lang_old.macro() if lang_old.macro() else lang_old
#     output[lang_old.pt3] = lang.pt3
# ```

iso_639_3_ind_to_iso_639_3_macro = {
    "acm": Language.ARABIC,
    "arz": Language.ARABIC,
    "ceb": Language.CEBUANO,
    "fin": Language.FINNISH,
    "hin": Language.HINDI,
    "ita": Language.ITALIAN,
    "khm": Language.KHMER,
    "lvs": Language.LATVIAN,
    "npi": Language.NEPALI,
    "pol": Language.POLISH,
    "slv": Language.SLOVENIAN,
    "swe": Language.SWEDISH,
    #  'tso': Language.TSONGA,
    #  'xho': Language.XHOSA,
    "afr": Language.AFRIKAANS,
    "asm": Language.ASSAMESE,
    "ces": Language.CZECH,
    "fra": Language.FRENCH,
    "jav": Language.JAVANESE,
    #  'kin': Language.KINYARWANDA,
    "mal": Language.MALAYALAM,
    "por": Language.PORTUGUESE,
    #  'sna': Language.SHONA,
    "swh": Language.SWAHILI,
    "tur": Language.TURKISH,
    "yor": Language.YORUBA,
    "als": Language.ALBANIAN,
    "azj": Language.AZERBAIJANI,
    "ckb": Language.KURDISH,
    #  'fuv': Language.FULAH,
    "hrv": Language.CROATIAN,
    "jpn": Language.JAPANESE,
    "kir": Language.KIRGHIZ,
    "mar": Language.MARATHI,
    #  'nso': Language.NORTHERN_SOTHO,
    "snd": Language.SINDHI,
    "tam": Language.TAMIL,
    "ukr": Language.UKRAINIAN,
    "zho": Language.CHINESE,
    "amh": Language.AMHARIC,
    #  'bam': Language.BAMBARA,
    "dan": Language.DANISH,
    #  'gaz': Language.OROMO,
    "hun": Language.HUNGARIAN,
    #  'kac': Language.KACHIN,
    "kor": Language.KOREAN,
    "mkd": Language.MACEDONIAN,
    #  'nya': Language.CHICHEWA,
    "ron": Language.ROMANIAN,
    "som": Language.SOMALI,
    "tel": Language.TELUGU,
    "urd": Language.URDU,
    "apc": Language.ARABIC,
    "ben": Language.BENGALI,
    "deu": Language.GERMAN,
    #  'grn': Language.GUARANI,
    "hye": Language.ARMENIAN,
    "kan": Language.KANNADA,
    "lao": Language.LAO,
    "mlt": Language.MALTESE,
    "ory": Language.ORIYA,
    "rus": Language.RUSSIAN,
    #  'sot': Language.SOUTHERN_SOTHO,
    "tgk": Language.TAJIK,
    "zsm": Language.MALAY,
    "arb": Language.ARABIC,
    "ell": Language.GREEK,
    "guj": Language.GUJARATI,
    #  'ibo': Language.IGBO,
    "kat": Language.GEORGIAN,
    #  'lin': Language.LINGALA,
    #  'mri': Language.MAORI,
    "pan": Language.PUNJABI,
    "shn": Language.SHAN,
    "spa": Language.SPANISH,
    "fil": Language.TAGALOG,
    "uzn": Language.UZBEK,
    #  'zul': Language.ZULU,
    #  'bod': Language.TIBETAN,
    "eng": Language.ENGLISH,
    "hat": Language.HAITIAN,
    #  'ilo': Language.ILOCANO,
    "kaz": Language.KAZAKH,
    "lit": Language.LITHUANIAN,
    "mya": Language.BURMESE,
    "pbt": Language.PASHTO,
    "sin": Language.SINHALA,
    "srp": Language.SERBIAN,
    "tha": Language.THAI,
    "vie": Language.VIETNAMESE,
    "ars": Language.ARABIC,
    "bul": Language.BULGARIAN,
    "est": Language.ESTONIAN,
    "udm": Language.UDMURT,
    #  'hau': Language.HAUSA,
    "ind": Language.INDONESIAN,
    #  'kea': Language.KABUVERDIANU,
    #  'lug': Language.GANDA,
    "nld": Language.DUTCH,
    "pes": Language.PERSIAN,
    #  'ssw': Language.SWATI,
    #  'tir': Language.TIGRINYA,
    "war": Language.WAR,
    "ary": Language.ARABIC,
    "cat": Language.CATALAN,
    "eus": Language.BASQUE,
    "que": Language.QUECHUA,
    "heb": Language.HEBREW,
    "isl": Language.ICELANDIC,
    #  'khk': Language.MONGOLIAN,
    #  'luo': Language.LUO,
    "nob": Language.NORWEGIAN,
    "plt": Language.MALAGASY,
    "slk": Language.SLOVAK,
    #  'sun': Language.SUNDANESE,
    #  'tsn': Language.TSWANA,
    #  'wol': Language.WOLOF
}
