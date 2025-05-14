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
    ACEHNESE = "ace"
    AFRIKAANS = "afr"
    AKAN = "aka"
    ALBANIAN = "sqi"
    AMHARIC = "amh"
    ARABIC = "ara"
    ARMENIAN = "hye"
    ASSAMESE = "asm"
    ASTURIAN = "ast"
    AWADHI = "awa"
    AYACUCHO_QUECHUA = "quy"
    AZERBAIJANI = "aze"
    BALINESE = "ban"
    BAMBARA = "bam"
    BANJAR = "bjn"
    BASHKIR = "bak"
    BASQUE = "eus"
    BELARUSIAN = "bel"
    BEMBA = "bem"
    BENGALI = "ben"
    BHOJPURI = "bho"
    BIHARI = "bih"  # Deprecated
    BOSNIAN = "bos"
    BRETON = "bre"
    BUGINESE = "bug"
    BULGARIAN = "bul"
    BURMESE = "mya"
    CATALAN = "cat"
    CEBUANO = "ceb"
    CENTRAL_ATLAS_TAMAZIGHT = "tzm"
    CENTRAL_AYMARA = "ayr"
    CENTRAL_KANURI = "knc"
    CENTRAL_KURDISH = "ckb"
    CHHATTISGARHI = "hne"
    CHINESE = "zho"
    CHOKWE = "cjk"
    CRIMEAN_TATAR = "crh"
    CROATIAN = "hrv"
    CZECH = "ces"
    DANISH = "dan"
    DARI = "prs"
    DIVEHI = "div"
    DUTCH = "nld"
    DYULA = "dyu"
    DZONGKHA = "dzo"
    EASTERN_PANJABI = "pan"
    EASTERN_YIDDISH = "ydd"
    EGYPTIAN_ARABIC = "arz"
    ENGLISH = "eng"
    ESPERANTO = "epo"
    ESTONIAN = "est"
    EWE = "ewe"
    FAROESE = "fao"
    FIJIAN = "fij"
    FINNISH = "fin"
    FON = "fon"
    FRENCH = "fra"
    FRIULIAN = "fur"
    GALICIAN = "glg"
    GANDA = "lug"
    GEORGIAN = "kat"
    GERMAN = "deu"
    GREEK = "ell"
    GUARANI = "grn"
    GUJARATI = "guj"
    HAITIAN = "hti"
    HAITIAN_CREOLE = "hat"
    HALH_MONGOLIAN = "khk"
    HAUSA = "hau"
    HEBREW = "heb"
    HINDI = "hin"
    HUNGARIAN = "hun"
    ICELANDIC = "isl"
    IGBO = "ibo"
    ILOCANO = "ilo"
    INDONESIAN = "ind"
    IRISH = "gle"
    ITALIAN = "ita"
    JAPANESE = "jpn"
    JAVANESE = "jav"
    JINGPHO = "kac"
    KABIYE = "kbp"
    KABUVERDIANU = "kea"
    KABYLE = "kab"
    KAMBA = "kam"
    KANNADA = "kan"
    KASHMIRI = "kas"
    KAZAKH = "kaz"
    KHMER = "khm"
    KIKONGO = "kon"
    KIKUYU = "kik"
    KIMBUNDU = "kmb"
    KINYARWANDA = "kin"
    KIRGHIZ = "kir"
    KOREAN = "kor"
    KURDISH = "kur"
    KYRGYZ = "kir"
    LAO = "lao"
    LATGALIAN = "ltg"
    LATIN = "lat"
    LATVIAN = "lav"
    LIGURIAN = "lij"
    LIMBURGISH = "lim"
    LINGALA = "lin"
    LITHUANIAN = "lit"
    LOMBARD = "lmo"
    LUBA_KASAI = "lua"
    LUO = "luo"
    LUXEMBOURGISH = "ltz"
    MACEDONIAN = "mkd"
    MAGAHI = "mag"
    MAITHILI = "mai"
    MALAGASY = "mlg"
    MALAY = "msa"
    MALAYALAM = "mal"
    MALTESE = "mlt"
    MAORI = "mri"
    MARATHI = "mar"
    MEITEI = "mni"
    MESOPOTAMIAN_ARABIC = "acm"
    MINANGKABAU = "min"
    MIZO = "lus"
    MODERN_STANDARD_ARABIC = "arb"
    MOROCCAN_ARABIC = "ary"
    MOSSI = "mos"
    NAJDI_ARABIC = "ars"
    NEPALI = "nep"
    NIGERIAN_FULFULDE = "fuv"
    NORTHERN_KURDISH = "kmr"
    NORTHERN_SOTHO = "nso"
    NORTHERN_UZBEK = "uzn"
    NORTH_AZERBAIJANI = "azj"
    NORTH_LEVANTINE_ARABIC = "apc"
    NORWEGIAN = "nor"
    NORWEGIAN_BOKMAL = "nob"
    NORWEGIAN_NYNORSK = "nno"
    NUER = "nus"
    NYANJA = "nya"
    OCCITAN = "oci"
    ODIA = "ory"
    ORIYA = "ori"
    PANGASINAN = "pag"
    PAPIAMENTO = "pap"
    PASHTO = "pus"
    PERSIAN = "fas"
    PLATEAU_MALAGASY = "plt"
    POLISH = "pol"
    PORTUGUESE = "por"
    PUNJABI = "pan"
    QUECHUA = "que"
    ROMANIAN = "ron"
    RUNDI = "run"
    RUSSIAN = "rus"
    SAMOAN = "smo"
    SANGO = "sag"
    SANSKRIT = "san"
    SANTALI = "sat"
    SARDINIAN = "srd"
    SCOTTISH_GAELIC = "gla"
    SERBIAN = "srp"
    SERBOCROATIAN = "hbs"  # Deprecated
    SHAN = "shn"
    SHONA = "sna"
    SICILIAN = "scn"
    SILESIAN = "szl"
    SINDHI = "snd"
    SINHALA = "sin"
    SLOVAK = "slk"
    SLOVENIAN = "slv"
    SOMALI = "som"
    SORANI = "ckb"
    SOUTHERN_PASHTO = "pbt"
    SOUTHERN_SOTHO = "sot"
    SOUTHWESTERN_DINKA = "dik"
    SOUTH_AZERBAIJANI = "azb"
    SOUTH_LEVANTINE_ARABIC = "ajp"
    SPANISH = "spa"
    STANDARD_LATVIAN = "lvs"
    STANDARD_MALAY = "zsm"
    STANDARD_TIBETAN = "bod"
    SUNDANESE = "sun"
    SWAHILI = "swa"
    SWATI = "ssw"
    SWEDISH = "swe"
    TAGALOG = "tgl"
    TAJIK = "tgk"
    TAMASHEQ = "taq"
    TAMIL = "tam"
    TATAR = "tat"
    TAIZZI_ADENI_ARABIC = "acq"
    TELUGU = "tel"
    THAI = "tha"
    TIGRINYA = "tir"
    TOK_PISIN = "tpi"
    TOSK_ALBANIAN = "als"
    TSONGA = "tso"
    TSWANA = "tsn"
    TUMBUKA = "tum"
    TUNISIAN_ARABIC = "aeb"
    TURKISH = "tur"
    TURKMEN = "tuk"
    TWI = "twi"
    UDMURT = "udm"
    UKRAINIAN = "ukr"
    UMBUNDU = "umb"
    URDU = "urd"
    UYGHUR = "uig"
    UZBEK = "uzb"
    VENETIAN = "vec"
    VIETNAMESE = "vie"
    WAR = "war"
    WARAY = "war"
    WELSH = "cym"
    WESTERN_FRISIAN = "fry"
    WESTERN_PERSIAN = "pes"
    WEST_CENTRAL_OROMO = "gaz"
    WOLOF = "wol"
    XHOSA = "xho"
    YIDDISH = "yid"
    YORUBA = "yor"
    YUE_CHINESE = "yue"
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


def manage_duplicate_language_codes(langcode):
    if langcode == "npi":  # Nepali
        langcode = "nep"
    if langcode == "swh":  # Swahili
        langcode = "swa"
    return langcode


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
