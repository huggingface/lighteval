# pylint: disable=C0103  # noqa: CPY001
################################################################################
# Taken from https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py

from collections import namedtuple as _namedtuple


def normalize(tokens, case=False):
    """

    Lowercases and turns tokens into distinct words.

    """

    return [str(t).lower() if not case else str(t) for t in tokens]


################################################################################


class Fragments:
    Match = _namedtuple("Match", ("summary", "text", "length"))

    def __init__(self, summary, text, case=False):
        # self._tokens = tokenize

        if isinstance(summary, str):
            self.summary = summary.split()
        else:
            self.summary = summary
        if isinstance(text, str):
            self.text = text.split()
        else:
            self.text = text

        self._norm_summary = normalize(self.summary, case)
        self._norm_text = normalize(self.text, case)

        self._match(self._norm_summary, self._norm_text)

    def overlaps(self):
        """

        Return a list of Fragments.Match objects between summary and text.
        This is a list of named tuples of the form (summary, text, length):

            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment

        """

        return self._matches

    def strings(self, min_length=0, summary_base=True):
        """

        Return a list of explicit match strings between the summary and reference.
        Note that this will be in the same format as the strings are input. This is
        important to remember if tokenization is done manually. If tokenization is
        specified automatically on the raw strings, raw strings will automatically
        be returned rather than SpaCy tokenized sequences.

        Arguments:

            - min_length (int): filter out overlaps shorter than this (default = 0)
            - raw (bool): return raw input rather than stringified
                - (default = False if automatic tokenization, True otherwise)
            - summary_base (true): strings are based of summary text (default = True)

        Returns:

            - list of overlaps, where overlaps are strings or token sequences

        """

        # Compute the strings against the summary or the text?

        base = self.summary if summary_base else self.text

        # Generate strings, filtering out strings below the minimum length.

        strings = [base[i : i + length] for i, j, length in self.overlaps() if length > min_length]

        # By default, we just return the tokenization being used.
        # But if they user wants a raw string, then we convert.
        # Mostly, this will be used along with spacy.

        # if self._tokens and raw:

        #    for i, s in enumerate(strings):
        #        strings[i] = str(s)

        # Return the list of strings.

        return strings

    def coverage(self, summary_base=True):
        """
        Return the COVERAGE score of the summary and text.

        Arguments:

            - summary_base (bool): use summary as numerator (default = True)

        Returns:

            - decimal COVERAGE score within [0, 1]
        """

        numerator = sum(o.length for o in self.overlaps())

        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.text)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def density(self, summary_base=True):
        """

        Return the DENSITY score of summary and text.

        Arguments:

            - summary_base (bool): use summary as numerator (default = True)

        Returns:

            - decimal DENSITY score within [0, ...]

        """

        numerator = sum(o.length**2 for o in self.overlaps())

        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.text)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def compression(self, text_to_summary=True):
        """

        Return compression ratio between summary and text.

        Arguments:

            - text_to_summary (bool): compute text/summary ratio (default = True)

        Returns:

            - decimal compression score within [0, ...]

        """

        ratio = [len(self.text), len(self.summary)]

        try:
            if text_to_summary:
                return ratio[0] / ratio[1]
            else:
                return ratio[1] / ratio[0]

        except ZeroDivisionError:
            return 0

    def _match(self, a, b):
        """

        Raw procedure for matching summary in text, described in paper.

        """

        self._matches = []

        a_start = b_start = 0

        while a_start < len(a):
            best_match = None
            best_match_length = 0

            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start

                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1

                    length = a_end - a_start

                    if length > best_match_length:
                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length

                    b_start = b_end

                else:
                    b_start += 1

            b_start = 0

            if best_match:
                if best_match_length > 0:
                    self._matches.append(best_match)

                a_start += best_match_length

            else:
                a_start += 1


################################################################################
