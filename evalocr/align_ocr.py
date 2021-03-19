"""
align_ocr.py
Align original OCR string to corrected string.
Corrected string is tokenized, but apart from that stays as is.
"""

import statistics
import re

from Levenshtein import distance
from Levenshtein import ratio as lev_ratio

from nltk import word_tokenize


# Note: In order for nltk to tokenize with language='danish', the Punkt Tokenizer Models have to be installed.
# Create a nltk_data dir, e.g. /Users/phb514/nltk_data or /usr/local/share/nltk_data
# Create a subdir /Users/phb514/nltk_data/tokenizers
# Download Punkt Tokenizer Models from http://www.nltk.org/nltk_data
# Extract the downloaded punkt.zip, and place the "punkt" dir in /Users/phb514/nltk_data/tokenizers.


class Alignment(object):
    """
    Object with list of aligned tokens (original form + corrected form) and some alignment stats etc.
    CER is character error rate: Levenshtein distance by character count of correct word. 0 = perfect to 100 = bad.
    Ratio is a Levenshtein similarity measure (0 = no match to 1 = string equality).
    """

    def __init__(self, aligned_tokens: list, aligned_orig: tuple, correct: tuple, matchtypes: tuple, matches: tuple):

        def get_cer(lev, corr):
            """Calculate character error rate."""
            if lev == 0:
                return float(0)
            else:
                return round(lev / len(corr) * 100, 2)

        self.aligned_tokens = aligned_tokens
        self.aligned_orig = aligned_orig
        self.correct = correct
        self.matchtypes = matchtypes
        self.matches = matches
        self.lev_dists = [distance(o, c) for o, c in zip(aligned_orig, correct)]
        self.cers = [get_cer(lev, corr) for lev, corr in zip(self.lev_dists, correct)]
        self.ratios = [round(lev_ratio(tok.orig, tok.corr), 2) for tok in aligned_tokens]
        self.avg_correct = round(sum(self.matches) / len(matches), 2)
        self.avg_lev_dist = round(statistics.mean(self.lev_dists), 2)
        self.avg_cer = round(statistics.mean(self.cers), 2)
        self.avg_ratio = round(statistics.mean(self.ratios), 2)

    def __repr__(self):
        attr_reprs = [f'\n{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'Alignment({", ".join(attr_reprs)})'


class AlignToken(object):
    """Single token alignment with aligned original token, correct token, etc."""

    def __init__(self, orig: str, corr: str, matchtype: str):
        self.orig = orig
        self.corr = corr
        self.matchtype = matchtype

    def __repr__(self):
        attr_reprs = [f"{k}: '{v}'" for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'AlignToken({", ".join(attr_reprs)})'


def recursive_token_align(orig: tuple, corr: tuple, sep='_', orig_tokens=tuple(), corr_tokens=tuple()):
    """
    Align orig to corr so that Levenshtein ratio (similarity) is iteratively maximised.
    Recurse so that corr is recursively split into first token and residual string.
    The resulting token count of orig will match corr.
    Use sep to consistently split and join tokens.
    """

    def iter_align(orig_toks, first_tok, rest):
        """
        Iteratively test alignment of all splits of _orig to first_tok and rest (e.g. 'x' 'xyy' <> 'xx' 'yy')
        Return overall best split.
        """
        lev_ratio_sum = 0
        orig_split = [orig_toks[0], sep.join(orig_toks[1:])]  # Default split: first element of orig + rest.
        for i in range(len(orig_toks) + 1):
            part1, part2 = orig_toks[:i], orig_toks[i:]
            pt1_ratio = lev_ratio(''.join(part1), first_tok)
            pt2_ratio = lev_ratio(''.join(part2), ''.join(rest))
            ratio_sum = pt1_ratio + pt2_ratio
            # If either part1 or part2 is a perfect match, return without iterating further
            if pt1_ratio == 1 or pt2_ratio == 1:
                return [sep.join(part1), sep.join(part2)]
            if ratio_sum > lev_ratio_sum:
                lev_ratio_sum = ratio_sum
                orig_split = [sep.join(part1), sep.join(part2)]
        return orig_split

    # If there is only one correct token, just return it with everything in the original tuple joined.
    if len(corr) == 1:
        return (sep.join(orig),), corr
    else:
        # Make sure there are at least two elements in orig
        if len(orig) < 2:
            orig += ('_',) * (2 - len(orig))
        split = iter_align(orig, corr[0], corr[1:])
        orig_tokens += (split[0],)
        corr_tokens += (corr[0],)

        if len(corr) == 2:
            # No more binary splits to do: Add the last token and return
            orig_tokens += (split[1],)
            corr_tokens += (sep.join(corr[1:]),)  # Join should be redundant as this should be the last token.
            return tuple([tok if tok else '_' for tok in orig_tokens]), corr_tokens
        else:
            # Recurse to align next token(s)
            return recursive_token_align(tuple(split[1].split(sep)), tuple(corr[1:]),
                                         sep=sep, orig_tokens=orig_tokens, corr_tokens=corr_tokens)


def make_alignment_obj(orig: tuple, corr: tuple):
    """Make Alignment object from tuple of original tokens and tuple of correct tokens."""

    def determine_matches(orig_tokens, corr_tokens):
        """Determine whether (aligned) original token and correct token match."""
        matchlist = []
        for orig_token, corr_token in zip(orig_tokens, corr_tokens):
            matchval = bool(orig_token == corr_token)
            matchlist.append(matchval)
        return tuple(matchlist)

    def determine_types(orig_tokens, corr_tokens):
        """Determine the type of each token alignment."""
        typelist = []
        for orig_token, corr_token in zip(orig_tokens, corr_tokens):
            if '_' in orig_token and len(orig_token) > 1 and orig_token.replace('_', '') == corr_token:
                typeval = 'same_chars'
            elif orig_token == corr_token:
                typeval = 'match'
            elif '[-]' not in corr_token and '_' not in orig_token:
                typeval = f'lev_{str(distance(orig_token, corr_token))}'
            elif '[-]' not in corr_token and '_' in orig_token:
                typeval = f'split_lev_{str(distance(orig_token, corr_token))}'
            # TODO: Add more/better categories. Split into levenshtein with and without underscores ...
            else:
                typeval = 'blaha'
            typelist.append(typeval)
        return tuple(typelist)

    matchtypes = determine_types(orig, corr)
    matches = determine_matches(orig, corr)
    align_toks = [AlignToken(*args) for args in zip(orig, corr, matchtypes)]
    return Alignment(aligned_tokens=align_toks, aligned_orig=orig, correct=corr,
                     matchtypes=matchtypes, matches=matches)


def preprocess_input(orig: str, corr: str):
    """Preprocess input strings for alignment. (Clean a little, tokenize)."""
    # Pad punctuation with whitespace
    orig = re.sub(r'([.,:;„"»«\'!?()])', r' \1 ', orig)
    corr = re.sub(r'([.,:;„"»«\'!?()])', r' \1 ', corr)
    # Get rid of gold standard hyphens
    corr = re.sub(r'\[[ -]+\]', '', corr)
    origtup = tuple(word_tokenize(orig, language='danish'))
    corrtup = tuple(word_tokenize(corr, language='danish'))
    return origtup, corrtup


def align_ocr(original, corrected):
    """Align two strings. Return alignment object"""
    origtup, corrtup = preprocess_input(original, corrected)
    aligned_orig, aligned_corr = recursive_token_align(origtup, corrtup)
    alignment = make_alignment_obj(aligned_orig, aligned_corr)
    # Add original and correct string
    alignment.orig_str = original
    alignment.corr_str = corrected
    return alignment


def main():
    orig = '„Hr. E ta tsra a d Helmer, tlieoloZios'
    corr = '„Hr. Etatsraad Helmer, Candidatus theologiæ'
    orig = """hendes Hensigt at syre Knuds Lig med sig t il L)t."""
    corr = """hendes Hensigt at føre Knuds Lig med sig til St."""
    orig = 'B a rn , jeg elsker over A lt. "'
    corr = 'Barn, jeg elsker over Alt."'
    alignment = align_ocr(orig, corr)
    print(alignment)


if __name__ == '__main__':
    main()
