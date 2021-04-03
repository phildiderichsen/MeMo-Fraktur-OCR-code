"""
align_ocr.py
Align original OCR string to corrected string.
Corrected string is tokenized, but apart from that stays as is.
"""

import statistics
import re

from Levenshtein import distance
from Levenshtein import ratio as lev_ratio
from difflib import SequenceMatcher
from itertools import chain
from myutils import tokenize


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


def recursive_token_align(corr: tuple, orig: tuple, sep='_', orig_tokens=tuple(), corr_tokens=tuple()):
    """
    Align orig to corr so that Levenshtein ratio (similarity) is iteratively maximised.
    Recurse so that corr is recursively split into first token and residual string.
    The resulting token count of orig will match corr.
    Use sep to consistently split and join tokens.
    """
    def iter_align(orig_toks, first_tok, rest):
        """
        Iteratively test alignment of all splits of orig_toks to first_tok and rest (e.g. 'x' 'xyy' <> 'xx' 'yy')
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
        return corr, (sep.join(orig),)
    # If any input tuple is empty, return with the other tuple joined
    elif not corr and not orig:
        return tuple(), tuple()
    elif not corr:
        return tuple(), (sep.join(orig), )
    elif not orig:
        return (sep.join(corr), ), tuple()
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
            return corr_tokens, tuple([tok if tok else '_' for tok in orig_tokens])
        else:
            # Recurse to align next token(s)
            return recursive_token_align(tuple(corr[1:]), tuple(split[1].split(sep)), sep=sep, orig_tokens=orig_tokens,
                                         corr_tokens=corr_tokens)


def get_matching_chunks(seqmatch: SequenceMatcher):
    """Get matching chunks from the sequence being aligned."""
    match_list = [block for block in seqmatch.get_matching_blocks() if block.size]
    correct = [seqmatch.a[match.a:match.a + match.size] for match in match_list]
    matching = [seqmatch.b[match.b:match.b + match.size] for match in match_list]
    a_start = [match.a for match in match_list]
    return [(start, c, m) for start, c, m in zip(a_start, correct, matching)]


def get_nonmatching_chunks(seqmatch: SequenceMatcher):
    """Get non-matching chunks from the sequence being aligned."""
    match_list = [block for block in seqmatch.get_matching_blocks() if block.size]
    nonmatch_start_idxs_a = [0] + [match.a + match.size for match in match_list]
    nonmatch_end_idxs_a = [match.a for match in match_list]
    nonmatch_idxs_a = zip(nonmatch_start_idxs_a, nonmatch_end_idxs_a)
    nonmatch_chunks_a = [seqmatch.a[idx[0]:idx[1]] for idx in nonmatch_idxs_a]

    nonmatch_start_idxs_b = [0] + [match.b + match.size for match in match_list]
    nonmatch_end_idxs_b = [match.b for match in match_list]
    nonmatch_idxs_b = zip(nonmatch_start_idxs_b, nonmatch_end_idxs_b)
    nonmatch_chunks_b = [seqmatch.b[idx[0]:idx[1]] for idx in nonmatch_idxs_b]
    nonmatch_chunks = [[start, a, b] for start, a, b in
                       zip(nonmatch_start_idxs_a, nonmatch_chunks_a, nonmatch_chunks_b)]

    # Repair chunks that do not have the same number of tokens.
    repaired_chunks = []
    for chunk in nonmatch_chunks:
        if len(chunk[1]) == len(chunk[2]):
            repaired_chunks.append(chunk)
        else:
            if not chunk[2]:
                chunk[2] = '_'
            rep_chunk = [chunk[0]] + list(recursive_token_align(chunk[1], chunk[2]))
            repaired_chunks.append(rep_chunk)
    return [tuple(chnk) for chnk in repaired_chunks]


def integrate_junk(merged: list):
    """Append tuples where correct part is empty, to the next tuple."""
    new_merged = []
    junk = tuple()
    for tup in merged:
        if tup[1]:
            if junk:
                orig_tup = tup[2]
                new_orig_first = '_'.join([junk[0], orig_tup[0]])
                tup = (tup[0], tup[1], (new_orig_first,) + orig_tup[1:])
                junk = tuple()
            new_merged.append(tup)
        else:
            junk = ('_'.join(junk + tup[2]),)
    if junk:
        tup = new_merged[-1]
        orig_tup = tup[2]
        new_orig_last = '_'.join([orig_tup[-1], junk[0]])
        new_merged[-1] = (tup[0], tup[1], orig_tup[:-1] + (new_orig_last,))
    return new_merged


def align_b_to_a(a: tuple, b: tuple):
    """Align b tuple to a tuple - paste tokens if necessary. Return aligned b tuple."""
    seqmatch = SequenceMatcher(None, a, b)
    matching = get_matching_chunks(seqmatch)
    nonmatching = get_nonmatching_chunks(seqmatch)
    merged = sorted(matching + nonmatching)
    merged = integrate_junk(merged)
    aligned_tokens = list(chain.from_iterable([tup[2] for tup in merged]))
    return tuple(aligned_tokens)


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
    # Get rid of gold standard hyphens
    corr = re.sub(r'\[[ -]+\]', '', corr)
    origtup = tuple(tokenize(orig))
    corrtup = tuple(tokenize(corr))
    return origtup, corrtup


def align_ocr(original, corrected):
    """Align two strings. Return alignment object"""
    origtup, corrtup = preprocess_input(original, corrected)
    aligned_corr, aligned_orig = recursive_token_align(corrtup, origtup)
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
