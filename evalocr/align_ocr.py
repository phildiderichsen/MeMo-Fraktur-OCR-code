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


def align_b_to_a(a: tuple, b: tuple):
    """Align b tuple to a tuple - paste tokens if necessary. Return aligned b tuple."""
    seqmatch = SequenceMatcher(None, a, b)
    match_idxs = get_align_indexes(seqmatch)
    aligned_chunks = [(a[mi.ai:mi.aj], b[mi.bi:mi.bj]) for mi in match_idxs]
    # Now repair tuples that are not the same length, and repair empty tuples.
    aligned_chunks_repaired = repair_nonmatching(aligned_chunks)
    aligned_chunks_repaired = integrate_junk(aligned_chunks_repaired)
    aligned_tokens = tuple(chain.from_iterable([tup[1] for tup in aligned_chunks_repaired]))
    return aligned_tokens


def get_align_indexes(seqmatch: SequenceMatcher):
    """Get indexes for matching and nonmatching parts of two token tuples (from SequenceMatcher)."""

    class MatchIndexes(object):
        """Start/end indexes for a matching block of sequences a and b, with a match indicator."""

        def __init__(self, a_i: int, a_j: int, b_i: int, b_j: int, match: bool):
            """[ab]i: Start index, [ab]j: End index, match: Is this a matching tuple or not?"""
            self.ai, self.aj = a_i, a_j
            self.bi, self.bj = b_i, b_j
            self.match = match

        def __repr__(self):
            attr_reprs = [f'{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('__')]
            return f'MatchIndexes({", ".join(attr_reprs)})'

    matchblocks = seqmatch.get_matching_blocks()
    align_indexes = []
    for mpair in zip(matchblocks, matchblocks[1:]):
        ai = mpair[0].a           # Indexes from the a side
        aj = ai + mpair[0].size
        ak = mpair[1].a
        bi = mpair[0].b           # Indexes from the b side
        bj = bi + mpair[0].size
        bk = mpair[1].b
        align_indexes.append(MatchIndexes(ai, aj, bi, bj, match=True))
        align_indexes.append(MatchIndexes(aj, ak, bj, bk, match=False))
    # Fill in any missing mismatches at the beginning
    if align_indexes[0].ai > 0 or align_indexes[0].bi > 0:
        new_aj, new_bj = align_indexes[0].ai, align_indexes[0].bi
        align_indexes = [MatchIndexes(0, new_aj, 0, new_bj, match=False)] + align_indexes
    return align_indexes


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


def repair_nonmatching(aligned_chunks):
    """Repair tuples that are not the same length, with recursive token align."""
    aligned_chunks_repaired = []
    for chunk in aligned_chunks:
        if len(chunk[0]) == len(chunk[1]):
            aligned_chunks_repaired.append(chunk)
        else:
            chunklist = list(chunk)
            if not chunklist[1]:
                chunklist[1] = ('_', )
            rep_chunk = tuple(list(recursive_token_align(chunklist[0], chunklist[1])))
            aligned_chunks_repaired.append(rep_chunk)
    return aligned_chunks_repaired


def integrate_junk(merged: list):
    """Append tuples where correct part is empty, to the next tuple."""
    new_merged = []
    junk = tuple()
    for tup in merged:
        if tup[0]:
            if junk:
                orig_tup = tup[1]
                new_orig_first = '_'.join([junk[0], orig_tup[0]])
                tup = (tup[0], (new_orig_first,) + orig_tup[1:])
                junk = tuple()
            new_merged.append(tup)
        else:
            junk = ('_'.join(junk + tup[1]),)
    if junk:
        tup = new_merged[-1]
        orig_tup = tup[1]
        new_orig_last = '_'.join([orig_tup[-1], junk[0]])
        new_merged[-1] = (tup[0], orig_tup[:-1] + (new_orig_last,))
    return new_merged


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
    orig = 'Det B a rn , jeg elsker over A lt. "'
    corr = 'Og Det Barn, jeg elsker over Alt."'
    origtup = tuple(orig.split())
    corrtup = tuple(corr.split())
    corrtup = ('en', 'Uge', 'efter', 'dette', 'sit', 'første', 'Besøg', 'var', 'han', 'forlovet', 'og', 'allerede', 'en', 'Maaned', 'efter', 'gift', 'med', 'den', 'skjønne', ',', 'kun', 'syttenaarige', 'Ida', 'Krabbe', '.', 'Søviggaard', '*', ')', 'har', 'en', 'overmaade', 'deilig', 'Beliggenhed', 'i', 'den', 'sydostlige', 'Del', 'af', 'Vendsyssel', ',', 'ved', 'Kattegattet', '.', 'Skjønne', ',', 'med', 'Lyng', 'og', 'Skov', 'bevoxede', 'Bakker', 'omgive', 'Gaarden', 'mod', 'Nord', ',', 'Syd', 'og', 'Vest', ';', 'østenfor', 'den', 'strække', 'sig', 'derimod', 'græsrige', 'Enge', 'helt', 'ud', 'til', 'Havet', '.', 'Forbi', 'Gaarden', 'og', 'gjennem', 'disse', 'Enge', 'ud', 'imod', 'Kattegattet', 'strømmer', 'en', 'Aa', ',', 'hvis', 'Bredder', ',', 'paa', 'Herresædets', 'Enemærker', ',', 'ere', 'tæt', 'bevoxede', 'med', 'høje', 'Træer', ',', 'hvilke', 'om', 'Somren', 'paa', 'mange', 'Steder', 'danne', 'et', 'saa', 'tæt', 'Løvtag', 'over', 'Aaen', ',', 'at', 'Solens', 'Straaler', 'ikke', 'kunne', 'trænge', 'derigjennem', '.', 'Naar', 'man', 'i', 'nogen', 'Tid', 'har', 'ladet', 'sig', 'glide', 'i', 'en', 'Baad', 'ned', 'med', 'Strømmen', 'mellem', 'disse', 'Træer', ',', 'der', 'staae', 'som', 'en', 'Mur', 'til', 'begge', 'Sider', ',', 'i', 'den', 'høitidelige', 'Dunkelhed', ',', 'som', 'dannes', 'af', 'de', 'mørkegrønne', 'Hvælvinger', ',', 'og', 'man', 'pludselig', 'kommer', 'ud', 'i', 'det', 'fulde', 'Dagslys', 'og', 'seer', 'det', 'blaa', ',', 'solbeskinnede', 'Hav', ',', 'da', 'gribes', 'man', 'af', 'Undren', 'og', 'Glæde', ',', 'og', 'Hjertet', 'føler', 'sig', 'mildt', 'bevæget', 'ved', 'dette', 'yndige', 'og', 'rige', 'Naturbillede', '.', 'Paa', 'dette', 'Herresæde', 'boede', 'nu', 'Eiler', 'Grubbe', 'og', 'Ida', 'Krabbe', 'som', 'Ægtefolk', '.', '—', 'Da', 'Grubbe', 'strax', 'efter', '*', ')', 'Findes', 'ikke', 'under', 'dette', 'Navn', '.')
    origtup = ('10', 'en', 'Uge', 'efter', 'dette', 'sit', 'forste', 'Besog', 'var', 'han', 'forlovet', 'og', 'allerede', 'en', 'Maaned', 'efter', 'gift', 'med', 'den', 'stjonne', ',', 'kun', 'syttenaarige', 'Ida', 'Krabbe', '.', 'Soviggaard', '“', ')', 'har', 'en', 'overmaade', 'deilig', 'Be—', 'liggenhed', 'i', 'den', 'sydostlige', 'Del', 'af', 'Vendsyssel', ',', 'ved', 'Kattegattet', '.', 'Skjonne', ',', 'med', 'Lyng', 'og', 'Skovp', 'bevoxede', 'Bakker', 'omgive', 'Gaarden', 'mod', 'Nord', ',', 'Syd', 'og', 'Vest', ';', 'ostenfor', 'den', 'strække', 'sig', 'derimod', 'grasrige', 'Enge', 'helt', 'ud', 'til', 'Havet', '.', 'Forbi', 'Gaarden', 'og', 'gijiennem', 'disse', 'Enge', 'ud', 'imod', 'Kattegattet', 'strommer', 'en', 'Aa', ',', 'hyvis', 'Bredder', ',', 'paa', 'Herresædets', 'Enemarker', ',', 'ere', 'tat', 'be—', 'voxede', 'med', 'hoje', 'Traer', ',', 'hvilke', 'om', 'Somren', 'paa', 'mange', 'Steder', 'danne', 'et', 'saa', 'tæœæt', 'Levtag', 'over', 'Aaen', ',', 'at', 'Solens', 'Straaler', 'ikke', 'kunne', 'trœnge', 'derigjennem', '.', 'Naar', 'man', 'i', 'nogen', 'Tid', 'har', 'ladet', 'sig', 'glide', 'i', 'en', 'Baad', 'ned', 'med', 'Strommen', 'mellem', 'disse', 'Traer', ',', 'der', 'stage', 'som', 'en', 'Mur', 'til', 'begge', 'Sider', ',', 'i', 'den', 'höitidelige', 'Dunkelhed', ',', 'som', 'dannes', 'af', 'de', 'merkegronne', 'Hvalpvinger', ',', 'og', 'man', 'pludselig', 'kommer', 'ud', 'i', 'det', 'fulde', 'Dagslys', 'og', 'seer', 'det', 'blaa', ',', 'solbestinnede', 'Hav', ',', 'da', 'gribes', 'man', 'af', 'Undren', 'og', 'Glaæde', ',', 'og', 'Hijertet', 'foler', 'sig', 'mildt', 'bevaget', 'ved', 'dette', 'yndige', 'og', 'rige', 'Naturbillede', '.', 'Paa', 'dette', 'Herresæde', 'boede', 'nu', 'Eiler', 'Grubbe', 'og', 'Ida', 'Krabbe', 'som', 'Xgtefolk', '.', '—', 'Da', 'Grubbe', 'strax', 'efter', '*', ')', 'Findes', 'ilke', 'under', 'dette', 'Navn', '.')
    alignment = align_ocr(orig, corr)
    # print(alignment)
    print(align_b_to_a(corrtup, origtup))


if __name__ == '__main__':
    main()
