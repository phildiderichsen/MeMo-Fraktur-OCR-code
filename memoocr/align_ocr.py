"""
align_ocr.py
Align original OCR string to corrected string.
Corrected string is tokenized, but apart from that stays as is.
"""

import statistics
import re
import myutils as util

from Levenshtein import distance
from Levenshtein import ratio as lev_ratio
from difflib import SequenceMatcher
from itertools import chain


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
            if lev == 0 or not corr:
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
        # TODO Håndter på en bedre måde når matches er [] ...
        self.avg_correct = round(statistics.mean(matches), 2) if matches else 0
        self.avg_lev_dist = round(statistics.mean(self.lev_dists), 2) if self.lev_dists else 0
        self.avg_cer = round(statistics.mean(self.cers), 2) if self.cers else 0
        self.avg_ratio = round(statistics.mean(self.ratios), 2) if self.ratios else 0

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
    # with open('/Users/phb514/Downloads/aligned_chunks.txt', 'w') as f:
    #     f.write('\n'.join([str(x) for x in aligned_chunks]))
    # Find and fix sequences with big length mismatches
    bad_seq_indexes = get_bad_seq_indexes(aligned_chunks, mismatch=8)
    if bad_seq_indexes:
        aligned_chunks = fix_bad_seqs(aligned_chunks, bad_seq_indexes)
    # with open('/Users/phb514/Downloads/fixd_aligned_chunks.txt', 'w') as f:
    #     f.write('\n'.join([str(x) for x in aligned_chunks]))
    # Now repair tuples that are not the same length, and repair empty tuples.
    aligned_chunks_repaired = repair_nonmatching(aligned_chunks)
    aligned_chunks_repaired = integrate_junk(aligned_chunks_repaired)
    aligned_tokens = tuple(chain.from_iterable([tup[1] for tup in aligned_chunks_repaired]))
    return aligned_tokens


def get_bad_seq_indexes(aligned_chunks: list, mismatch):
    """Get indexes of badly aligned subsequences to be fixed separately."""
    tuple_lengths = [(len(chnk[0]), len(chnk[1])) for chnk in aligned_chunks]
    len_diffs = [x[0] - x[1] for x in tuple_lengths]
    zl = list(zip(len_diffs, len_diffs[2:]))
    badseqs = []
    badseq = []
    for i, tp in enumerate(zl):
        # STARTING a bad sequence
        if not badseq and abs(tp[0]) > mismatch and abs(tp[1]) > mismatch:
            badseq.append(i)
        # ENDING a bad sequence
        elif badseq and abs(tp[0]) > mismatch and not abs(tp[1]) > mismatch:
            badseq.append(i + 1)
            badseqs.append(tuple(badseq))
            badseq = []
    return badseqs


def fix_bad_seqs(aligned_chunks, bad_index_pairs):
    """Replace bad subsequences with freshly aligned ones."""

    def fix_bad_seq(bad_chunks):
        """Fix a single bad chunk sequence by making it one chunk and aligning again."""
        a, b = tuple([tuple(util.flatten([x for x, _ in bad_chunks])), tuple(util.flatten([y for _, y in bad_chunks]))])
        seqmatch = SequenceMatcher(None, a, b)
        match_idxs = get_align_indexes(seqmatch)
        # if ..: eliminate empty tuples: ((), ()).
        new_aligned = [(a[mi.ai:mi.aj], b[mi.bi:mi.bj]) for mi in match_idxs if a[mi.ai:mi.aj] or b[mi.bi:mi.bj]]
        return new_aligned

    flat_indexes = sorted(list(set(util.flatten(bad_index_pairs))))
    start_sublist = aligned_chunks[:flat_indexes[0]]
    end_sublist = aligned_chunks[flat_indexes[-1]:]

    new_index_pairs = list(zip(flat_indexes, flat_indexes[1:]))
    enumerated_sublists = enumerate([aligned_chunks[i:j] for i, j in new_index_pairs])
    recombined = [fix_bad_seq(sublist) if (i % 2 == 0) else sublist for i, sublist in enumerated_sublists]

    return start_sublist + util.flatten(recombined) + end_sublist


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

    align_indexes = []
    matchblocks = seqmatch.get_matching_blocks()
    if len(matchblocks) == 1:
        mb = matchblocks[0]
        return [MatchIndexes(0, mb.a, 0, mb.b, bool(mb.size))]
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


def recursive_token_align(corr: tuple, orig: tuple, sep='☐', orig_tokens=tuple(), corr_tokens=tuple()):
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
            orig += (sep,) * (2 - len(orig))
        split = iter_align(orig, corr[0], corr[1:])
        orig_tokens += (split[0],)
        corr_tokens += (corr[0],)

        if len(corr) == 2:
            # No more binary splits to do: Add the last token and return
            orig_tokens += (split[1],)
            corr_tokens += (sep.join(corr[1:]),)  # Join should be redundant as this should be the last token.
            return corr_tokens, tuple([tok if tok else sep for tok in orig_tokens])
        else:
            # Recurse to align next token(s)
            return recursive_token_align(tuple(corr[1:]), tuple(split[1].split(sep)), sep=sep, orig_tokens=orig_tokens,
                                         corr_tokens=corr_tokens)


def repair_nonmatching(aligned_chunks, sep='☐'):
    """Repair tuples that are not the same length, with recursive token align."""
    aligned_chunks_repaired = []
    for chunk in aligned_chunks:
        if len(chunk[0]) == len(chunk[1]):
            aligned_chunks_repaired.append(chunk)
        else:
            chunklist = list(chunk)
            if not chunklist[1]:
                chunklist[1] = (sep, )
            try:
                rep_chunk = tuple(list(recursive_token_align(chunklist[0], chunklist[1])))
            except RecursionError:
                # TODO Too big mismatching chunks may or may not still be an issue.
                print('chunklist[0]:')
                print(chunklist[0])
                print('chunklist[1]:')
                print(chunklist[1])
                print()

            aligned_chunks_repaired.append(rep_chunk)
    return aligned_chunks_repaired


def integrate_junk(merged: list, sep='◇'):
    """Append tuples where correct part is empty, to the next tuple."""
    new_merged = []
    junk = tuple()
    for tup in merged:
        if tup[0]:
            if junk:
                orig_tup = tup[1]
                new_orig_first = sep.join([junk[0], orig_tup[0]])
                tup = (tup[0], (new_orig_first,) + orig_tup[1:])
                junk = tuple()
            new_merged.append(tup)
        else:
            junk = (sep.join(junk + tup[1]),)
    if junk:
        tup = new_merged[-1]
        orig_tup = tup[1]
        new_orig_last = sep.join([orig_tup[-1], junk[0]])
        new_merged[-1] = (tup[0], orig_tup[:-1] + (new_orig_last,))
    return new_merged


def align_conll_tuples(vrt_tups: list, conll_tups: list):
    """Align a list of CONLL token tuples to a list of VRT file tuples; merge the tuples."""

    def merge_tokentups(_vrt_tups, _conll_tups):
        """Merge VRT token tuples and modified CONLL tuples, assuming the tokens match."""
        # ct[0], ct[2], ct[4]: Token # in sentence, lemma, PoS
        mod_conll_tups = [(ct[0], ct[2], ct[4]) for ct in _conll_tups]
        return [tup1 + tup2 for tup1, tup2 in zip(_vrt_tups, mod_conll_tups)]

    vrt_tokens = [tup[0] for tup in vrt_tups]
    conll_tokens = [tup[1] for tup in conll_tups]
    # If the two token lists are identical: Merge and return
    if vrt_tokens == conll_tokens:
        return merge_tokentups(vrt_tups, conll_tups)
    else:
        aligned_vrt_tups, aligned_conll_tups = align_nonmatching_conll_tuples(vrt_tups, conll_tups)
        return merge_tokentups(aligned_vrt_tups, aligned_conll_tups)


def align_nonmatching_conll_tuples(vrt_tups, conll_tups):
    """
    Align a list of CONLL nonmatching token tuples to a list of VRT file tuples; merge the tuples.
    Note: Token number in sentence will no longer be consecutive, but still monotonically increasing.
    - Remove any tokens in CONLL that do not exist in VRT. TODO: Token enumeration will no longer be consecutive ..
    - Tokens in VRT not in CONLL: Insert dummy CONLL token ('_', '_', '_'). TODO: Fix _ instead of token number ..
    - Different token in CONLL than in VRT: Keep token number, replace CONLL annotations with '_'.
    """
    vrt_tokens = [tup[0] for tup in vrt_tups]
    conll_tokens = [tup[1] for tup in conll_tups]
    align_idxs = get_align_indexes(SequenceMatcher(None, vrt_tokens, conll_tokens))
    aligned_vrt_tups = []
    aligned_conll_tups = []
    for ali in align_idxs:
        if ali.match:
            [aligned_vrt_tups.append(tup) for tup in vrt_tups[ali.ai:ali.aj]]
            [aligned_conll_tups.append(tup) for tup in conll_tups[ali.bi:ali.bj]]
        else:
            # Note: We now know that no tokens match; just replace any nonmatching annotations with dummies.
            if not vrt_tups[ali.ai:ali.aj]:
                # print('Remove any CONLL tokentuples not in VRT.')
                pass
            elif len(vrt_tups[ali.ai:ali.aj]) > len(conll_tups[ali.bi:ali.bj]):
                # print('Replace missing or nonmatching CONLL token(s) with dummies')
                conll_dummies = [('_', ) * 14] * len(vrt_tups[ali.ai:ali.aj])
                [aligned_vrt_tups.append(tup) for tup in vrt_tups[ali.ai:ali.aj]]
                [aligned_conll_tups.append(tup) for tup in conll_dummies]
            elif len(vrt_tups[ali.ai:ali.aj]) == len(conll_tups[ali.bi:ali.bj]):
                # print('VRT and CONLL tokens do not match. Keep only token number.')
                quasi_dummies = [(tup[0], ) + ('_', ) * 13 for tup in conll_tups[ali.bi:ali.bj]]
                [aligned_vrt_tups.append(tup) for tup in vrt_tups[ali.ai:ali.aj]]
                [aligned_conll_tups.append(tup) for tup in quasi_dummies]
    return aligned_vrt_tups, aligned_conll_tups


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
    origtup = tuple(util.tokenize(orig))
    corrtup = tuple(util.tokenize(corr))
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
    # print(align_b_to_a(corrtup, origtup))
    vrt_tuples = [('Uge', '2', '1', '22', 'Uge', 'NA', '1.0', 'NA', 'match', 'NA'), ('efter', '3', '1', '22', 'efter', 'NA', '1.0', 'NA', 'match', 'NA'), ('dette', '4', '1', '22', 'dette', 'NA', '1.0', 'NA', 'match', 'NA'), ('Besøg', '7', '1', '22', 'Besog', '1', '0.8', '0.2', 'lev_1', 'ø=o'), ('var', '8', '1', '22', 'var', 'NA', '1.0', 'NA', 'match', 'NA'), ('han', '9', '1', '22', 'han', 'NA', '1.0', 'NA', 'match', 'NA'), ('forlovet', '10', '1', '22', 'forlovet', 'NA', '1.0', 'NA', 'match', 'NA'), ('og', '1', '2', '22', 'og', 'NA', '1.0', 'NA', 'match', 'NA'), ('allerede', '2', '2', '22', 'allerede', 'NA', '1.0', 'NA', 'match', 'NA'), ('en', '3', '2', '22', 'en', 'NA', '1.0', 'NA', 'match', 'NA'), ('Maaned', '4', '2', '22', 'Maaned', 'NA', '1.0', 'NA', 'match', 'NA'), ('efter', '5', '2', '22', 'efter', 'NA', '1.0', 'NA', 'match', 'NA'), ('gift', '6', '2', '22', 'gift', 'NA', '1.0', 'NA', 'match', 'NA'), ('med', '7', '2', '22', 'med', 'NA', '1.0', 'NA', 'match', 'NA'), ('den', '8', '2', '22', 'den', 'NA', '1.0', 'NA', 'match', 'NA'), ('skjønne', '9', '2', '22', 'stjonne', '2', '0.71', '0.29', 'lev_2', 'k=t, ø=o'), ('kun', '1', '3', '22', 'kun', 'NA', '1.0', 'NA', 'match', 'NA'), ('syttenaarige', '2', '3', '22', 'syttenaarige', 'NA', '1.0', 'NA', 'match', 'NA'), ('Ida', '3', '3', '22', 'Ida', 'NA', '1.0', 'NA', 'match', 'NA'), ('Krabbe', '4', '3', '22', 'Krabbe', 'NA', '1.0', 'NA', 'match', 'NA'), ('.', '5', '3', '22', '.', 'NA', '1.0', 'NA', 'match', 'NA'), ('Søviggaard', '1', '4', '22', 'Soviggaard', '1', '0.9', '0.1', 'lev_1', 'ø=o'), ('*', '2', '4', '22', '“', '1', 'NA', '1.0', 'lev_1', '*=“'), (')', '3', '4', '22', ')', 'NA', '1.0', 'NA', 'match', 'NA'), ('har', '4', '4', '22', 'har', 'NA', '1.0', 'NA', 'match', 'NA'), ('en', '5', '4', '22', 'en', 'NA', '1.0', 'NA', 'match', 'NA')]
    conll_tuples = [('1', 'en', 'en', 'en', 'pron', 'pron', '_', '_', '_', '_', '_', '_', '_', '_'), ('2', 'Uge', 'Uge', 'Uge', 'sb', 'sb', '_', '_', '_', '_', '_', '_', '_', '_'), ('3', 'efter', 'efter', 'efter', 'præp', 'præp', '_', '_', '_', '_', '_', '_', '_', '_'), ('5', 'sit', 'sin', 'sin', 'pron', 'pron', '_', '_', '_', '_', '_', '_', '_', '_'), ('6', 'første', 'først', 'først', 'adj', 'adj', '_', '_', '_', '_', '_', '_', '_', '_'), ('8', 'var', 'vare', 'vare', 'v', 'v', '_', '_', '_', '_', '_', '_', '_', '_'), ('9', 'han', 'han', 'han', 'pron', 'pron', '_', '_', '_', '_', '_', '_', '_', '_'), ('10', 'forlovet', 'forlove', 'forlove', 'præt', 'præt', '_', '_', '_', '_', '_', '_', '_', '_'), ('11', 'og', 'og', 'og', 'konj', 'konj', '_', '_', '_', '_', '_', '_', '_', '_'), ('12', 'allerede', 'allerede', 'allerede', 'adv', 'adv', '_', '_', '_', '_', '_', '_', '_', '_'), ('13', 'en', 'en', 'en', 'pron', 'pron', '_', '_', '_', '_', '_', '_', '_', '_'), ('14', 'Maaned', 'Maaned', 'Maaned', 'sb', 'sb', '_', '_', '_', '_', '_', '_', '_', '_'), ('15', 'efter', 'efter', 'efter', 'adv', 'adv', '_', '_', '_', '_', '_', '_', '_', '_'), ('16', 'gift', 'gift', 'gift', 'adj', 'adj', '_', '_', '_', '_', '_', '_', '_', '_'), ('17', 'med', 'med', 'med', 'præp', 'præp', '_', '_', '_', '_', '_', '_', '_', '_'), ('18', 'den', 'den', 'den', 'pron', 'pron', '_', '_', '_', '_', '_', '_', '_', '_'), ('20', ',', ',', ',', 'ZD', 'ZD', '_', '_', '_', '_', '_', '_', '_', '_'), ('22', 'syttenaarige', 'syttenaarig', 'syttenaarig', 'adj', 'adj', '_', '_', '_', '_', '_', '_', '_', '_'), ('25', '.', '.', '.', 'ZP', 'ZP', '_', '_', '_', '_', '_', '_', '_', '_'), ('1', 'Søviggaard', 'Søviggaard', 'Søviggaard', 'sb', 'sb', '_', '_', '_', '_', '_', '_', '_', '_'), ('2', '*', '*', '*', 'sb', 'sb', '_', '_', '_', '_', '_', '_', '_', '_'), ('3', ')', ')', ')', 'ZD', 'ZD', '_', '_', '_', '_', '_', '_', '_', '_'), ('4', 'har', 'have', 'have', 'v', 'v', '_', '_', '_', '_', '_', '_', '_', '_'), ('5', 'en', 'en', 'en', 'pron', 'pron', '_', '_', '_', '_', '_', '_', '_', '_')]

    aligned_tokentuples = align_conll_tuples(vrt_tuples, conll_tuples)
    for at in aligned_tokentuples:
        print(at)


if __name__ == '__main__':
    main()
