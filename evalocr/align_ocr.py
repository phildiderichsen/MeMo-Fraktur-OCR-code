import statistics
import re

from difflib import SequenceMatcher
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
    Object with aligned original string as tuple and correct string as tuple.
    CER is character error rate: Levenshtein distance by character count of correct word. 0 = perfect to 100 = bad.
    Ratio is a correctness measure from SequenceMathcer - 0 = bad to 1 = perfect.
    """

    def __init__(self, aligned_tokens: list, aligned_orig: tuple, correct: tuple, types: tuple, matchtypes: tuple, matches: tuple):
        def get_dist(x, y, match):
            return 0 if match else distance(x, y)

        def get_ratio(x, y, match):
            return 1 if match else round(SequenceMatcher(None, x, y).ratio(), 2)

        def get_cer(lev, corr):
            if lev == 0:
                return float(0)
            else:
                # TODO: Handle hyphen more systematically ...
                return lev / len(re.sub(r'\[-\].*', '', corr)) if '[-]' in corr and corr != '[-]' else lev / len(corr)

        self.aligned_tokens = aligned_tokens
        self.aligned_orig = aligned_orig
        self.correct = correct
        self.types = types
        self.matchtypes = matchtypes
        self.matches = matches
        self.lev_dists = [get_dist(o, c, m) for o, c, m in zip(aligned_orig, correct, matches)]
        self.cers = [round(get_cer(lev, corr) * 100, 2) for lev, corr in zip(self.lev_dists, correct)]
        self.ratios = [get_ratio(o, c, m) for o, c, m in zip(aligned_orig, correct, matches)]
        self.avg_correct = round(sum(self.matches) / len(matches), 2)
        self.avg_lev_dist = round(statistics.mean(self.lev_dists), 2)
        self.avg_cer = round(statistics.mean(self.cers), 2)
        self.avg_ratio = round(statistics.mean(self.ratios), 2)
        self.corr_tokens = tuple()  # Will be added in align_ocr() TODO: Do this less ad hoc ..

    def __repr__(self):
        attr_reprs = [f'\n{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'Alignment({", ".join(attr_reprs)})'


class SeqChunk(object):
    """Intermediate chunk of sequence (either orig or corr) - either matching or not."""

    def __init__(self, chunk: tuple, start: int, match: bool):
        self.chunk = chunk
        self.start = start
        self.match = match

    def __repr__(self):
        attr_reprs = [f'{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'SeqChunk({", ".join(attr_reprs)})'


class AlignChunk(object):
    """Intermediate chunk of alignment (with orig and corr) - either matching or not."""

    def __init__(self, orig_chunk: tuple, corr_chunk: tuple):
        self.orig_chunk = orig_chunk
        self.corr_chunk = corr_chunk
        self.type = 'nil'

    def __repr__(self):
        attr_reprs = [f'{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'AlignChunk({", ".join(attr_reprs)})'


class AlignToken(object):
    """Single token alignment with aligned original token, correct token, etc."""

    def __init__(self, orig: str, corr: str, matchrule: str, matchtype: str):
        self.orig = orig
        self.corr = corr
        self.matchrule = matchrule
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

    def iter_align(_orig, first_tok, rest):
        """
        Iteratively test alignment of all splits of _orig to first_tok and rest.
        Return overall best split.
        """
        lev_ratio_sum = 0
        _split = []
        for i in range(len(_orig) + 1):
            part1, part2 = _orig[:i], _orig[i:]
            pt1_ratio = lev_ratio(''.join(part1), first_tok)
            pt2_ratio = lev_ratio(''.join(part2), ''.join(rest))
            ratio_sum = pt1_ratio + pt2_ratio
            # If either part1 or part2 is a perfect match, return without iterating further
            if pt1_ratio == 1 or pt2_ratio == 1:
                return [sep.join(part1), sep.join(part2)]
            if ratio_sum > lev_ratio_sum:
                lev_ratio_sum = ratio_sum
                _split = [sep.join(part1), sep.join(part2)]
        return _split

    # If there is only one correct token, just return it with everything in the original tuple joined.
    if len(corr) == 1:
        return (sep.join(orig),), corr
    else:
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
                                         sep='_', orig_tokens=orig_tokens, corr_tokens=corr_tokens)


def align_ocr(original, corrected):
    """Align two sequences of tokens."""

    # Requirements:
    # - Resulting sequences must have the same length.
    # - If there are multiple corrected words for an original word, pair the closest match.

    def get_match_chunks(word_seq: tuple, a_or_b: int, seq_match):
        """Get matching chunks from SequenceMatcher object."""
        match_chunks = [SeqChunk(chunk=word_seq[idx[a_or_b]:idx[a_or_b] + idx.size],
                                 start=idx[a_or_b],
                                 match=True)
                        for idx in seq_match]
        return [mc for mc in match_chunks if mc.chunk]

    def get_nonmatch_chunks(word_seq: tuple, a_or_b: int, seq_match):
        """Get non-matching chunks from SequenceMatcher object.
            a_or_b: index 0 or 1 representing the a or b sequence in the SequenceMatcher."""
        nonmatch_start_idxs = [0] + [match[a_or_b] + match.size for match in seq_match]
        nonmatch_end_idxs = [match[a_or_b] for match in seq_match]
        nonmatch_idxs = list(zip(nonmatch_start_idxs, nonmatch_end_idxs))
        nonmatch_chunks = [SeqChunk(chunk=word_seq[idx[0]:idx[1]],
                                    start=idx[0],
                                    match=False)
                           for idx in nonmatch_idxs]
        return [nm for nm in nonmatch_chunks if nm.chunk]

    def make_ordered_chunks(orig: tuple, corr: tuple, seq_match):
        """Return matching and non-matching alignment chunks in correct order."""
        # TODO: Refactor to take (origtup, corrtup) tuple instead of either or? ("orig, 0" is redundant)
        orig_match_chunks = get_match_chunks(orig, 0, seq_match)
        orig_nonmatch_chunks = get_nonmatch_chunks(orig, 0, seq_match)
        corr_match_chunks = get_match_chunks(corr, 1, seq_match)
        corr_nonmatch_chunks = get_nonmatch_chunks(corr, 1, seq_match)
        orig_chunks = sorted(orig_match_chunks + orig_nonmatch_chunks, key=lambda chunk: chunk.start)
        corr_chunks = sorted(corr_match_chunks + corr_nonmatch_chunks, key=lambda chunk: chunk.start)

        def recursive_align(seq1, seq2):
            seq1_matches = [chunk.match for chunk in seq1]
            seq2_matches = [chunk.match for chunk in seq2]
            seq1_flat_chunks = [''.join(chunk.chunk) for chunk in seq1]
            seq2_flat_chunks = [''.join(chunk.chunk) for chunk in seq2]
            if seq1_matches == seq2_matches:
                return seq1, seq2
            else:
                # Find first divergence and fix it
                matchblocks = SequenceMatcher(None, seq1_flat_chunks, seq2_flat_chunks).get_matching_blocks()
                divergence_idx = [m.a == m.b for m in matchblocks].index(False)
                diverging_block = matchblocks[divergence_idx]
                # If seq1 has elements that seq2 doesn't: Fill with dummy SeqChunk. Note: 'start' val -1 is a dummy.
                if diverging_block.a > diverging_block.b:
                    seq2 = seq2[:diverging_block.b] + [SeqChunk(('_',), -1, False)] + seq2[diverging_block.b:]
                # The other way round
                else:
                    seq1 = seq1[:diverging_block.a] + [SeqChunk(('_',), -1, False)] + seq1[diverging_block.a:]
                return recursive_align(seq1, seq2)

        orig_chunks, corr_chunks = recursive_align(orig_chunks, corr_chunks)
        return [AlignChunk(orig_chunk=chunk[0].chunk,
                           corr_chunk=chunk[1].chunk) for chunk in zip(orig_chunks, corr_chunks)]

    def align_nonmatching(chunklist):
        """Align non-matching chunks."""

        def align_same_chars(chunk):
            """Align chunks where characters match, but whitespace doesn't."""

            # TODO: T o Mcend ncermede sig til B ordet. Den
            #  To Mænd nærmede sig til Bordet. Den
            #  [('Den', 'Bordet')]
            def make_rgx(chars): return f"(?:{'_*'.join([re.escape(c) for c in chars])})"

            orig_str = '_'.join(chunk.orig_chunk)
            corr_rgx = re.compile('|'.join([make_rgx(s) for s in chunk.corr_chunk]))
            matches = tuple(corr_rgx.findall(orig_str))
            chunk.orig_chunk = matches
            chunk.type = 'same_chars'
            return chunk

        def align_two_to_one(chunk):
            """Map best-matching of two correct words to original word, and the other to '_'.
                If no good match: return chunk as is."""

            def get_ratio(orig, corr):
                return SequenceMatcher(None, orig, corr).ratio()

            match_1 = get_ratio(chunk.orig_chunk[0], chunk.corr_chunk[0])
            match_2 = get_ratio(chunk.orig_chunk[0], chunk.corr_chunk[1])
            chunk.type = 'two_to_one'
            if match_1 > match_2 and match_1 > .5:
                chunk.orig_chunk = (chunk.orig_chunk[0], '_')
                return chunk
            elif match_2 > match_1 and match_2 > .5:
                chunk.orig_chunk = ('_', chunk.orig_chunk[0])
                return chunk
            else:
                return chunk

        def handle_hyphen(chunk):
            """Hyphenated words at end of line: Just join for now."""
            # TODO: Handle instances like this, where "tid" and "ligere" become separate tokens.
            #  Original: t i l M a g t derinde. Den omtalte Pensionist, der tid ligere
            #  Corrected: til Magt derinde. Den omtalte Pensionist, der tid[- ]ligere
            # Chunk has more than just the actual hyphenated word (first part is longer than 1)
            if len(chunk.corr_chunk[:chunk.corr_chunk.index('[')]) > 1:
                # Join parts that are not '[-]' with underscores.
                # TODO: Ought to be aligned properly instead ...
                chunk.orig_chunk = ('_'.join(chunk.orig_chunk),)
                chunk.corr_chunk = (re.sub(r'_?\[_-_\]_?', '[-]', '_'.join(chunk.corr_chunk)),)
                chunk.type = 'hyphen1'
            # Handle cases like ('u', 'v', 'ilkaarligt') - ('uvil', '[', '-', ']', 'kaarligt')
            elif ''.join(chunk.orig_chunk).startswith(chunk.corr_chunk[0]):
                chunk.orig_chunk = (''.join(chunk.orig_chunk),)
                chunk.corr_chunk = (''.join(chunk.corr_chunk),)
                chunk.type = 'hyphen2'
            else:
                # Join parts that are not '[-]' with underscores.
                chunk.orig_chunk = ('_'.join(chunk.orig_chunk),)
                chunk.corr_chunk = (re.sub(r'_?\[_-_\]_?', '[-]', '_'.join(chunk.corr_chunk)),)
                chunk.type = 'hyphen3'
            return chunk

        def align_partly_matches(chunk):
            """Align partly matching chunks."""
            new_orig_tuple, new_corr_tuple = recursive_token_align(chunk.orig_chunk, chunk.corr_chunk)
            chunk.orig_chunk = new_orig_tuple
            chunk.corr_chunk = new_corr_tuple
            chunk.type = 'part_match2'
            return chunk

        for chnk in chunklist:
            if not chnk.orig_chunk == chnk.corr_chunk:
                # Characters match, but whitespace doesn't.
                if ''.join(chnk.orig_chunk) == ''.join(chnk.corr_chunk):
                    chnk = align_same_chars(chnk)
                # Hyphenated words at end of line.
                elif '[-]' in ''.join(chnk.corr_chunk):
                    chnk = handle_hyphen(chnk)
                # Two correct words for one word in the original.
                elif len(chnk.orig_chunk) == 1 and len(chnk.corr_chunk) == 2:
                    chnk = align_two_to_one(chnk)
                # Chunk lengths are equal, and good partly match
                elif (len(chnk.orig_chunk) == len(chnk.corr_chunk) and
                      SequenceMatcher(None, ''.join(chnk.orig_chunk), ''.join(chnk.corr_chunk)).ratio() > .6):
                    chnk.type = 'part_match1'
                # Good partly match - figure out alignment by process of elimination.
                elif SequenceMatcher(None, ''.join(chnk.orig_chunk), ''.join(chnk.corr_chunk)).ratio() > .6:
                    chnk = align_partly_matches(chnk)
                # If nothing else, chunk lengths are equal - keep as is
                elif len(chnk.orig_chunk) == len(chnk.corr_chunk):
                    chnk.type = 'same_length'
                # Other/complicated cases - concatenate everything with '_'
                else:
                    chnk.orig_chunk = ('_'.join(chnk.orig_chunk),)
                    chnk.corr_chunk = ('_'.join(chnk.corr_chunk),)
                    chnk.type = 'other'
        return chunklist

    def chunks2alignment(chunklist, corrtup):
        """Make Alignment object from list of aligned chunks."""

        def determine_matches(orig_tokens, corr_tokens):
            """Determine whether (aligned) original token and correct token match."""
            matchlist = []
            for orig_token, corr_token in zip(orig_tokens, corr_tokens):
                # Hyphenated words at end of line: If first part matches original, consider it a match.
                # TODO: Handle hyphens more systematically ...
                if '[-]' in corr_token and corr_token != '[-]':
                    matchval = orig_token.startswith(corr_token.split('[-]')[0])
                else:
                    matchval = bool(orig_token == corr_token)
                matchlist.append(matchval)
            return tuple(matchlist)

        def determine_types(orig_tokens, corr_tokens, _types):
            """Determine the type of each token alignment."""
            typelist = []
            for orig_token, corr_token, orig_type in zip(orig_tokens, corr_tokens, _types):
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
                    typeval = orig_type
                typelist.append(typeval)
            return tuple(typelist)

        new_orig = tuple([token for chunk in chunklist for token in chunk.orig_chunk])
        new_corr = tuple([token for chunk in chunklist for token in chunk.corr_chunk])
        # Hack to get rid of added null tokens at beginning of correct line
        if new_corr[0] == '_':
            new_orig = tuple(['_'.join([new_orig[0], new_orig[1]])] + list(new_orig[2:]))
            new_corr = new_corr[1:]
        # Hack to get rid of remaining added null tokens in correct line
        if '_' in new_corr:
            indexes = [i for i, x in enumerate(new_corr) if x == '_']
            tweakd_orig = ['_' + x if i in indexes else x for i, x in enumerate(new_orig)]
            new_orig = tuple(' '.join(tweakd_orig).replace(' _', '_').split(' '))
            new_corr = tuple([x for x in new_corr if x != '_'])
        # A last hack to clean up the rest TODO Align to corrtup from the getgo!
        if len(new_orig) != len(corrtup):
            new_orig, new_corr = recursive_token_align(new_orig, corrtup)

        types = tuple([chunk.type for chunk in chunklist for _ in chunk.corr_chunk])
        matchtypes = determine_types(new_orig, new_corr, types)
        matches = determine_matches(new_orig, new_corr)
        align_toks = [AlignToken(*args) for args in zip(new_orig, new_corr, types, matchtypes)]
        return Alignment(aligned_tokens=align_toks, aligned_orig=new_orig, correct=new_corr,
                         types=types, matchtypes=matchtypes, matches=matches)

    origtup = tuple(word_tokenize(original, language='danish'))
    # Get rid of hyphens TODO Is this the right place ..
    corrected = re.sub(r'\[[ -]+\]', '', corrected)
    corrtup = tuple(word_tokenize(corrected, language='danish'))

    seq_matcher = SequenceMatcher(None, origtup, corrtup)
    matching = seq_matcher.get_matching_blocks()

    chunks = make_ordered_chunks(origtup, corrtup, matching)
    # print('Raw chunks:', chunks)
    chunks = align_nonmatching(chunks)
    # print('Chunks ...', chunks)
    alignment = chunks2alignment(chunks, corrtup)
    # Add unaltered, tokenized correct string as a tuple
    alignment.corr_tokens = corrtup
    # Add original and correct string
    alignment.orig_str = original
    alignment.corr_str = corrected
    print('Alignment:', alignment, '\n\n')
    return alignment


def main():
    orig = '„Hr. E ta tsra a d Helmer, tlieoloZios'
    corr = '„Hr. Etatsraad Helmer, Candidatus theologiæ'
    orig = """hendes Hensigt at syre Knuds Lig med sig t il L)t."""
    corr = """hendes Hensigt at føre Knuds Lig med sig til St."""
    alignment = align_ocr(orig, corr)
    print(alignment)


if __name__ == '__main__':
    main()
