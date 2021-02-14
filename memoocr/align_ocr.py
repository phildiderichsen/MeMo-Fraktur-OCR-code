from difflib import SequenceMatcher
import re

from nltk import word_tokenize


# Note: In order for nltk to tokenize with language='danish', the Punkt Tokenizer Models have to be installed.
# Create a nltk_data dir, e.g. /Users/phb514/nltk_data or /usr/local/share/nltk_data
# Create a subdir /Users/phb514/nltk_data/tokenizers
# Download Punkt Tokenizer Models from http://www.nltk.org/nltk_data
# Extract the downloaded punkt.zip, and place the "punkt" dir in /Users/phb514/nltk_data/tokenizers.


class Alignment(object):
    """Object with aligned original string as tuple and correct string as tuple.
        Ratio is a correctness measure."""

    def __init__(self, aligned_orig: tuple, correct: tuple, matches: tuple, ratio: float):
        self.aligned_orig = aligned_orig
        self.correct = correct
        self.matches = matches
        self.ratio = ratio
        self.prop_correct = sum(self.matches) / len(matches)

    def __repr__(self):
        """Human-readable representation of Alignment"""
        attr_reprs = [f'\n"{k}": "{v}"' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'Alignment({", ".join(attr_reprs)})'


class SeqChunk(object):
    """Intermediate chunk of sequence (either orig or corr) - either matching or not."""

    def __init__(self, chunk: tuple, start: int, match: bool):
        self.chunk = chunk
        self.start = start
        self.match = match

    def __repr__(self):
        """Human-readable representation of SeqChunk"""
        attr_reprs = [f'"{k}": "{v}"' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'SeqChunk({", ".join(attr_reprs)})'


class AlignChunk(object):
    """Intermediate chunk of alignment (with orig and corr) - either matching or not."""

    def __init__(self, orig_chunk: tuple, corr_chunk: tuple, match: bool):
        self.orig_chunk = orig_chunk
        self.corr_chunk = corr_chunk
        self.match = match

    def __repr__(self):
        """Human-readable representation of SeqChunk"""
        attr_reprs = [f'"{k}": "{v}"' for k, v in self.__dict__.items() if not k.startswith('__')]
        return f'AlignChunk({", ".join(attr_reprs)})'


def align_ocr(original, corrected):
    """Recursively align two sequences of tokens."""
    # Requirements:
    # - Resulting sequences must have the same length.
    # - If there are multiple corrected words for an original word, pair the closest match.

    def get_match_chunks(word_seq: tuple, a_or_b: int, seq_match):
        """Get non-matching chunks from SequenceMatcher object."""
        match_chunks = [SeqChunk(chunk=word_seq[idx[a_or_b]:idx[a_or_b] + idx.size],
                                 start=idx[a_or_b],
                                 match=True) for idx in seq_match]
        return [mc for mc in match_chunks if mc.chunk]

    def get_nonmatch_chunks(word_seq: tuple, a_or_b: int, seq_match):
        """Get non-matching chunks from SequenceMatcher object.
            a_or_b: index 0 or 1 representing the a or b sequence in the SequenceMatcher."""
        nonmatch_start_idxs = [0] + [match[a_or_b] + match.size for match in seq_match]
        nonmatch_end_idxs = [match[a_or_b] for match in seq_match]
        nonmatch_idxs = list(zip(nonmatch_start_idxs, nonmatch_end_idxs))
        nonmatch_chunks = [SeqChunk(chunk=word_seq[idx[0]:idx[1]],
                                    start=idx[0],
                                    match=False) for idx in nonmatch_idxs]
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
        return [AlignChunk(orig_chunk=chunk[0].chunk,
                           corr_chunk=chunk[1].chunk,
                           match=chunk[0].match) for chunk in zip(orig_chunks, corr_chunks)]

    def align_nonmatching(chunklist):
        """Align non-matching chunks."""

        def align_same_chars(chunk):
            """Align chunks where characters match, but whitespace doesn't."""
            def make_rgx(chars): return f"(?:{'_*'.join([re.escape(c) for c in chars])})"
            orig_str = '_'.join(chunk.orig_chunk)
            corr_rgx = re.compile('|'.join([make_rgx(s) for s in chunk.corr_chunk]))
            matches = tuple(corr_rgx.findall(orig_str))
            chunk.orig_chunk = matches
            return chunk

        def align_two_to_one(chunk):
            """Map best-matching of two correct words to original word, and the other to '_'.
                If no good match: return chunk as is."""
            def get_ratio(orig, corr): return SequenceMatcher(None, orig, corr).ratio()
            match_1 = get_ratio(chunk.orig_chunk[0], chunk.corr_chunk[0])
            match_2 = get_ratio(chunk.orig_chunk[0], chunk.corr_chunk[1])
            if match_1 > match_2 and match_1 > .5:
                chunk.orig_chunk = (chunk.orig_chunk[0], '_')
                return chunk
            elif match_2 > match_1 and match_2 > .5:
                chunk.orig_chunk = ('_', chunk.orig_chunk[0])
                return chunk
            else:
                return chunk

        def handle_hyphen(chunk):
            """Hyphenated words at end of line:
                If first part of corrected word matches original word, consider it a match."""
            if chunk.orig_chunk[0].startswith(chunk.corr_chunk[0]):
                chunk.match = True
            chunk.corr_chunk = (''.join(chunk.corr_chunk),)
            return chunk

        def align_partly_matches(chunk):
            """Figure out partly matching chunks."""
            def make_rgx(chars): return f"({'_*'.join([re.escape(c) for c in chars])})"
            orig_str = '_'.join(chunk.orig_chunk)
            corr_str = '_'.join(chunk.corr_chunk)
            # Collect correct tokens that match the OCR string.
            matching_correct_words = []
            for token in chunk.corr_chunk:
                rgx = make_rgx(token)
                if re.search(rgx, orig_str):
                    matching_correct_words.append(token)
            # If the matching tokens match as a single combined regex, we're on to something.
            combined_rgx = f"(.*)({'.*'.join([make_rgx(m) for m in matching_correct_words])})(.*)"
            if re.search(combined_rgx, orig_str):
                # Identify split points based on matching parts
                for token in matching_correct_words:
                    matching_correct_rgx = make_rgx(token)
                    orig_str = re.sub(matching_correct_rgx, r'<split>\1', orig_str)
            aligned_list = [x.strip('_') for x in orig_str.split('_<split>')]
            aligned_tuple = tuple([x.replace('<split>', '') for x in aligned_list])
            chunk.orig_chunk = aligned_tuple
            return chunk

        for chnk in chunklist:
            if not chnk.match:
                # Characters match, but whitespace doesn't.
                if ''.join(chnk.orig_chunk) == ''.join(chnk.corr_chunk):
                    chnk = align_same_chars(chnk)
                # Hyphenated words at end of line.
                elif len(chnk.orig_chunk) == 1 and '[-]' in ''.join(chnk.corr_chunk):
                    chnk = handle_hyphen(chnk)
                # Two correct words for one word in the original.
                elif len(chnk.orig_chunk) == 1 and len(chnk.corr_chunk) == 2:
                    chnk = align_two_to_one(chnk)
                # Good partly match - figure out alignment by process of elimination.
                elif SequenceMatcher(None, ''.join(chnk.orig_chunk), ''.join(chnk.corr_chunk)).ratio() > .6:
                    chnk = align_partly_matches(chnk)
                # If nothing else, chunk lengths are equal - keep as is
                elif len(chnk.orig_chunk) == len(chnk.corr_chunk):
                    pass
                # Other/complicated cases - concatenate everything with '_'
                else:
                    chnk.orig_chunk = ('_'.join(chnk.orig_chunk),)
                    chnk.corr_chunk = ('_'.join(chnk.corr_chunk),)
        return chunklist

    def chunks2alignment(chunklist, matchratio):
        """Make Alignment object from list of aligned chunks."""
        new_orig = tuple([token for chunk in chunklist for token in chunk.orig_chunk])
        new_corr = tuple([token for chunk in chunklist for token in chunk.corr_chunk])
        matches = tuple([chunk.match for chunk in chunklist for _ in chunk.corr_chunk])
        return Alignment(new_orig, new_corr, matches, matchratio)

    origtup = tuple(word_tokenize(original, language='danish'))
    corrtup = tuple(word_tokenize(corrected, language='danish'))

    seq_matcher = SequenceMatcher(None, origtup, corrtup)
    matching = seq_matcher.get_matching_blocks()
    ratio = seq_matcher.ratio()

    print('Original:', original)
    print('Corrected:', corrected)
    print('Token sequence match:', matching)
    print()

    chunks = make_ordered_chunks(origtup, corrtup, matching)
    chunks = align_nonmatching(chunks)
    alignment = chunks2alignment(chunks, ratio)
    print('Aligned Chunks:', alignment)
    print()
    print()
    return alignment


def main():
    orig = '„Hr. E ta tsra a d Helmer, tlieoloZios'
    corr = '„Hr. Etatsraad Helmer, Candidatus theologiæ'
    align_ocr(orig, corr)


if __name__ == '__main__':
    main()
