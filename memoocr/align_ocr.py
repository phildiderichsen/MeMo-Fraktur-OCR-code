import difflib
from nltk import word_tokenize


# Note: In order for nltk to tokenize with language='danish', the Punkt Tokenizer Models have to be installed.
# Create a nltk_data dir, e.g. /Users/phb514/nltk_data or /usr/local/share/nltk_data
# Create a subdir /Users/phb514/nltk_data/tokenizers
# Download Punkt Tokenizer Models from http://www.nltk.org/nltk_data
# Extract the downloaded punkt.zip, and place the "punkt" dir in /Users/phb514/nltk_data/tokenizers.


class Alignment(object):
    """Object with aligned original string as tuple and correct string as tuple.
        Ratio is a correctness measure."""

    def __init__(self, aligned: tuple, correct: tuple, ratio: float):
        self.aligned = aligned
        self.correct = correct
        self.ratio = ratio

    def __repr__(self):
        """Human-readable representation of Alignment"""
        attr_reprs = [f'"{k}": "{v}"' for k, v in self.__dict__.items() if not k.startswith('__')]
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
    # - Resulting sequences must have at least something in common (ratio of > .7?)
    # - Resulting sequences must have the same length.
    # - Non-matching words that are paired must have at least something in common.
    # - If there are multiple corrected words for an original word, pair the closest match.

    origtup = tuple(word_tokenize(original, language='danish'))
    corrtup = tuple(word_tokenize(corrected, language='danish'))

    seq_matcher = difflib.SequenceMatcher(None, origtup, corrtup)
    matching = seq_matcher.get_matching_blocks()
    ratio = seq_matcher.ratio()

    print('Original tuple:', origtup)
    print('Corrected tuple:', corrtup)
    print('Sequence match:', matching)
    print()

    def get_match_chunks(word_seq: tuple, a_or_b: int, seq_match):
        """Get non-matching chunks from SequenceMatcher object."""
        match_chunks = [SeqChunk(chunk=word_seq[idx[a_or_b]:idx[a_or_b] + idx.size],
                                 start=idx[a_or_b],
                                 match=True) for idx in seq_match]
        return [mc for mc in match_chunks if mc.chunk]

    def get_nonmatch_chunks(word_seq: tuple, a_or_b: int, seq_match):
        """Get non-matching chunks from SequenceMatcher object.
            a_or_b: index 0 or 1 representing the a or b sequence in the SequenceMatcher.
            TODO: Make docstring clearer ..."""
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
        for chunk in chunklist:
            if not chunk.match:
                chunk.orig_chunk = ('_'.join(chunk.orig_chunk),)
                chunk.corr_chunk = ('_'.join(chunk.corr_chunk),)
        return chunklist

    def chunks2alignment(chunklist, matchratio):
        """Make Alignment object from list of aligned chunks."""
        new_orig = tuple([token for chunk in chunklist for token in chunk.orig_chunk])
        new_corr = tuple([token for chunk in chunklist for token in chunk.corr_chunk])
        return Alignment(new_orig, new_corr, matchratio)

    chunks = make_ordered_chunks(origtup, corrtup, matching)
    chunks = align_nonmatching(chunks)
    print('Chunks:', chunks)
    print()
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
