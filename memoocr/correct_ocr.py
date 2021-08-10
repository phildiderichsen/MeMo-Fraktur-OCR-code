"""
correct_ocr.py
Correct individual OCR files from an input directory.
"""
# Spelling correction using symspell from Oliver
import os
import re
import sys
import myutils as util

from datetime import datetime
from symspellpy import SymSpell, Verbosity
from myutils import sorted_listdir, tokenize, fix_hyphens
from memoocr.align_ocr import align_b_to_a


def main():
    starttime = datetime.now()
    conf = util.get_config('DEFAULT')
    sym_wordcorrect(conf)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def correct_easy_fraktur_errors(uncorrected_dir, corrected_dir):
    """Manually correct 'safe' and easy OCR errors. Designed for the Tesseract fraktur traineddata."""
    # Sort novels, just because; then correct each novel
    sorted_novels = sorted_listdir(uncorrected_dir)
    for novel in sorted_novels:
        print(f'Running correct_easy_fraktur_errors() on {novel} ...\n')
        novel_str = get_novel_string(novel, uncorrected_dir)

        corrected_novel_str = re.sub(r'œæ', 'æ', novel_str)
        corrected_novel_str = re.sub(r'æœ', 'æ', corrected_novel_str)
        corrected_novel_str = re.sub(r'œe', 'æ', corrected_novel_str)
        corrected_novel_str = re.sub(r'eœ', 'æ', corrected_novel_str)
        corrected_novel_str = re.sub(r'œ', 'æ', corrected_novel_str)

        # Create output folder if not exists and write to file
        outfolder = os.path.join(corrected_dir, novel)
        try:
            os.makedirs(outfolder)
        except FileExistsError:
            pass
        outpath = os.path.join(outfolder, os.path.basename(novel) + '.corrected.txt')
        print(outpath)
        with open(outpath, 'w') as f:
            f.write(corrected_novel_str + "\n")


def correct_hard_fraktur_errors(uncorrected_dir, intermediate, corrected_dir):
    """Manually correct harder OCR errors by looking at 'dan' OCR. Designed for the Tesseract fraktur traineddata."""
    # Sort novels, just because; then correct each novel
    sorted_novels = [n for n in sorted_listdir(uncorrected_dir) if n != '.DS_Store']  # Hack alert!
    for novel in sorted_novels:
        print(f'Running correct_hard_fraktur_errors() on {novel} ...\n')
        novel_str = get_novel_string(novel, uncorrected_dir)
        dan_novel_str = get_novel_string(novel, os.path.join(intermediate, 'tess_out_dan'))
        frk_novel_str = get_novel_string(novel, os.path.join(intermediate, 'tess_out_frk'))
        kb_novel_str = get_novel_string(novel, os.path.join(intermediate, 'orig_pages'))

        dan_replacements = [('o', 'ø'), ('a', 'æ'), ('e', 'æ'), ('J', 'I'), ('t', 'k'), ('o', 'æ'), ('D', 'Ø')]
        corrected_novel_str = alt_ocr_correct(novel_str, dan_novel_str, dan_replacements)
        frk_replacements = [('t', 'k'), ('g', 'a')]
        corrected_novel_str = alt_ocr_correct(corrected_novel_str, frk_novel_str, frk_replacements)
        kb_replacements = [('J', 'I')]
        corrected_novel_str = alt_ocr_correct(corrected_novel_str, kb_novel_str, kb_replacements)

        # Create output folder if not exists and write to file
        outfolder = os.path.join(corrected_dir, novel)
        try:
            os.makedirs(outfolder)
        except FileExistsError:
            pass
        outpath = os.path.join(outfolder, os.path.basename(novel) + '.corrected.txt')
        print(outpath)
        with open(outpath, 'w') as f:
            f.write(corrected_novel_str + "\n")


def alt_ocr_correct(novel_str, alt_novel_str, replacements):
    """
    Replace x with y in a relatively safe way, informed by alternative OCR.
    The process is perfomed in chunks to make sure any odd local correction is not overgeneralized.
    """
    novel_str = novel_str.replace('¶', '___PILCROW___')
    novel_str = novel_str.replace('\n', ' ¶ ')
    alt_novel_str = alt_novel_str.replace('¶', '___PILCROW___')
    alt_novel_str = alt_novel_str.replace('\n', ' ¶ ')
    novel_tokens = tuple(tokenize(novel_str))
    alt_tokens = tuple(tokenize(alt_novel_str))
    aligned_alt_tokens = align_b_to_a(novel_tokens, alt_tokens)
    tokens_chunklist = util.chunk_list(list(zip(novel_tokens, aligned_alt_tokens)), n=250)
    novel_str_chunks = []
    for chunk in tokens_chunklist:
        novel_str_chunks.append(alt_ocr_correct_chunk(chunk, replacements))
    joined_novel_str_chunks = ' '.join(novel_str_chunks)
    joined_novel_str_chunks = joined_novel_str_chunks.replace(' ¶ ', '\n')
    joined_novel_str_chunks = joined_novel_str_chunks.replace('___PILCROW___', '¶')
    return joined_novel_str_chunks


def alt_ocr_correct_chunk(chunk, replacements):
    """In a single chunk, replace x with y in a relatively safe way, informed by alternative OCR."""
    chunk_novel_tokens = tuple([x[0] for x in chunk])
    chunk_aligned_alt_tokens = tuple([x[1] for x in chunk])
    chunk_novel_str = ' '.join(chunk_novel_tokens)
    corr_dict = dict()
    for char, repl in replacements:
        corr_dict.update(get_correction_dict(chunk_novel_tokens, chunk_aligned_alt_tokens, char, repl))
    rgx = re.compile(r'\b(' + '|'.join(map(re.escape, corr_dict.keys())) + r')\b')
    if corr_dict:
        print('corr_dict:', corr_dict)
        return rgx.sub(lambda match: corr_dict[match.group(0)], chunk_novel_str)
    else:
        return chunk_novel_str


def get_correction_dict(novel_tokens: tuple, aligned_alt_tokens: tuple, x: str, y: str):
    """Get string substitution pairs with char x replaced by y, based on information from alternative OCR."""

    def get_correction_pair(frakturtoken: str, alttoken: str, frakturchar='o', altchar='ø'):
        """If there are e.g. 'ø's in alttoken where frakturtoken has 'o's, replace the 'o's with 'ø's."""
        altchar_indexes = [i for i, char in enumerate(alttoken) if char == altchar]
        frakchars = list(frakturtoken)
        for i in altchar_indexes:
            if len(frakchars) > i:
                if frakchars[i] == frakturchar:
                    frakchars[i] = altchar
        replacement = ''.join(frakchars)
        return frakturtoken, replacement

    def good_pair(frak: str, alt: str, frakchar: str, altchar: str):
        """Can a useful correction pair be generated from the token pair frak: alt (will anything actually change?)"""
        # If the OCR form is among the most frequent words, don't correct it.
        if frak.lower() in util.most_frequent:
            return False
        # Note: This does not yield anything:
        # If the OCR form is a known word (on the freqlist) and the alternative form is not, don't correct it.
        # if frak.lower() in util.freqlist_forms and alt.lower() not in util.freqlist_forms:
        #    return False
        # Ensure that 'o' is actually in the OCR form, and 'ø' is in the alt form, for an 'o' => 'ø' correction.
        if not all([frakchar in frak, altchar in alt]):
            return False
        else:
            # We have a good pair if the set of frakchar indexes overlaps with the set of altchar indexes =>
            # at least one relevant replacement at the same index. E.g. 'tyste' and 'tyske' will overlap at i = 3.
            frak_indexes = set([i for i, char in enumerate(frak) if char == frakchar])
            alt_indexes = set([i for i, char in enumerate(alt) if char == altchar])
            return frak_indexes.intersection(alt_indexes)

    tokenpairs = zip(novel_tokens, aligned_alt_tokens)
    return dict([get_correction_pair(a, b, x, y) for a, b in tokenpairs if good_pair(a, b, x, y)])


def sym_wordcorrect(conf, uncorrected_dir, corrected_dir):
    """Correct OCR files from inputdir specified in config.ini - using word level SymSpell"""
    print("Initialize SymSpell")
    sym_spell = SymSpell()
    param_tuple, param_str = util.get_params(conf)
    dictionary_path = conf[param_tuple[1]]
    sym_spell.load_dictionary(dictionary_path, 0, 1)

    # Sort novels, just because; then correct each novel
    sorted_novels = sorted_listdir(uncorrected_dir)
    for novel in sorted_novels:
        print(f'Running sym_wordcorrect() on {novel} ...\n')
        novel_str = get_novel_string(novel, uncorrected_dir)
        # Correct individual words using SymSpell
        corrected_novel_str = word_correct_text(novel_str, sym_spell)
        # Create output folder if not exists and write to file
        outfolder = os.path.join(corrected_dir, novel)
        try:
            os.makedirs(outfolder)
        except FileExistsError:
            pass
        outpath = os.path.join(outfolder, os.path.basename(novel) + '.corrected.txt')
        print(outpath)
        with open(outpath, 'w') as f:
            f.write(corrected_novel_str + "\n")


def get_novel_string(novel, novels_dir):
    """Create a single string from novel pages."""
    # TODO Hack to accommodate missing page specifications on the dir name in the default case ...
    if not os.path.isdir(os.path.join(novels_dir, novel)):
        novel = re.sub(r'-s\d.{0,5}$', '', novel)
    novel_pages = sorted_listdir(os.path.join(novels_dir, novel))
    if not novel_pages:
        sys.exit(f'\nERROR: No pages found in {os.path.join(novels_dir, novel)}. Aborting.\n')
    # Create one big string from pages. Keep newlines.
    novel_pagestrings = get_novel_pagestrings(novel_pages, novels_dir, novel)
    novel_pagestrings = fix_hyphens(novel_pagestrings)
    novel_string = '\n'.join(novel_pagestrings)
    # Eliminate hyphenation in the text
    novel_string = '\n'.join(fix_hyphens([line for line in novel_string.splitlines()]))
    return novel_string


def spell_corrected(term, sym_spell, word_split):
    suggestions = sym_spell.lookup_compound(term,
                                            max_edit_distance=2,
                                            ignore_non_words=True,
                                            split_phrase_by_space=True,
                                            ignore_term_with_digits=False,
                                            transfer_casing=True)
    # Create lists to compare with/without punctuation
    # PD: This is brittle since a suggestion may well be a conflation of two 'word_split' items.
    # This means the chk_list may become shorter than the in_list, yielding an index error.
    # This is a segmentation problem right here.
    corrected = suggestions[0].term
    in_list = word_split.findall(term)
    chk_list = word_split.findall(corrected)
    # To keep punctuation we take the original phrase and do word by word replacement
    out_term = ''
    word_count = 0
    # if len(in_list) == len(chk_list):
    for word in in_list:
        if len(word) == 1:
            out_term = term.replace(word, word)
        else:
            out_term = term.replace(word, chk_list[word_count])
        word_count += 1
    return out_term


def word_correct_text(text, sym_spell):
    """Correct individual words in text using SymSpell. Keep newlines."""
    lines = text.splitlines()
    word_corr_lines = []
    for line in lines:
        tokens = tokenize(line)
        suggestion_tups = [(t, get_word_suggestion(t, sym_spell)) if len(t) > 1 else (t, t) for t in tokens]
        tokens = [tup[1] if tup[1] else tup[0] for tup in suggestion_tups]
        word_corr_lines.append(' '.join(tokens))
    return '\n'.join(word_corr_lines)


def get_word_suggestion(word, sym_spell):
    """Get symspell suggestion for a single _word."""
    # Remove obvious noise
    if word in ["*", "ð", "—", "——", "———", "—————"]:
        suggestion = None
    # Keep informative punctuation
    elif word in ["—", ",", ".", ":", ";", "-", "?", "!", "'", '"']:
        suggestion = word
    # If not noise of useful punctuation, use sym_spell to correct
    else:
        option = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=2, transfer_casing=True)
        if option:
            suggestion = option[0]._term
        else:
            suggestion = word
    return suggestion


def line_correct_text(text, sym_spell):
    # Regex pattern1 Word split function, used to keep punctuation later
    word_split = re.compile(r"[^\W]+", re.U)
    lines = text.splitlines()
    corrected_lines = [spell_corrected(line, sym_spell, word_split) for line in lines]
    return '\n'.join(corrected_lines)


def get_novel_pagestrings(sorted_pages, uncorrected_dir, novel):
    """Get meaningful lines from each page in novel as a string. Return list of page strings."""
    pages = []
    for page in sorted_pages:
        pagestring = get_novel_pagestring(uncorrected_dir, novel, page)
        pages.append(pagestring)
    return pages


def get_novel_pagestring(uncorrected_dir, novel, page):
    """Get meaningful lines from one novel page as a string"""

    # Some intuitive/heuristic criteria for noise lines:
    # Top of page (before first real line); short line; few real letters. Mask2: Similar criteria, but anywhere.
    def noise_ratio(s):
        """Ratio of non-frequent chars in s (with ' ' eliminated)."""
        s = s.replace(' ', '')
        return len(re.findall(r'[^a-zA-Z!;,.?]', s)) / len(s) if len(s) else 0

    def short_ratio(s):
        """Ratio of short tokens (1, 2 chars)."""
        tokenlist = s.split()
        n_short = len([len(tok) for tok in tokenlist if len(tok) < 3])
        return n_short / len(tokenlist)

    def blatant(line):
        """Is line a blatant noise line? (Many noise chars or many short tokens)."""
        return noise_ratio(line) > .6 or short_ratio(line) > .5

    # def linemask(i, line, frst):
    #     return i < frst and len(line) < 15 and len(re.findall(r'[a-zA-Z]', line)) < 6

    # def mask2(line):
    #     return len(line) < 15 and noise_ratio(line) > .6

    def get_first_l(_lines):
        """Return index of first plausible text line."""
        lines_ok = [noise_ratio(ln) < .3 and short_ratio(ln) < .5 and len(ln) > 10 for ln in _lines]
        return lines_ok.index(True)

    with open(os.path.join(uncorrected_dir, novel, page), "r") as f:
        pagetext = f.read()
        # Only non-empty lines
        lines = [line for line in pagetext.splitlines() if not re.match(r'\s*$', line)]

    if not lines:
        sys.stderr.write(f'WARNING: Empty page ({os.path.join(novel, page)}).\n')
        return ''
    # Identify (index of) first meaningful line
    try:
        first = get_first_l(lines)
    except ValueError:
        sys.stderr.write(f'WARNING: No meaningful lines on page ({os.path.join(novel, page)}).\n')
        return ''
    # Remove noise lines
    # TODO Removal of initial noise should be improved. ML classification?
    lines = [line for i, line in enumerate(lines) if i >= first and not blatant(line)]
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
