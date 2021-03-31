"""
correct_ocr.py
Correct individual OCR files from an input directory.
"""
# Spelling correction using symspell from Oliver
import configparser
import os
import re
import sys

from datetime import datetime
from symspellpy import SymSpell, Verbosity
from nltk import word_tokenize


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))

    correct_ocr(config['DEFAULT'])

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def correct_ocr(conf):
    """Correct OCR files from inputdir specified in config.ini """
    print("Initialize SymSpell")
    sym_spell = SymSpell()
    dictionary_path = os.path.join(conf["metadir"], "frequency_dict_da_sm.txt")
    bigram_path = os.path.join(conf["metadir"], "bigrams_dict_da_sm.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    # Sort novels, just because; then correct each novel
    uncorrected_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    sorted_novels = sorted(os.listdir(uncorrected_dir))
    for novel in sorted_novels:
        corrected_novel_str = correct_novel(novel, uncorrected_dir, sym_spell)
        # Create output folder if not exists and write to file
        outfolder = os.path.join(conf['intermediatedir'], '3-corrected', novel)
        try:
            os.makedirs(outfolder)
        except FileExistsError:
            pass
        outpath = os.path.join(outfolder, os.path.basename(novel) + '.corrected.txt')
        print(outpath)
        with open(outpath, 'w') as f:
            f.write(corrected_novel_str + "\n")


def correct_novel(novel, uncorrected_dir, sym_spell):
    """Correct OCR text from a whole novel."""
    print(f"Working on {novel}")
    novel_pages = os.listdir(os.path.join(uncorrected_dir, novel))
    # Sort the pages, so that they're appended in the correct order
    novel_pages.sort(key=natural_keys)

    # Create one big string from pages. Keep newlines.
    novel_string = get_novel_lines(novel_pages, uncorrected_dir, novel)
    # Eliminate hyphenation in the text
    novel_string = handle_hyphenation(novel_string)
    # Correct individual words using SymSpell
    novel_string = word_correct_text(novel_string, sym_spell)
    # Correct lines using SymSpell
    # novel_string = line_correct_text(novel_string, sym_spell)
    return novel_string


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """

    def atoi(_text):
        return int(_text) if _text.isdigit() else _text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


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
        tokens = word_tokenize(line, language='danish')
        tokens = [get_word_suggestion(t, sym_spell) if len(t) > 1 else t for t in tokens]
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


def get_novel_lines(sorted_pages, uncorrected_dir, novel):
    """Get all meaningful lines from all pages in novel as one big string."""

    # Some intuitive/heuristic criteria for noise lines:
    # Top of page (before first real line); short line; few real letters. Mask2: Similar criteria, but anywhere.
    def noise_ratio(s): return len(re.findall(r'[^a-zA-Z!;,.?]', s)) / len(s) if len(s) else 0
    def linemask(i, line, frst): return i < frst and len(line) < 15 and len(re.findall(r'[a-zA-Z]', line)) < 6
    def mask2(line): return len(line) < 15 and noise_ratio(line) > .6

    def get_first_l(_lines):
        """Return index of first plausible text line."""
        lines_ok = [noise_ratio(line) < .2 and len(line) > 10 for line in _lines]
        return lines_ok.index(True)

    pages = []
    for page in sorted_pages:
        with open(os.path.join(uncorrected_dir, novel, page), "r") as f:
            # Note: Only non-empty lines
            lines = [line for line in f.read().splitlines() if line]
        if not lines:
            sys.stderr.write(f'WARNING: Empty page ({os.path.join(novel, page)}).\n')
            continue
        # Identify (index of) first meaningful line
        try:
            first = get_first_l(lines)
        except ValueError:
            sys.stderr.write(f'WARNING: No meaningful lines on page ({os.path.join(novel, page)}).\n')
            continue
        # Remove noise lines
        lines = [line for i, line in enumerate(lines) if not linemask(i, line, first) and not mask2(line)]
        # Remove empty lines
        lines = [line for line in lines if not re.match(r'\s*$', line)]
        pages.append('\n'.join(lines))
    novel_string = '\n'.join(pages)
    return novel_string


def handle_hyphenation(text):
    """ELiminate hyphenation in text - keep newlines."""
    hyphen_corrected = re.sub(r'(\S+)[⸗—-]\n(\S+)', r'\1\2\n', text)
    newline_corrected = re.sub(r'\n+', '\n', hyphen_corrected)
    return newline_corrected


if __name__ == '__main__':
    main()
