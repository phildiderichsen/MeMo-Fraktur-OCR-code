# Spelling correction using symspell from Oliver
import configparser
import os
import re

from datetime import datetime
from symspellpy import SymSpell, Verbosity
from nltk import word_tokenize


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
    # => Fixing it by only doing the SymSpell correction when in_list and chk_list are equally long.
    # Edit: Dropped this again ..
    corrected = suggestions[0].term
    in_list = word_split.findall(term)
    chk_list = word_split.findall(corrected)
    # To keep punctuation we take the original phrase and do word by word replacement
    out_term = []
    word_count = 0
    # if len(in_list) == len(chk_list):
    for word in in_list:
        if len(word) == 1:
            out_term = term.replace(word, word)
        else:
            out_term = term.replace(word, chk_list[word_count])
        word_count += 1
        # print('Out_term iteration:', out_term)
    # print('Out_term:', out_term)
    return out_term


def spell_correct_line(line, sym_spell):
    suggestions = sym_spell.lookup_compound(line,
                                            max_edit_distance=2,
                                            ignore_non_words=True,
                                            split_phrase_by_space=True,
                                            ignore_term_with_digits=False,
                                            transfer_casing=True)
    return suggestions[0].term


def correct_page_orig(conf, novel, page, sym_spell):
    """Correct a page of OCR text"""
    uncorrected_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    # Regex pattern1 - remove number from words like '55This'
    # num_remove = re.compile(r"\d*([^\d\W]+)\d*")
    # Regex pattern2 - pad punctuation with whitespace
    punct_pad = re.compile('([.,:;„"»«\'!?()])')
    # Regex pattern3 - strip addtional whitespace
    whitespace = re.compile(r'\s{2,}')
    # Regex pattern1 Word split function, used to keep punctuation later
    word_split = re.compile(r"[^\W]+", re.U)

    # Read the text in
    with open(os.path.join(uncorrected_dir, novel, page), "r") as f:
        text = f.read()

    # Pad punctuation with whitespace, so they count as tokens
    text = re.sub(punct_pad, r' \1 ', text)
    text = re.sub(whitespace, ' ', text)
    # list for sym_spell output
    output = []
    # Flatten lists into single string
    for word in text.split():
        suggestion = get_word_suggestion(word, sym_spell)
        if suggestion:
            output.append(suggestion)

    joined = ' '.join(output)
    joined = re.sub(punct_pad, r' \1 ', joined)
    joined = re.sub(whitespace, ' ', joined)
    # print('Joined:', joined)
    # Join corrected unigrams; check for bigrams
    suggested = spell_corrected(joined, sym_spell, word_split)
    # print('suggested:', suggested)
    # Join on punctuation again - replace with Regex
    suggested = ''.join(suggested).replace(" . ", ". ") \
        .replace(" , ", ", ") \
        .replace(" ; ", "; ") \
        .replace(" : ", ": ") \
        .replace(" ? ", "? ") \
        .replace(" ! ", "! ") \
        .replace(" “ ", " “ ") \
        .replace(" „ ", "„") \
        .replace(" “„ ", "“ „")
    return suggested


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


def word_correct_text(text, sym_spell):
    """Correct individual words in text using SymSpell. Keep newlines."""
    lines = text.splitlines()
    word_corr_lines = []
    for line in lines:
        tokens = word_tokenize(line, language='danish')
        tokens = [get_word_suggestion(t, sym_spell) if len(t) > 1 else t for t in tokens]
        word_corr_lines.append(' '.join(tokens))
    return '\n'.join(word_corr_lines)


def line_correct_text(text, sym_spell):
    lines = text.splitlines()
    corrected_lines = [spell_correct_line(line, sym_spell) for line in lines]
    return '\n'.join(corrected_lines)


def write_corrected_text(suggested, outfolder, filename):
    """Write suggested (= corrected) lines to file"""
    outpath = os.path.join(outfolder, filename)
    print(outpath)
    with open(outpath, 'w') as f:
        f.write(suggested + "\n")


def get_novel_lines(sorted_pages, uncorrected_dir, novel):
    """Get all meaningful lines from all pages in novel as one big string."""
    # Some intuitive criteria for initial noise lines:
    # Top of page; short line; few real letters.
    def linemask(i, line): return i < 5 and len(line) < 10 and len(re.findall(r'[a-zA-Z]', line)) < 5

    pages = []
    for page in sorted_pages:
        with open(os.path.join(uncorrected_dir, novel, page), "r") as f:
            # Note: Only non-empty lines
            lines = [line for line in f.read().splitlines() if line]
        # Remove initial noise lines
        lines = [line for i, line in enumerate(lines) if not linemask(i, line)]
        pages.append('\n'.join(lines))
    novel_string = '\n'.join(pages)
    return novel_string


def handle_hyphenation(text):
    """ELiminate hyphenation in text - keep newlines."""
    return re.sub(r'(\S+)[—-]\n(\S+)', r'\1\2\n', text)


def correct_ocr(conf):
    """Correct OCR files specified in inputdir specified in config.ini """
    print("Initialize SymSpell")
    sym_spell = SymSpell()
    dictionary_path = os.path.join(conf["metadir"], "frequency_dict_da_sm.txt")
    bigram_path = os.path.join(conf["metadir"], "bigrams_dict_da_sm.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    # Sort novels, just because
    uncorrected_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    sorted_novels = os.listdir(uncorrected_dir)
    # For each novel
    for novel in sorted_novels:
        print(f"Working on {novel}")
        outfolder = os.path.join(conf['intermediatedir'], '3-corrected', novel)
        # Create output folder if not exists
        try:
            os.makedirs(outfolder)
        except FileExistsError:
            pass
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
        novel_string = line_correct_text(novel_string, sym_spell)
        # Write to file
        outname = os.path.basename(novel) + '.corrected.txt'
        write_corrected_text(novel_string, outfolder, outname)


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))

    mode = 'test'
    correct_ocr(config[mode])

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
