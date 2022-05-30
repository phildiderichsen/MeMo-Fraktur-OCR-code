"""
correct_ocr.py
Correct individual OCR files from an input directory.
"""
# Spelling correction using symspell from Oliver
import math
import os
import re
import sys
import myutils as util

from collections import Counter
from datetime import datetime
from symspellpy import SymSpell, Verbosity
from memoocr import ROOT_PATH
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


def correct_easy_fraktur_errors(files_to_process, uncorrected_dir, corrected_dir):
    """Manually correct 'safe' and easy OCR errors. Designed for the Tesseract fraktur traineddata."""
    for novel in [f.replace('.pdf', '') for f in files_to_process]:
        print(f'Running correct_easy_fraktur_errors() on {novel} ...\n')
        novel_str = get_novel_string(novel, uncorrected_dir)
        if not novel_str:
            print('WARNING (correct_easy_fraktur_errors()): No novel_str.')
            continue
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


def correct_hard_fraktur_errors(files_to_process, uncorrected_dir, intermediate, corrected_dir):
    """Manually correct harder OCR errors by looking at 'dan' OCR. Designed for the Tesseract fraktur traineddata."""
    for novel in [f.replace('.pdf', '') for f in files_to_process]:
        print(f'Running correct_hard_fraktur_errors() on {novel} ...\n')
        novel_str = get_novel_string(novel, uncorrected_dir)
        if not novel_str:
            print('WARNING (correct_hard_fraktur_errors()): No novel_str.')
            continue
        dan_novel_str = get_novel_string(novel, os.path.join(intermediate, 'tess_out_dan'))
        if not dan_novel_str:
            print('WARNING (correct_hard_fraktur_errors()): No dan_novel_str.')
            continue
        dan_replacements = [('o', 'ø'), ('a', 'æ'), ('e', 'æ'), ('J', 'I'), ('t', 'k'), ('o', 'æ'), ('D', 'Ø'),
                            ('u', 'n'), ('t', 'f'), ('t', 'l'), ('t', 'k')]
        corrected_novel_str = alt_ocr_correct(novel_str, dan_novel_str, dan_replacements)

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
    novel_tokens = tuple(util.tokenize(novel_str))
    alt_tokens = tuple(util.tokenize(alt_novel_str))
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


def sym_wordcorrect(files_to_process, conf, uncorrected_dir, corrected_dir):
    """Correct OCR files from inputdir specified in config.ini - using word level SymSpell"""
    print("Initialize SymSpell")
    sym_spell = SymSpell()
    param_tuple, param_str = util.get_params(conf)
    dictionary_path = conf[param_tuple[1]]
    # This is how unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered_names30000_augmented.txt was made :)
    # new_dic_path = os.path.join(os.path.dirname(dictionary_path),
    #                            'unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered_names30000_augmented.txt')
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=3, upper=42)
    # Lower: 1: 96.54; 2: 56.53; 3: 96.53; 4: 96.53; 40: 96.53; 400: 96.52; 4000: 96.43; 40000: 96.54; 400000: 96.62; 400000000: 96.64
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=100000): 96.55
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=10000): 96.69
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=7500): 96.69
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=5000): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=4000): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=3200): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=2900): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=2700): 96.70
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=2500): 96.69
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=3, upper=5000): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=10, upper=5000): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=100, upper=5000): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=1000, upper=5000): 96.68
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=10, upper=7500): 96.69
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=2, upper=6000): 96.69
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=10, upper=2900): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=100, upper=2900): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=1000, upper=2900): 96.69
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=150, upper=2900): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=190, upper=2900): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=200, upper=2900): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=250, upper=2900): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=300, upper=2900): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=320, upper=2900): 96.71
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=400, upper=2900): 96.70
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=500, upper=2900): 96.70
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=200, upper=3000): 96.76
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=200, upper=3200): 96.76
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=200, upper=3500): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=3, upper=41): 96.76
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=3, upper=42): 96.76
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=3, upper=43): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=3, upper=45): 96.74
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=4, upper=42): 96.76
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=4, upper=43): 96.75
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=4, upper=45): 96.74
    # make_augmented_dictionary(dictionary_path, new_dic_path, lower=5, upper=42): 96.71

    sym_spell.load_dictionary(dictionary_path, 0, 1)

    for novel in [f.replace('.pdf', '') for f in files_to_process]:
        print(f'Running sym_wordcorrect() on {novel} ...\n')
        novel_str = get_novel_string(novel, uncorrected_dir)
        if not novel_str:
            print('WARNING (sym_wordcorrect()): No novel_str.')
            continue
        # By-novel frequency augmentation
        if conf.name == 'correct':
            new_dic_path = os.path.join(ROOT_PATH, 'augmented_freqs.txt')
            make_novel_augmented_dictionary(dictionary_path, novel_str, new_dic_path, lower=2, upper=250)
            sym_spell.load_dictionary(new_dic_path, 0, 1)

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


def make_augmented_dictionary(dic_path, new_dic_path, lower=2, upper=10000):
    """Add raw OCR corpus frequency data to frequency dictionary"""
    with open(dic_path, 'r') as f:
        dicfreqs = [x.split() for x in f.read().splitlines()]
    corpusdicfile = 'Memo-testkorpus-1-brill-korp-alle-filer-i-et-korpus-freqs.wplus.not1.txt'
    with open(os.path.join(os.path.dirname(dic_path), corpusdicfile), 'r') as f:
        corpusfreqs = [x.split() for x in f.read().splitlines()]
    freqdic_20plus_sum = sum([int(x[1]) for x in dicfreqs if int(x[1]) >= 20])
    corpusfreqs_20plus_sum = sum([int(x[1]) for x in corpusfreqs if int(x[1]) >= 20])
    ratio = freqdic_20plus_sum / corpusfreqs_20plus_sum
    corpusfreqs = [x for x in corpusfreqs if lower <= int(x[1]) <= upper]
    new_corpusfreqs = {x[0]: (int(x[1]), math.ceil(int(x[1]) * ratio)) for x in corpusfreqs}
    freqdict = dict(dicfreqs)
    augmented_freqdict = freqdict.copy()
    for k, v in new_corpusfreqs.items():
        augmented_freqdict[k] = str(v[1])
    augmented_freqs_sorted = sorted([(int(v), k) for k, v in augmented_freqdict.items()], reverse=True)
    new_dic_str = '\n'.join([f'{k} {v}' for v, k in augmented_freqs_sorted])

    with open(new_dic_path, 'w') as f:
        f.write(new_dic_str)


def make_freqlist(novelstr):
    """Make a frequency list"""
    novelstring = novelstr.replace(util.PAGEBREAK, '').lower()
    tokens = util.tokenize(novelstring)
    tokens = [t for t in tokens if t not in ',.„“?!;—:»']
    freqs = Counter(tokens)
    sorted_freqs = sorted([f for f in freqs.items()], key=lambda x: (-x[1], x[0]))
    return sorted_freqs


def make_novel_augmented_dictionary(dic_file, novelstr, new_dic_path, lower=2, upper=10000):
    """Add raw OCR corpus frequency data from one novel to frequency dictionary"""
    with open(dic_file, 'r') as f:
        dicfreqs = [x.split() for x in f.read().splitlines()]
    novelfreqs = make_freqlist(novelstr)
    freqdic_top50_sum = sum([int(x[1]) for x in dicfreqs[:50]])
    novelfreqs_top50_sum = sum([int(x[1]) for x in novelfreqs[:50]])
    ratio = freqdic_top50_sum / novelfreqs_top50_sum
    novelfreqs = [x for x in novelfreqs if lower <= int(x[1]) <= upper]
    new_corpusfreqs = {x[0]: (int(x[1]), math.ceil(int(x[1]) * ratio)) for x in novelfreqs}
    freqdict = dict(dicfreqs)
    augmented_freqdict = freqdict.copy()
    for k, v in new_corpusfreqs.items():
        augmented_freqdict[k] = str(v[1])
    augmented_freqs_sorted = sorted([f for f in augmented_freqdict.items()], key=lambda x: (-int(x[1]), x[0]))
    new_dic_str = '\n'.join([f'{tup[0]} {tup[1]}' for tup in augmented_freqs_sorted])
    with open(new_dic_path, 'w') as outfile:
        outfile.write(new_dic_str)


def get_novel_string(novel, novels_dir):
    """Create a single string from novel pages."""
    # TODO Hack to accommodate missing page specifications on the dir name in the default case ...
    if not os.path.isdir(os.path.join(novels_dir, novel)):
        novel = re.sub(r'-s\d.{0,5}$', '', novel)
    novel_pages = util.sorted_listdir(os.path.join(novels_dir, novel))
    if not novel_pages:
        print(f'\nERROR: No pages found in {os.path.join(novels_dir, novel)}. Skipping.\n')
        return None

    # Create one big string from pages. Keep newlines.
    novel_pagestrings = get_novel_pagestrings(novel_pages, novels_dir, novel)
    novel_pagestrings = util.fix_hyphens(novel_pagestrings)
    novel_string = f' {util.PAGEBREAK} '.join(novel_pagestrings)
    # Eliminate hyphenation in the text
    novel_string = '\n'.join(util.fix_hyphens([line for line in novel_string.splitlines()]))
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
        tokens = util.tokenize(line)
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
            # Cancel suggestion for a few select false positives.
            if (word, suggestion) in [('Hr', 'Er'), ('Høgefjer', 'Søgefjer'), ('efterlært', 'efterlæst'),
                                      ('Gjæstemildhed', 'Gjcestemildhed'), ('bedachtsam', 'bedachfsam'),
                                      ('Eunucherne', 'Puncherne'), ('Hofpersonale', 'Togpersonale'),
                                      ('Fyrstesøn', 'Fyrslesøn'), ('müssen', 'messen'), ('Zeit', 'Seit'),
                                      ('benutzen', 'bendtsen'), ('Størreparten', 'tørveparten'), ('trangt', 'fragt'),
                                      ('Indtagelsen', 'Undtagelsen'), ('Stormand', 'formand'),
                                      ('vollendet', 'vollenden'), ('Für', 'For'), ('Liedlein', 'Kindlein'),
                                      ('erdacht', 'erwacht'), ('sie', 'sig'), ('Sie', 'Sig'), ('Mädchen', 'Madchen'),
                                      ('Fos', 'For'), ('Afkjølende', 'Afkjølede'), ('Spydstikket', 'Spydstokkes')]:
                suggestion = word
            if util.PAGEBREAK in word:
                suggestion = word
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

    def is_space_or_pagenum(line, i):
        """Identify lines consisting of only spaces and page numbers at top of page."""
        if re.match(r'\s*$', line):
            return True
        if i == 0 and re.match(r'\s*\w{1,3}\s*$', line):
            return True
        else:
            return False

    # Handle BOM ...
    bom = '\ufeff'
    with open(os.path.join(uncorrected_dir, novel, page), mode='r', encoding='utf-8') as f:
        pagetext = f.read()
        if pagetext.startswith(bom):
            pagetext = pagetext[1:]

    # Only non-empty lines, and not pagenumbers at top of page.
    lines = [line for i, line in enumerate(pagetext.splitlines()) if not is_space_or_pagenum(line, i)]
    if not lines:
        sys.stderr.write(f'WARNING: Empty page ({os.path.join(novel, page)}).\n')
        return ''

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
