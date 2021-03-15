"""
Analyze MeMo OCR cleaning

Take a folder of OCR'ed text data and compare it to gold standard data.

OCR vs. gold:
- Report word error rate (WER) and character error rate (CER). (OCR vs. gold).
    - CER (Kettunen & Koistinen 2019) is equivalent to levenshtein / word length.
- Report average levenshtein and ratio per line.
- Report average levenshtein and ratio per text.
- Report overall average levenshtein and ratio.
- Show an overview of errors/error types: Aggregate by levenshtein distance.

Baseline vs. our cleaning procedure:
- Report precision, recall, F1
    - True positive: Correcting an actual error in the baseline data.
    - False negative: Failing to correct an actual error in the baseline data.
    - False positive: Correcting a non-error in the baseline data.
    - Precision: True positives / true and false positives. (How many corrections are valid).
    - Recall: True positives / true positives and false negatives. (How many actual errors were corrected).
    - See https://en.wikipedia.org/wiki/Precision_and_recall#/media/File:Precisionrecall.svg
    - F1 = 2 * Precision * Recall / (Precision + Recall). (Balance between P & R).
"""

import configparser
import os
import re
import statistics
from difflib import SequenceMatcher
import Levenshtein as lev

import pandas as pd

pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False

from evalocr.align_ocr import align_ocr


def main():
    """Run the evaluation."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    ocrdata = get_ocr_data(config)
    evaldata = get_gold_data(config)
    align_lines(ocrdata, evaldata)
    # make_eval_df(ocrdata, evaldata, use_cache=True)


def blaha(evaldata):
    print('ALIGNMENT EXAMPLES')
    print_align_examples(evaldata, ratio=.99)

    eval_df = make_eval_df(evaldata, use_cache=True)

    print('ERROR STATISTICS BY TYPE')
    print(make_freq_breakdown(eval_df, 'type'))
    print()

    print('ERROR STATISTICS BY OPERATION')
    print(make_opcode_breakdown(eval_df, n=3))
    print()

    print('ERROR STATISTICS BY MATCH')
    print(make_freq_breakdown(eval_df, 'match'))
    print()

    print('ERROR STATISTICS BY LEVENSHTEIN DISTANCE')
    print(make_freq_breakdown(eval_df, 'lev_dist'))
    print()

    print('LEVENSHTEIN > 3')
    large_lev_dist = eval_df.loc[eval_df.lev_dist > 3][['aligned_orig', 'correct', 'lev_dist', 'type', 'orig_line']]
    print(large_lev_dist)
    print()
    print('SAME-CHAR ERRORS')
    same_char_data = eval_df.loc[eval_df['type'] == 'same_chars'][['aligned_orig', 'correct', 'orig_line', 'corr_line']]
    print(same_char_data)
    print()
    print('SAME-CHAR ERRORS AGGREGATED')
    same_char_agg = eval_df.loc[eval_df['type'] == 'same_chars'] \
        .groupby('correct') \
        .agg({'correct': 'count', 'aligned_orig': lambda x: str(set(x))})
    print(same_char_agg)
    print()
    print('CORRECT ONE-CHAR TOKENS')
    one_char_correct = eval_df.loc[eval_df['correct'].str.len() == 1] \
        .groupby('correct') \
        .agg({'correct': 'count'})
    print(one_char_correct)
    correct_one_chars = eval_df.loc[eval_df['correct'].isin(list('aioIO'))] \
        .groupby('correct') \
        .agg({'correct': 'count', 'aligned_orig': lambda x: str(set(x))})
    print(correct_one_chars)
    correct_orig_one_chars = eval_df.loc[eval_df['aligned_orig'].isin(list('aioIO'))] \
        .groupby('aligned_orig') \
        .agg({'aligned_orig': 'count', 'correct': lambda x: str(set(x))})
    print(correct_orig_one_chars)

    # Create data for unit testing
    # large_lev_breakdown = tabulate_large_lev_cases(eval_df, n=1, m=2)
    # print(large_lev_breakdown)
    # print('\n'.join(large_lev_breakdown['pair'].to_list()))
    # print(large_lev_breakdown.shape)


def get_ocr_data(config):
    """
    Get a dict from novel title (with year etc.) to text from the first page available for that novel,
    from a folder of novel folders each containing one or more pages of OCR text.
    """
    ocrdir = config['DEFAULT']['ocrdir_for_eval']
    datadict = {}
    for foldername in sorted(os.listdir(ocrdir)):
        orig_pdfname = foldername.split('-')[0]
        firstfile = sorted(os.listdir(os.path.join(ocrdir, foldername)))[0]
        filepath = os.path.join(ocrdir, foldername, firstfile)
        with open(filepath, 'r', encoding='utf8') as ocrfile:
            ocrdata = ocrfile.read()
        datadict[orig_pdfname] = ocrdata
    return datadict


def get_gold_data(config):
    """Get a dict from novel title (with year etc.) to text from the first page available for that novel."""
    df = pd.read_csv(config['DEFAULT']['evaldata'], sep='\t', encoding='utf8')
    # Remove rows with [paragraph] etc.
    df = df[~df.iloc[:, 1].str.contains(r'\[(paragraph|section|page)', regex=True, na=False)]
    # Remove empty lines
    df = df[~df.iloc[:, 0].isna()]
    # Remove page numbers
    df = df[~((df['Korrekt tekst'].str.len() < 10) & (df['Korrekt tekst'].str.contains(r'.*\d+.*', regex=True)))]
    # Aggregate texts by joining lines
    df = df.groupby(['Dokumentnavn (år_forfatter_titel)', 'Sidetal i PDF'])\
           .agg({'Korrekt tekst': lambda x: '\n'.join(x)})\
           .reset_index()
    # Keep only the first page from each novel: Group by novel, aggregate by getting first row.
    df = df.groupby(['Dokumentnavn (år_forfatter_titel)'])\
           .agg({'Korrekt tekst': lambda x: x.iloc[0]})\
           .reset_index()
    datadict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    datadict = {k.removesuffix('.pdf'): v for k, v in datadict.items()}
    return datadict


def print_align_lines(ocr_lines, gold_lines):
    aligned_lines = list(zip(ocr_lines, gold_lines))
    print('{:<70}{}'.format('OCR', 'Gold'))
    for pair in aligned_lines:
        print('{:<70}{}'.format(pair[0], pair[1]))
    print()
    print()


def align_lines(ocr_data: dict, gold_data: dict):
    # We assume that ocr_data and gold_data have the same novels. Otherwise just fix it manually.
    for novel in ocr_data:
        print('Novel:', novel)
        ocr_text, gold_text = ocr_data[novel], gold_data[novel]
        ocr_lines, gold_lines = recursive_align(ocr_text, gold_text, 0)
        print_align_lines(ocr_lines, gold_lines)


def inv_lev_ratio(s1, s2):
    """Levenshtein.ratio inverted into a kind of distance"""
    return 1 - lev.ratio(s1, s2)


def recursive_align(ocr_data, gold_data, i=0):
    """
    Recursively refine alignment of lines in ocr_data and gold_data.
    Return the best alignment out of a few tries.
    Stopping criteria:
    - Absolute criterion: If avg. relative levenshtein is low, it must be a good match.
    - Otherwise: Lowest avg. relative levenshtein in 5 iterations
    """
    ocr_lines, gold_lines = ocr_data.splitlines(), gold_data.splitlines()
    ocr_gold_pairs = list(zip(ocr_lines, gold_lines))
    n_lines = len(ocr_gold_pairs)
    relative_linedists = [inv_lev_ratio(*pair) for pair in ocr_gold_pairs]
    avg_dist = statistics.mean(relative_linedists)
    if avg_dist < .1:
        # If the average distance is very low, just return the alignment as is
        return ocr_lines[:n_lines], gold_lines[:n_lines]
    elif i == 5:
        # After 5 iterations, return the best alignment at this point
        return ocr_lines[:n_lines], gold_lines[:n_lines]
    else:
        # Else, recurse with the best shifting of ocr vs. gold
        print('Iterating ...')
        best_shift = get_best_shift(ocr_lines, gold_lines, relative_linedists)
        if best_shift['avg_dist'] < avg_dist:
            return recursive_align('\n'.join(best_shift['ocr']), '\n'.join(best_shift['gold']), i=i+1)
        else:
            return ocr_lines[:n_lines], gold_lines[:n_lines]


def get_best_shift(_ocr_lines, _gold_lines, distances, n=10):
    """
    Shift alignment one step at various points (i.e. at n biggest differences),
    both in one direction and the other. Return the best one.
    """
    # Get indexes of n biggest differences - and always index 0.
    n = len(distances) if len(distances) < n else n
    shift_indexes = sorted(range(len(distances)), key=lambda x: distances[x], reverse=True)[:n]
    shift_indexes = [0] + shift_indexes if 0 not in shift_indexes else shift_indexes

    shifts = []
    for idx in shift_indexes:
        # Shift the OCR side by inserting an easily recognizable dummy line 'XX'.
        shifted_ocr = _ocr_lines[:idx] + ['XX'] + _ocr_lines[idx:]
        rel_dists_ocrshift = [inv_lev_ratio(*pair) for pair in zip(shifted_ocr, _gold_lines)]
        avg_dists_ocr = statistics.mean(rel_dists_ocrshift)
        shifts.append({'index': idx, 'ocr': shifted_ocr, 'gold': _gold_lines,
                       'dists': rel_dists_ocrshift, 'avg_dist': avg_dists_ocr})
        # Shift the Gold side by inserting an easily recognizable dummy line 'XX'.
        shifted_gold = _gold_lines[:idx] + ['XX'] + _gold_lines[idx:]
        rel_dists_goldshift = [inv_lev_ratio(*pair) for pair in zip(_ocr_lines, shifted_gold)]
        avg_dists_gold = statistics.mean(rel_dists_goldshift)
        shifts.append({'index': idx, 'ocr': _ocr_lines, 'gold': shifted_gold,
                       'dists': rel_dists_goldshift, 'avg_dist': avg_dists_gold})
    avg_dists = [dct['avg_dist'] for dct in shifts]
    best_shift_idx = avg_dists.index(min(avg_dists))
    return shifts[best_shift_idx]


def make_eval_df(ocr_data: dict, gold_data: dict, use_cache=True):
    for novel in ocr_data:
        print()
        print('Novel:', novel)
        print('OCR:', ocr_data[novel])
        print()
        print('Gold:', gold_data[novel])
        print()
        #alignment = align_ocr(ocr_data[novel], gold_data[novel])



def make_eval_df2(evaldata, use_cache=True):
    """Create a pandas dataframe with original and correct token data."""
    def create_df_from_scratch(_evaldata):
        """Align original and gold data and make a dataset."""
        _df = pd.DataFrame()
        """
        for linedict in _evaldata:
            alignment = align_ocr(linedict['orig'], linedict['corr'])
            linedf = pd.DataFrame(zip(alignment.aligned_orig, alignment.correct,
                                      alignment.types, alignment.matches,
                                      alignment.lev_dists, alignment.cers,
                                      alignment.ratios),
                                  columns='aligned_orig correct type match lev_dist cer ratio'.split())
            linedf['novel'] = linedict['novel'].replace('.extr.txt', '')
            linedf['orig_line'] = linedict['orig']
            linedf['corr_line'] = linedict['corr']
            _df = _df.append(linedf)
        """
        print(list(_df))
        _df.to_csv('new_eval_df.csv', index=False, quoting=2)
        return _df

    if use_cache:
        try:
            df = pd.read_csv('eval_df.csv')
            print('ATTENTION: Using saved dataset from eval_df.csv')
        except FileNotFoundError:
            df = create_df_from_scratch(evaldata)
    else:
        df = create_df_from_scratch(evaldata)
    return df


def make_freq_breakdown(df, col):
    """Show a frequency breakdown of the values in a dataframe column."""
    counts = df[[col]].value_counts()
    percs = df[[col]].value_counts(normalize=True).multiply(100).round(2)
    return pd.DataFrame({'count': counts, 'percentage': percs}).reset_index()


def tabulate_large_lev_cases(df, n=2, m=3):
    """Tabulate unique orig-corr pairs that contain a levenshtein distance greater than n."""
    cases = df.loc[(df.lev_dist >= n) & (df.lev_dist <= m)][['orig_line', 'corr_line']]
    cases['pair'] = '("' + cases[['orig_line', 'corr_line']].agg('", "'.join, axis=1) + '"),'
    return cases.groupby('pair')[['orig_line']].agg(lambda x: len(x)).reset_index().sort_values(by='orig_line')


def print_align_examples(evaldata, n=10, ratio=.9):
    """Get n examples of alignments."""
    i = 0
    for linedict in evaldata:
        alignment = align_ocr(linedict['orig'], linedict['corr'])
        if alignment.avg_ratio < ratio:
            zipped = zip(alignment.aligned_orig, alignment.correct, alignment.types)
            maxlengths = [max(len(x), len(y), len(z)) for x, y, z in zipped]
            print('    '.join([f"{x + ' ' * (y - len(x))}" for x, y in zip(alignment.aligned_orig, maxlengths)]))
            print('    '.join([f"{x + ' ' * (y - len(x))}" for x, y in zip(alignment.correct, maxlengths)]))
            print('    '.join([f"{x + ' ' * (y - len(x))}" for x, y in zip(alignment.types, maxlengths)]))
            print()
            i += 1
        if i == n:
            break


def make_opcode_breakdown(df, n=3):
    """Return frequency statistics on concrete replacements, deletions etc. in tokens with lev <= n"""
    def get_op_str(a, b, lev, _n):
        """Return a single string summarizing which operations will transform a into b."""
        # Make generalized xxx patterns out of word pairs that are equal except for spaces (underscores).
        if '_' in a and re.sub('_', '', a) == b:
            a = re.sub(r'[^_]', 'x', a)
            b = re.sub(r'\w', 'X', b)
        s = SequenceMatcher(None, a, b)
        opcode_list = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'equal':
                pass
            # elif lev <= _n:
            #     opcode_list.append(f"'{a[i1:i2]}' --> '{b[j1:j2]}'")
            # else:
            #     opcode_list.append('other')
            else:
                opcode_list.append(f"'{a[i1:i2]}' --> '{b[j1:j2]}'")
        return ', '.join(opcode_list)

    cases = df[['aligned_orig', 'correct', 'lev_dist']].fillna('')
    cases['ops'] = cases[['aligned_orig', 'correct', 'lev_dist']].agg(lambda x: get_op_str(*x, n), axis=1)
    return make_freq_breakdown(cases, 'ops')


if __name__ == '__main__':
    main()
