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
from Levenshtein import distance

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


def blaha():
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


def align_lines(ocr_data: dict, gold_data: dict):
    # We assume that ocr_data and gold_data have the same novels. Otherwise just fix it manually.
    for novel in ocr_data:
        print('Novel:', novel)
        ocr_text, gold_text = ocr_data[novel], gold_data[novel]
        ocr_lines, gold_lines = recursive_align(ocr_text, gold_text, 0, [], [], [])


def recursive_align(ocr_data, gold_data, i=0, avgs=[], stdevs=[], pairs_list=[]):
    """
    Recursively refine alignment of lines in ocr_data and gold_data.
    Return the best alignment out of a few tries.
    Stopping criteria:
    - (Almost) all relative Levenshtein line differences < .5
    - No relative Levenshteins greater than 4
    - Lowest average and stdev of relative Lev in 5 iterations
    """
    def get_pair_stats(pairs):
        _rel_linedists = [distance(*pair) / (max(len(pair[0]), len(pair[0]))+1) for pair in pairs]
        _avg_dist = statistics.mean(_rel_linedists)
        _stdev_dist = statistics.stdev(_rel_linedists)
        return _rel_linedists, _avg_dist, _stdev_dist

    ocr_lines, gold_lines = ocr_data.splitlines(), gold_data.splitlines()
    ocr_gold_pairs = list(zip(ocr_lines, gold_lines))
    n_lines = len(ocr_gold_pairs)
    pairs_list.append(ocr_gold_pairs)
    relative_linedists, avg_dist, stdev_dist = get_pair_stats(ocr_gold_pairs)
    print('Relative_linedists:', [round(x, 2) for x in relative_linedists])
    print('avg_dist', avg_dist)
    print('stdev_dist', stdev_dist)
    avgs.append(avg_dist)
    print('avgs', avgs)
    stdevs.append(stdev_dist)
    print('stdevs', stdevs)
    # Stopping criteria
    low_thresh, big_thresh = .5, 4
    levs_are_low = statistics.mean([d < low_thresh for d in relative_linedists]) > .9
    print('levs_are_low', levs_are_low)
    no_big_levs = not any([d > big_thresh for d in relative_linedists])
    print('no_big_levs', no_big_levs)
    is_lowest_avg = avg_dist == min(avgs)
    print('is_lowest_avg', is_lowest_avg)
    is_lowest_stdev = stdev_dist == min(stdevs)
    print('is_lowest_stdev', is_lowest_stdev)

    if all([levs_are_low, no_big_levs, is_lowest_avg, is_lowest_stdev, i <= 5]):
        print('All good ..')
        print()
        return ocr_lines[:n_lines], gold_lines[:n_lines]  #  '\n'.join(ocr_lines[:n_lines]), '\n'.join(gold_lines[:n_lines])
    elif i == 5:
        print('i == 5, Breaking out!')
        print()
        return [''], ['']  # Return the best alignment at this point
    else:
        print('Iterating ...')
        # Find first lev > low_thresh: The place to attempt a shift. TODO: Handle if there is no True val ..
        first_large_lev = [d > low_thresh for d in relative_linedists].index(True)
        # Try shifting in one direction, then the other. Check where avg and stdev improves the most.
        shifted_ocr = ocr_lines[:first_large_lev] + [''] + ocr_lines[first_large_lev:]
        shifted_gold = gold_lines[:first_large_lev] + [''] + gold_lines[first_large_lev:]
        rel_dists_ocrshift, avg_d_ocrshift, stdev_d_ocrshift = get_pair_stats(zip(shifted_ocr, gold_lines))
        rel_dists_goldshift, avg_d_goldshift, stdev_d_goldshift = get_pair_stats(zip(ocr_lines, shifted_gold))
        # Hack measure: shifted avg as percentage of nonshifted + shifted stdev as percentage of nonshifted.
        # Smaller is better.
        ocrshift_improvement = (avg_d_ocrshift / avg_dist) + (stdev_d_ocrshift / stdev_dist)
        goldshift_improvement = (avg_d_goldshift / avg_dist) + (stdev_d_goldshift / stdev_dist)
        if ocrshift_improvement < 1 and ocrshift_improvement < goldshift_improvement:
            print('OCRshift improvement ...')
            return recursive_align('\n'.join(shifted_ocr), gold_data, i=i+1, avgs=avgs, stdevs=stdevs, pairs_list=pairs_list)
        elif goldshift_improvement < 1 and goldshift_improvement < ocrshift_improvement:
            print('Goldshift improvement ...')
            return recursive_align(ocr_data, '\n'.join(shifted_gold), i=i+1, avgs=avgs, stdevs=stdevs, pairs_list=pairs_list)
        else:
            print('No improvement either way ...')
            print()
            aligned_lines = list(zip(ocr_lines[:n_lines], gold_lines[:n_lines]))
            print('{:<70}{}'.format('OCR', 'Gold'))
            for pair in aligned_lines:
                print('{:<70}{}'.format(pair[0], pair[1]))
            print()
            print()
            return [''], ['']


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
