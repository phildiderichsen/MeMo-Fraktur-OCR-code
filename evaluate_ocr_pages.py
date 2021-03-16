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
import statistics
import Levenshtein as lev
import pandas as pd

from evalocr.align_ocr import align_ocr
from evalocr.analyze_errors import make_stats

pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False


def main():
    """Run OCR error analysis."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    ocrdata = get_ocr_data(config)
    golddata = get_gold_data(config)
    evaldata = align_lines(ocrdata, golddata)
    eval_df = make_eval_df(evaldata, use_cache=True)
    make_stats(eval_df, config['test'], 'eval_ocr.txt')


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
    """Return dict from novel to dict with ocr and gold text."""
    # We assume that ocr_data and gold_data have the same novels. Otherwise just fix it manually.
    eval_dict = {}
    for novel in ocr_data:
        print('Novel:', novel)
        ocr_text, gold_text = ocr_data[novel], gold_data[novel]
        # Align lines
        ocr_lines, gold_lines = recursive_align(ocr_text, gold_text)
        eval_dict[novel] = {'ocrlines': ocr_lines, 'goldlines': gold_lines}
        # print_align_lines(ocr_lines, gold_lines)
    return eval_dict


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


def make_eval_df(eval_data: dict, use_cache=True):
    """Create evaluation dataset with each token on its own line. (Or get it from file)."""
    def create_df_from_scratch(evaldata):
        total_df = pd.DataFrame()
        for novel in evaldata:
            aligned_lines = list(zip(evaldata[novel]['ocrlines'], evaldata[novel]['goldlines']))
            novel_df = make_novel_df(novel, aligned_lines)
            total_df = total_df.append(novel_df)
            total_df.to_csv('ocr_eval_df.csv', index=False, quoting=2)
        return total_df

    if use_cache:
        try:
            df = pd.read_csv('ocr_eval_df.csv')
            print('ATTENTION: Using saved dataset from ocr_eval_df.csv')
        except FileNotFoundError:
            df = create_df_from_scratch(eval_data)
    else:
        df = create_df_from_scratch(eval_data)
    return df


def make_novel_df(novel: str, aligned_lines: list):
    """Token-align original and gold data from a novel and make a one-novel dataset."""
    print(novel)
    novel_df = pd.DataFrame()
    for linepair in aligned_lines:
        alignment = align_ocr(*linepair)
        linedf = pd.DataFrame(zip(alignment.aligned_orig, alignment.correct,
                                  alignment.types, alignment.matches,
                                  alignment.lev_dists, alignment.cers,
                                  alignment.ratios),
                              columns='aligned_orig correct type match lev_dist cer ratio'.split())
        linedf['novel'] = novel
        linedf['orig_line'] = linepair[0]
        linedf['corr_line'] = linepair[1]
        novel_df = novel_df.append(linedf)
    return novel_df


if __name__ == '__main__':
    main()
