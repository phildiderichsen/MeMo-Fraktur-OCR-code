"""
Analyze MeMo OCR cleaning

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
import pandas as pd

from memoocr.align_ocr import align_ocr
from evalocr.analyze_errors import make_stats, print_align_examples

pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False


def get_evaldata(config, n=None):
    with open(config['DEFAULT']['evaldata'], 'r', encoding='utf8') as evaldata:
        data = evaldata.read().splitlines()
    # Remove header if present.
    if 'Dokumentnavn' in data[0]:
        data = data[1:n]
    data = [line.split('\t') for line in data]
    datadicts = [dict(zip(['novel', 'orig', 'corr', 'comment'], fields)) for fields in data]
    # Remove empty lines
    datadicts = [dd for dd in datadicts if dd['novel']]
    # Remove '[paragraph]', '[section]', '[section + line]', '[page]' lines.
    rgx = r'\[(paragraph|section|page)'
    datadicts = [dd for dd in datadicts if not re.search(rgx, dd['orig'])]
    return datadicts


def make_eval_df(evaldata, config, use_cache=True):
    """Create a pandas dataframe with token data."""
    eval_df_path = os.path.join(config['DEFAULT']['intermediatedir'], 'eval_datasets')
    eval_df_file = os.path.join(eval_df_path, 'eval_df.csv')
    print('eval_df_file:')
    print(eval_df_file)

    def create_df_from_scratch(_evaldata, _eval_df_path, _eval_df_file):
        _df = pd.DataFrame()
        for linedict in _evaldata:
            alignment = align_ocr(linedict['orig'], linedict['corr'])
            linedf = pd.DataFrame(zip(alignment.aligned_orig, alignment.correct,
                                      alignment.matchtypes, alignment.matches,
                                      alignment.lev_dists, alignment.cers,
                                      alignment.ratios),
                                  columns='aligned_orig correct matchtype match lev_dist cer ratio'.split())
            linedf['novel'] = linedict['novel'].replace('.extr.txt', '')
            linedf['orig_line'] = linedict['orig']
            linedf['corr_line'] = linedict['corr']
            _df = _df.append(linedf)
        print(list(_df))
        os.makedirs(_eval_df_path, exist_ok=True)
        _df.to_csv(_eval_df_file, index=False, quoting=2)
        return _df

    if use_cache:
        try:
            df = pd.read_csv(eval_df_file)
            print('ATTENTION: Using saved dataset from eval_df.csv')
        except FileNotFoundError:
            df = create_df_from_scratch(evaldata, eval_df_path, eval_df_file)
    else:
        df = create_df_from_scratch(evaldata, eval_df_path, eval_df_file)
    return df


def main():
    """Run OCR error analysis."""
    config = configparser.ConfigParser()
    config.read(os.path.join('config', 'config.ini'))
    evaldata = get_evaldata(config, None)
    print('ALIGNMENT EXAMPLES')
    print_align_examples(evaldata, ratio=.99)

    eval_df = make_eval_df(evaldata, config, use_cache=False)
    make_stats(eval_df, config['DEFAULT'], 'analyze_gold.txt')


if __name__ == '__main__':
    main()
