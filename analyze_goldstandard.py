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
import re
import pandas as pd

pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False

from evalocr.align_ocr import align_ocr


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


def make_eval_df(evaldata):
    """Create a pandas dataframe with token data."""
    try:
        df = pd.read_csv('eval_df.csv')
        print('ATTENTION: Using saved dataset from eval_df.csv')
        return df
    except FileNotFoundError:
        df = pd.DataFrame()
        for linedict in evaldata:
            alignment = align_ocr(linedict['orig'], linedict['corr'])
            linedf = pd.DataFrame(zip(alignment.aligned_orig, alignment.correct,
                                      alignment.types, alignment.matches,
                                      alignment.lev_dists, alignment.cers,
                                      alignment.ratios),
                                  columns='aligned_orig correct type match lev_dist cer ratio'.split())
            linedf['novel'] = linedict['novel'].replace('.extr.txt', '')
            linedf['orig_line'] = linedict['orig']
            linedf['corr_line'] = linedict['corr']
            df = df.append(linedf)
        print(list(df))
        df.to_csv('eval_df.csv', index=False, quoting=2)
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
            print('    '.join([f"{x + ' '*(y - len(x))}" for x, y in zip(alignment.aligned_orig, maxlengths)]))
            print('    '.join([f"{x + ' '*(y - len(x))}" for x, y in zip(alignment.correct, maxlengths)]))
            print('    '.join([f"{x + ' '*(y - len(x))}" for x, y in zip(alignment.types, maxlengths)]))
            print()
            i += 1
        if i == n:
            break


def main():
    """Run the OCR pipeline."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    evaldata = get_evaldata(config, None)
    print('ALIGNMENT EXAMPLES')
    print_align_examples(evaldata, ratio=.99)
    eval_df = make_eval_df(evaldata)
    type_breakdown = make_freq_breakdown(eval_df, 'type')
    match_breakdown = make_freq_breakdown(eval_df, 'match')
    lev_breakdown = make_freq_breakdown(eval_df, 'lev_dist')
    print('ERROR STATISTICS BY TYPE')
    print(type_breakdown)
    print()
    print('ERROR STATISTICS BY MATCH')
    print(match_breakdown)
    print()
    print('ERROR STATISTICS BY LEVENSHTEIN DISTANCE')
    print(lev_breakdown)
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
    same_char_agg = eval_df.loc[eval_df['type'] == 'same_chars']\
                           .groupby('correct')\
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


if __name__ == '__main__':
    main()
