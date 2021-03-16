"""
analyze_errors.py
Functions for analyzing a dataset of OCR tokens aligned with corresponding Gold tokens.
"""

import re
from difflib import SequenceMatcher
import pandas as pd
from evalocr.align_ocr import align_ocr

pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False


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


def make_stats(eval_df):
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

