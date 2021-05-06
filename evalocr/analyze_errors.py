"""
analyze_errors.py
Functions for analyzing a dataset of OCR tokens aligned with corresponding Gold tokens.
"""
import os
import re
from difflib import SequenceMatcher
import pandas as pd
from memoocr.align_ocr import align_ocr

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
            zipped = zip(alignment.aligned_orig, alignment.correct, alignment.matchtypes)
            maxlengths = [max(len(x), len(y), len(z)) for x, y, z in zipped]
            print('    '.join([f"{x + ' ' * (y - len(x))}" for x, y in zip(alignment.aligned_orig, maxlengths)]))
            print('    '.join([f"{x + ' ' * (y - len(x))}" for x, y in zip(alignment.correct, maxlengths)]))
            print('    '.join([f"{x + ' ' * (y - len(x))}" for x, y in zip(alignment.matchtypes, maxlengths)]))
            print()
            i += 1
        if i == n:
            break


def make_opcode_breakdown(df):
    """Return frequency statistics on concrete replacements, deletions etc. in tokens with lev <= n"""
    cases = df[['aligned_orig', 'correct', 'lev_dist']].fillna('')
    cases['ops'] = cases[['aligned_orig', 'correct', 'lev_dist']].agg(lambda x: get_op_str(*x), axis=1)
    return make_freq_breakdown(cases, 'ops')


def get_op_str(a, b):
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
        else:
            opcode_list.append(f"{a[i1:i2] if a[i1:i2] else '•'}={b[j1:j2] if b[j1:j2] else '•'}")
    return '+'.join(opcode_list)


def make_stats(eval_df, conf, filename):
    outfolder = os.path.join(conf['intermediatedir'], 'analyses')
    try:
        os.makedirs(outfolder)
    except FileExistsError:
        pass
    outpath = os.path.join(outfolder, filename)
    print(outpath)
    with open(outpath, 'w') as f:

        f.write('ERROR STATISTICS BY TYPE' + "\n")
        f.write(make_freq_breakdown(eval_df, 'matchtype').to_string() + "\n" + "\n")

        f.write('ERROR STATISTICS BY OPERATION')
        f.write(make_opcode_breakdown(eval_df, n=3).to_string() + "\n" + "\n")

        f.write('ERROR STATISTICS BY MATCH' + "\n")
        f.write(make_freq_breakdown(eval_df, 'match').to_string() + "\n" + "\n")

        f.write('ERROR STATISTICS BY LEVENSHTEIN DISTANCE' + "\n")
        f.write(make_freq_breakdown(eval_df, 'lev_dist').to_string() + "\n" + "\n")

        f.write('LEVENSHTEIN > 3' + "\n")
        large_lev_dist = eval_df.loc[eval_df.lev_dist > 3][['aligned_orig', 'correct', 'lev_dist', 'matchtype', 'orig_line']]
        f.write(large_lev_dist.to_string() + "\n" + "\n")

        f.write('SAME-CHAR ERRORS' + "\n")
        same_char_data = eval_df.loc[eval_df['matchtype'] == 'same_chars'][['aligned_orig', 'correct', 'orig_line', 'corr_line']]
        f.write(same_char_data.to_string() + "\n" + "\n")

        f.write('SAME-CHAR ERRORS AGGREGATED' + "\n")
        same_char_agg = eval_df.loc[eval_df['matchtype'] == 'same_chars'] \
            .groupby('correct') \
            .agg({'correct': 'count', 'aligned_orig': lambda x: str(set(x))})
        f.write(same_char_agg.to_string() + "\n" + "\n")

        f.write('CORRECT ONE-CHAR TOKENS' + "\n")
        one_char_correct = eval_df.loc[eval_df['correct'].str.len() == 1] \
            .groupby('correct') \
            .agg({'correct': 'count'})
        f.write(one_char_correct.to_string() + "\n" + "\n")
        correct_one_chars = eval_df.loc[eval_df['correct'].isin(list('aioIO'))] \
            .groupby('correct') \
            .agg({'correct': 'count', 'aligned_orig': lambda x: str(set(x))})
        f.write(correct_one_chars.to_string() + "\n")
        correct_orig_one_chars = eval_df.loc[eval_df['aligned_orig'].isin(list('aioIO'))] \
            .groupby('aligned_orig') \
            .agg({'aligned_orig': 'count', 'correct': lambda x: str(set(x))})
        f.write(correct_orig_one_chars.to_string() + "\n")

        # Create data for unit testing
        # large_lev_breakdown = tabulate_large_lev_cases(eval_df, n=1, m=2)
        # print(large_lev_breakdown)
        # print('\n'.join(large_lev_breakdown['pair'].to_list()))
        # print(large_lev_breakdown.shape)

