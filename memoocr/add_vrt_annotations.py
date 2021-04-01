"""
add_vrt_annotations.py
Add annotation layer(s) to VRT file based on token, line, and page alignment, and text id.
"""
import configparser
import os
import re
from datetime import datetime
from itertools import groupby
from nltk import word_tokenize
from evalocr.align_ocr import recursive_token_align
from evalocr.analyze_errors import get_op_str
from Levenshtein import distance as lev_dist
from Levenshtein import ratio as lev_ratio


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))
    conf = config['DEFAULT']

    outdir = os.path.join(conf['intermediatedir'], 'vrt2')
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    outfile = os.path.join(outdir, '1870_Lange_AaenOgHavet.vrt')
    add_ocr_tokens(os.path.join(conf['intermediatedir'], 'vrt', '1870_Lange_AaenOgHavet.vrt'),
                   os.path.join(conf['intermediatedir'], '2-uncorrected', '1870_Lange_AaenOgHavet-s10-11'),
                   outfile)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def add_ocr_tokens(vrt_file, novel_dir, outfile):
    """Align and add tokens from OCR pages to VRT file (and some difference measures)."""
    with open(vrt_file, 'r') as infile:
        lines = infile.read().splitlines()
    tokenlines = [tuple(line.split('\t')) for line in lines if not re.match(r'</?text', line)]
    # Group on line and page to get a tokenlist for each line.
    page_tokenlists = [tuple(grp) for _, grp in groupby(tokenlines, key=lambda tokentup: tokentup[-1])]

    novel_pages = sorted(os.listdir(novel_dir))
    if len(page_tokenlists) != len(novel_pages):
        raise Exception('Number of novel pages and number of VRT pages do not match.')
    with open(outfile, 'w') as out:
        out.write('<text>\n')
        for page, vrt_tokentups in zip(novel_pages, page_tokenlists):
            with open(os.path.join(novel_dir, page), 'r') as pagefile:
                pagetext = pagefile.read()
            page_tokens = tuple(word_tokenize(pagetext, language='danish'))
            vrt_tokens = tuple([tup[0] for tup in vrt_tokentups])
            aligned_page_toks, _ = recursive_token_align(vrt_tokens, page_tokens)
            # Hack to deal with long sequences of tokens due to unequal page samples
            aligned_page_toks = [tok if len(tok) < 100 else tok[:30] + '...' for tok in aligned_page_toks]
            new_vrt_tups = add_annotation_layer(vrt_tokentups, aligned_page_toks)
            new_vrt_tups = add_diff_measures(new_vrt_tups, vrt_tokens, aligned_page_toks)
            out.write('\n'.join(['\t'.join(x) for x in new_vrt_tups]))
        out.write('\n</text>')


def add_diff_measures(vrt_tokentups, vrt_tokens, aligned_page_toks):
    """Add difference measures pertaining to vrt_tokens and aligned_page_toks."""
    lev_dists = [lev_dist(v, a) for v, a in zip(vrt_tokens, aligned_page_toks)]
    lev_ratios = [round(lev_ratio(v, a), 2) for v, a in zip(vrt_tokens, aligned_page_toks)]
    cers = [round(1.0 - ratio, 2) for ratio in lev_ratios]
    types = [get_difftype(v, a) for v, a in zip(vrt_tokens, aligned_page_toks)]
    op_strings = [get_op_str(v, a) for v, a in zip(vrt_tokens, aligned_page_toks)]
    new_vrt_tups = add_annotation_layer(vrt_tokentups, lev_dists)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, lev_ratios)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, cers)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, types)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, op_strings)
    return new_vrt_tups


def add_annotation_layer(tokentups: list, annotations: list):
    """Add a layer of annotations (= new element to each token tuple)."""
    annotations = [str(annot) if annot else 'NA' for annot in annotations]
    return [tup + (ann,) for tup, ann in zip(tokentups, annotations)]


def get_difftype(str1, str2):
    """Determine the difference type of two strings."""
    if '_' in str1 and len(str1) > 1 and str1.replace('_', '') == str2:
        return 'same_chars'
    elif str1 == str2:
        return 'match'
    elif '[-]' not in str1 and '_' not in str2:
        return f'lev_{str(lev_dist(str1, str2))}'
    elif '[-]' not in str1 and '_' in str2:
        return f'split_lev_{str(lev_dist(str1, str2))}'
    else:
        return 'blaha'


if __name__ == '__main__':
    main()
