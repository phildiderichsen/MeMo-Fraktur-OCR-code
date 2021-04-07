"""
Utilities to work with VRT files (CWB input files in vertical format, one token per line).
"""
import itertools
import re


def split_vrt(vrt):
    """Generator that yields one <text>..</text> string at a time from a VRT file."""
    with open(vrt) as infile:
        grps = itertools.groupby(infile, key=lambda x: x.startswith("<text"))
        for k, grp in grps:
            if k:
                text_chain = itertools.chain([next(grp)], (next(grps)[1]))
                text = ''.join(text_chain).removesuffix('\n</corpus>')
                yield text


def vrt_text2tokens(vrt_text: str):
    """Take one <text>-element from a VRT file and return its tokens as one line. Note: assumes only <text> elements!"""
    vrt_lines = vrt_text.splitlines()
    token_lines = [line for line in vrt_lines if not re.match(r'</?text', line)]
    return [line.split('\t')[0] for line in token_lines]


