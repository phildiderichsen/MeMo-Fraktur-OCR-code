import itertools
import os
import re
from difflib import SequenceMatcher

from nltk import word_tokenize


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """

    def atoi(_text):
        return int(_text) if _text.isdigit() else _text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sorted_listdir(directory):
    """Version of listdir where the files are sorted in human order."""
    dirlist = os.listdir(directory)
    dirlist.sort(key=natural_keys)
    return dirlist


def tokenize(string):
    """Tokenize string with Danish NLTK tokenizer - also, split all punctuation."""
    # Pad punctuation with whitespace
    string = re.sub(r'([.,:;„"»«\'!?()])', r' \1 ', string)
    return word_tokenize(string, language='danish')


def fix_hyphens(stringlist: list):
    """Merge hyphenations across strings in stringlist"""
    # Escape any existing pilcrows, however unlikely ..
    stringlist = [s.replace('¶', '___PILCROW___') for s in stringlist]
    joined = '¶'.join(stringlist)
    # \f: form feed, which Tesseract puts at end of every page.
    dehyphenated = re.sub(r'(\w+)[⸗—-]+[\n\r\f]*¶\s*(\S+)\s*', r'\1\2¶', joined)
    new_stringlist = dehyphenated.split('¶')
    # Put back original pilcrows ..
    new_stringlist = [s.replace('___PILCROW___', '¶') for s in new_stringlist]
    return new_stringlist


def readfile(filename):
    with open(filename, 'r') as f:
        return f.read()


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


def get_op_str(a: str, b: str):
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