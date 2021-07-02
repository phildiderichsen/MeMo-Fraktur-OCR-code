import itertools
import os
import re
import shutil
from difflib import SequenceMatcher
from myutils.fraktur_filenames import frakturfiles

from nltk import word_tokenize
from datetime import datetime


class EvalPaths(object):
    """Class that specifies all relevant paths for the evaluation pipeline based on config.ini."""

    def __init__(self, conf, param_str):
        self.intermediate = os.path.join(conf['intermediatedir'], datetime.now().strftime('%Y-%m-%d'))
        self.ocr_kb_dir = os.path.join(self.intermediate, 'orig_pages')
        self.gold_novels_dir = os.path.join(self.intermediate, 'gold_pages')
        self.vrt_dir = os.path.join(self.intermediate, 'vrt', param_str)
        safe_makedirs(self.vrt_dir)
        self.analyses_dir = os.path.join(self.intermediate, 'analyses')
        safe_makedirs(self.analyses_dir)
        self.corp_label = conf['fraktur_gold_vrt_label']
        self.annotated_outdir = os.path.join(conf['annotated_outdir'], self.corp_label, param_str)
        safe_makedirs(self.annotated_outdir)
        self.img_dir = os.path.join(self.intermediate, '1-imgs')
        self.basic_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.vrt')
        self.annotated_gold_vrt_path = os.path.join(self.annotated_outdir, self.corp_label + '.annotated.vrt')
        self.local_annotated_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.annotated.vrt')


class CorrPaths(object):
    """Class that specifies all relevant paths for the correction pipeline based on config.ini."""

    def __init__(self, conf, param_str):
        self.memo_home = conf['memo_home']
        self.fulloutputdir = conf['fulloutputdir']
        self.noveloutdirs = [os.path.join(self.memo_home, d) for d in conf['novel_dirs'].split()]
        for d in self.noveloutdirs:
            safe_makedirs(d)
        self.frakturpaths = self.make_frakturpaths()
        self.img_dir = os.path.join(self.fulloutputdir, '1-imgs')

        # self.ocr_kb_dir = os.path.join(self.intermediate, 'orig_pages')
        # self.gold_novels_dir = os.path.join(self.intermediate, 'gold_pages')
        # self.vrt_dir = os.path.join(self.intermediate, 'vrt', param_str)
        # safe_makedirs(self.vrt_dir)
        # self.analyses_dir = os.path.join(self.intermediate, 'analyses')
        # safe_makedirs(self.analyses_dir)
        # self.corp_label = conf['fraktur_gold_vrt_label']
        # self.annotated_outdir = os.path.join(conf['annotated_outdir'], self.corp_label, param_str)
        # safe_makedirs(self.annotated_outdir)
        # self.basic_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.vrt')
        # self.annotated_gold_vrt_path = os.path.join(self.annotated_outdir, self.corp_label + '.annotated.vrt')
        # self.local_annotated_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.annotated.vrt')

    def make_frakturpaths(self):
        """Construct full paths to fraktur PDFs."""
        noveldir_contents = [[os.path.join(d, f) for f in os.listdir(d)] for d in self.noveloutdirs]
        novel_pdfs = [path for filelist in noveldir_contents for path in filelist]
        return [path for path in novel_pdfs if os.path.basename(path) in frakturfiles]


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def safe_copytree(dir1, dir2):
    """Copy recursively from dir1 to dir2, pass if dirs do not exist."""
    try:
        # TODO The script for generating pages of corrected gold standard text got lost.
        shutil.copytree(dir1, dir2)
    except FileExistsError:
        pass


def chunk_list(lst, n):
    """Split list into list of lists of n elements each"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


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
    tokenlist = word_tokenize(string, language='danish')
    # TODO This could probably be more appropriately handled somewhere else
    # Repair this one funny tokenization error. Actually, change the error to the same 'error' Tesseract makes.
    tokenlist = [t.replace('``', '“') for t in tokenlist]
    return tokenlist


def fix_hyphens(stringlist: list):
    """Merge hyphenations across strings in stringlist"""
    # Escape any existing pilcrows, however unlikely ..
    stringlist = [s.replace('¶', '___PILCROW___') for s in stringlist]
    joined = '¶'.join(stringlist)
    # \f: form feed, which Tesseract puts at end of every page. \xad: soft hyphen.
    dehyphenated = re.sub(r'(\w+)[\xad⸗—-]+[\n\r\f]*\s*¶\s*(\S+)\s*', r'\1\2¶', joined)
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
                text = ''.join(text_chain)
                text = re.sub(r'\n</corpus>\s*', '', text)
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


def print_and_write(string, outpath):
    """Print output, and also write it to file."""
    print(string)
    with open(outpath, 'a') as outfile:
        outfile.write(string)
        outfile.write('\n')


def get_params(conf):
    """Get all parameters used in the pipeline, plus a string of the parameters (for file names etc.)."""
    param_tuple = (conf['base_ocr'].split('_')[-1], conf['freqs'])
    correasy = 'correasy' if conf.getboolean('correct_easy') else ''
    corrhard = 'corrhard' if conf.getboolean('correct_hard') else ''
    corrocr = 'symwordcorr' if conf.getboolean('sym_wordcorrect') else ''
    param_tuple += (correasy, corrhard, corrocr)
    return param_tuple, '_'.join([x for x in param_tuple if x])


def write_frakturgold_mode(mode_template, gold_vrt_p_attrs, outpath):
    """Write Korp config (mode file) for the Frakturguld mode."""
    p_attrs = gold_vrt_p_attrs.split()
    mode_templ = readfile(mode_template)
    p_attr_templ = '''    {p_attr}: {{
        label : "{label}",
        opts : settings.defaultOptions,
        order : 1,
        stats_stringify: function(values) {{return values.join(" ")}}
        }}'''
    p_attr_confs = [p_attr_templ.format(p_attr=att, label=att.upper()) for att in p_attrs]
    with open(outpath, 'w') as outfile:
        outfile.write(mode_templ.format(p_attrs=',\n'.join(p_attr_confs)))


def write_frakturgold_encodescript(encodescript_templ, annotated_outdir, gold_vrt_p_attrs, outpath):
    """Write CWB encoding script for the Frakturguld mode."""
    p_attrs = gold_vrt_p_attrs.split()[1:]  # Skip first attr since 'word' must not be specified in CWB.
    script_templ = readfile(encodescript_templ)
    p_attr_templ = '-P {p_attr}'
    p_attr_confs = [p_attr_templ.format(p_attr=att) for att in p_attrs]
    with open(outpath, 'w') as outfile:
        pathlist = os.path.normpath(annotated_outdir).split(os.sep)
        novels_dir = os.path.join('$CORPORADIR', *pathlist[pathlist.index('annotated'):])
        outfile.write(script_templ.format(novels_dir=novels_dir, p_attrs=' '.join(p_attr_confs)))
