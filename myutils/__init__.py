import configparser
import csv
import itertools
import os
import pathlib
import re
import shutil
import sys
from difflib import SequenceMatcher
from memoocr import ROOT_PATH

from nltk import word_tokenize
from datetime import datetime


class Confs(object):
    """Class that makes the various configuration sections available from myutils."""

    def __init__(self, conf):
        self.evalconf = conf['eval']
        self.corrconf = conf['correct']
        self.tessconf = conf['tesseracttest']


class EvalPaths(object):
    """Class that specifies all relevant paths for the evaluation pipeline based on config.ini."""

    def __init__(self, conf, param_str):
        self.intermediate = os.path.join(conf['intermediatedir'], datetime.now().strftime('%Y-%m-%d'))
        self.files_to_process = readfile(conf['files_to_process']).splitlines()
        self.ocr_kb_dir = os.path.join(self.intermediate, 'orig_pages')
        self.gold_novels_dir = os.path.join(self.intermediate, 'gold_pages')

        self.singleline_dir = os.path.join(self.intermediate, 'singleline', param_str)
        safe_makedirs(self.singleline_dir)

        self.vrt_dir = os.path.join(self.intermediate, 'vrt', param_str)
        safe_makedirs(self.vrt_dir)

        self.analyses_dir = os.path.join(self.intermediate, 'analyses')
        safe_makedirs(self.analyses_dir)

        self.corp_label = conf['fraktur_gold_vrt_label']
        self.annotated_outdir = os.path.join(self.intermediate, 'annotated', self.corp_label, param_str)
        safe_makedirs(self.annotated_outdir)

        self.pdf_paths = [os.path.join(conf['inputdir'], f) for f in sorted_listdir(conf['inputdir']) if f.endswith('.pdf')]
        self.img_dir = os.path.join(self.intermediate, '1-imgs')
        self.basic_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.vrt')
        self.annotated_gold_vrt_path = os.path.join(self.annotated_outdir, self.corp_label + '.annotated.vrt')
        self.local_annotated_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.annotated.vrt')


class CorrPaths(object):
    """Class that specifies all relevant paths for the correction pipeline based on config.ini."""

    def __init__(self, conf):
        self.fulloutputdir = conf['fulloutputdir']
        self.base_ocr_dir = os.path.join(self.fulloutputdir, conf['base_ocr'])
        self.pdf_dir = conf['pdf_dir']
        self.ocr_kb_dir = os.path.join(self.fulloutputdir, 'orig_pages')
        self.files_to_process = readfile(conf['files_to_process']).splitlines()
        self.frakturpaths = []
        self.img_dir = conf['img_dir']

        self.singleline_dir = os.path.join(self.fulloutputdir, 'singleline')
        safe_makedirs(self.singleline_dir)

        self.vrt_dir = os.path.join(self.fulloutputdir, 'vrt')
        safe_makedirs(self.vrt_dir)

        self.corp_label = conf['fraktur_vrt_label']
        self.basic_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.vrt')
        self.local_annotated_gold_vrt_path = os.path.join(self.vrt_dir, self.corp_label + '.annotated.vrt')


class TessPaths(object):
    """Paths for the tesseract test mode."""

    def __init__(self, conf):
        self.imgdir = conf['imgdir']
        self.outdir = conf['outdir']


PAGEBREAK = '___PAGEBREAK___'


def get_config():
    """Get configuration parameters (paths, pipeline steps ...)."""
    config = configparser.ConfigParser()
    config.read(os.path.join(pathlib.Path(__file__).parent.parent, 'config', 'config.ini'))
    return config


def make_metadata_dict(pth):
    """Return a dict from file-IDs (corresponding to filenames in the current batch) to metadata for that file."""
    file_basenames = [f.replace('.pdf', '') for f in pth.files_to_process]
    metadata_path = os.path.join(ROOT_PATH, 'metadata.tsv')
    with open(metadata_path, newline='') as f:
        metadatareader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        metadatadicts = [row for row in metadatareader]
    metadict = dict()
    for f in file_basenames:
        clean_basename = f.replace('-FRAKTUR', '')
        for d in metadatadicts:
            try:
                if clean_basename in d['file_id'] or clean_basename in d['filename']:
                    # Make sanitized start (and end ..?) pages.
                    d['realstart'] = d['novelstart_rescan'] if d['novelstart_rescan'] else d['novel_start']
                    if not d['novel_end']:
                        sys.exit(f'No end page recorded for {d["filename"]}')
                    metadict[f] = d
            except TypeError:
                pass

    return metadict


def get_fraktur_metadata():
    """Return rows of metadata from metadata.tsv (copied from romankorpus_metadata_onedrive.xlsx)"""
    metadata_path = os.path.join(ROOT_PATH, 'metadata.tsv')
    with open(metadata_path, newline='') as f:
        metadatarows = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        frakturrows = [row for row in metadatarows
                       if row['typeface (roman or gothic)'] and row['filename']]
    return frakturrows


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def overwritedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)


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
    # Open BOM-tolerant UTF8.
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return f.read()


def flatten(seq):
    """Flatten a list or tuple to a flat list."""
    return [x for sublist in seq for x in sublist]


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
    # TODO Set up this order in the mode file:
    """TEXT ATTRIBUTES	Ny rækkefølge	Ny betegnelse
Tidtil: 235959	-18	
Tidfra: 000000	-17	
Datotil: 1881	-16	
Datofra: 1881	-15	
Forfatter: Kristian Gløersen	1	
Titel: Fra mit Friluftsliv	2	
Undertitel: Skildringer	3	
Dato: 1881	4	
Køn: m	5	
Nationalitet: no	6	
Pseudonym: [empty]	7	
Udgiver: Reitzel	8	Forlag
Skriftsnit: roman	9	Typografi
Pris: 1,6	10	
Sider: 133	11	
Illustrationer: n	12	
Kilde: KB	13	
Sætning nr.: 321	14	
"""
    p_attrs = gold_vrt_p_attrs.split()
    mode_templ = readfile(mode_template)
    p_attr_templ = '''    {p_attr}: {{
        label : "{label}",
        opts : settings.defaultOptions,
        order : 1
        }}'''
    p_attr_confs = [p_attr_templ.format(p_attr=att, label=att.upper()) for att in p_attrs]
    safe_makedirs(os.path.dirname(outpath))
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


def write_frakturcorr_encodescript(encodescript_templ, corr_vrt_p_attrs, outpath):
    """Write CWB encoding script for the Frakturguld mode."""
    p_attrs = corr_vrt_p_attrs.split()[1:]  # Skip first attr since 'word' must not be specified in CWB.
    script_templ = readfile(encodescript_templ)
    p_attr_templ = '-P {p_attr}'
    p_attr_confs = [p_attr_templ.format(p_attr=att) for att in p_attrs]
    safe_makedirs(os.path.dirname(outpath))
    with open(outpath, 'w') as outfile:
        outfile.write(script_templ.format(p_attrs=' '.join(p_attr_confs)))


def get_most_frequent(conf, n):
    """
    Return a set of the n most frequent words on the frequency list used.
    Note! This assumes that the frequency list is reverse sorted by frequency -
        except for all manually identified names which are at the very top of the list.
    """
    return set([line.split()[0] for line in readfile(conf[conf['freqs']]).splitlines()[:n]])


def get_freqlist_forms(conf):
    """Return a set of all forms on the frequency list used."""
    return set([line.split()[0] for line in readfile(conf[conf['freqs']]).splitlines()])


most_frequent = get_most_frequent(get_config()['DEFAULT'], 600)
freqlist_forms = get_freqlist_forms(get_config()['DEFAULT'])


def clean_datadirs(root_dir):
    """Make sure no .DS_Store files are in the way on Mac, and remove '_singlelines' suffix ..."""
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            # make sure what you want to keep isn't in the full filename
            if '.DS_Store' in name:
                os.remove(os.path.join(root, name))
        for dirname in dirs:
            if '_singlepages' in dirname:
                new_dirname = dirname.replace('_singlepages', '')
                os.rename(os.path.join(root, dirname), os.path.join(root, new_dirname))


def precheck_datadirs(pth):
    """Check:
        - Each pdf in files-to-process has a directory with a matching name.
        - Each directory has a pdf with a matching name.
    """
    pdf_dir = pth.pdf_dir
    singlelinedir = pth.base_ocr_dir
    pdf_files = sorted_listdir(pdf_dir)
    singlelinedirs = sorted_listdir(singlelinedir)
    problems = []
    for filename in pdf_files:
        if filename.replace('.pdf', '') not in singlelinedirs:
            problems.append(f'Filename {filename} does not have a matching directory in {singlelinedir}')
    for dirname in singlelinedirs:
        if f'{dirname}.pdf' not in pdf_files:
            problems.append(f'Dirname {dirname} does not have a matching file in {pdf_dir}')
    if problems:
        print('Precheck not passed. Problems:')
        print('\n'.join(problems))
        sys.exit()


def remove_kb_frontmatter(uncorrected_dir, metadata):
    """Hack to remove front and back(?) matter from KB singlefiles.
    Returns a new folder where front matter pages are removed."""
    # Make new orig_pages
    uncorrected_only_novel_pages_dir = uncorrected_dir.replace('orig_pages', 'orig_only_novel_pages')
    overwritedirs(uncorrected_only_novel_pages_dir)
    # Copy pages over only if they are after start and before end
    for folder in sorted_listdir(uncorrected_dir):
        destdir = os.path.join(uncorrected_only_novel_pages_dir, folder)
        overwritedirs(destdir)
        novel_start = int(metadata[folder]['realstart'])
        novel_end = int(metadata[folder]['novel_end'])
        if not novel_end:
            sys.exit(f'Removefrontmatter error: Novel end not recorded for {folder}')
        for i, pagefile in enumerate(sorted(sorted_listdir(os.path.join(uncorrected_dir, folder)))):
            sourcefile = os.path.join(uncorrected_dir, folder, pagefile)
            if novel_start <= i + 1 <= novel_end:
                shutil.copy(sourcefile, destdir)
    # Set uncorrected_dir to the new orig_pages
    return uncorrected_only_novel_pages_dir
