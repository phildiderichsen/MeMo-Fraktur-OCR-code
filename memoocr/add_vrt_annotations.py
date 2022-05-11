"""
add_vrt_annotations.py
Add annotation layer(s) to VRT file based on token, line, and page alignment, and text id.
"""
import configparser
import os
import re
from datetime import datetime
from itertools import groupby, chain
from memoocr.align_ocr import align_b_to_a, align_conll_tuples
from Levenshtein import distance as lev_dist
from Levenshtein import ratio as lev_ratio
from myutils import sorted_listdir, tokenize, fix_hyphens, readfile, get_op_str


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))
    conf = config['DEFAULT']

    vrttext = """<text id="1870_Lange_AaenOgHavet">
en	1	1	22
Uge	2	1	22
efter	3	1	22
dette	4	1	22
sit	5	1	22
første	6	1	22
Besøg	7	1	22
var	8	1	22
han	9	1	22
forlovet	10	1	22
og	1	2	22
allerede	2	2	22
en	3	2	22
Maaned	4	2	22
efter	5	2	22
gift	6	2	22
med	7	2	22
den	8	2	22
skjønne	9	2	22
,	10	2	22
kun	1	3	22
syttenaarige	2	3	22
Ida	3	3	22
Krabbe	4	3	22
.	5	3	22
Søviggaard	1	4	22
*	2	4	22
)	3	4	22
har	4	4	22
en	5	4	22
overmaade	6	4	22
deilig	7	4	22
Beliggenhed	8	4	22
i	1	5	22
den	2	5	22
sydostlige	3	5	22
Del	4	5	22
af	5	5	22
Vendsyssel	6	5	22
,	7	5	22
ved	8	5	22
Kattegattet	1	6	22
.	2	6	22
Skjønne	3	6	22
,	4	6	22
med	5	6	22
Lyng	6	6	22
og	7	6	22
Skov	8	6	22
bevoxede	9	6	22
Bakker	1	7	22
omgive	2	7	22
Gaarden	3	7	22
mod	4	7	22
Nord	5	7	22
,	6	7	22
Syd	7	7	22
og	8	7	22
Vest	9	7	22
;	10	7	22
østenfor	1	8	22
den	2	8	22
strække	3	8	22
sig	4	8	22
derimod	5	8	22
græsrige	6	8	22
Enge	7	8	22
helt	8	8	22
ud	1	9	22
til	2	9	22
Havet	3	9	22
.	4	9	22
Forbi	5	9	22
Gaarden	6	9	22
og	7	9	22
gjennem	8	9	22
disse	9	9	22
Enge	1	10	22
ud	2	10	22
imod	3	10	22
Kattegattet	4	10	22
strømmer	5	10	22
en	6	10	22
Aa	7	10	22
,	8	10	22
hvis	9	10	22
Bredder	1	11	22
,	2	11	22
paa	3	11	22
Herresædets	4	11	22
Enemærker	5	11	22
,	6	11	22
ere	7	11	22
tæt	8	11	22
bevoxede	9	11	22
med	1	12	22
høje	2	12	22
Træer	3	12	22
,	4	12	22
hvilke	5	12	22
om	6	12	22
Somren	7	12	22
paa	8	12	22
mange	9	12	22
Steder	1	13	22
danne	2	13	22
et	3	13	22
saa	4	13	22
tæt	5	13	22
Løvtag	6	13	22
over	7	13	22
Aaen	8	13	22
,	9	13	22
at	10	13	22
Solens	11	13	22
Straaler	1	14	22
ikke	2	14	22
kunne	3	14	22
trænge	4	14	22
derigjennem	5	14	22
.	6	14	22
Naar	7	14	22
man	8	14	22
i	1	15	22
nogen	2	15	22
Tid	3	15	22
har	4	15	22
ladet	5	15	22
sig	6	15	22
glide	7	15	22
i	8	15	22
en	9	15	22
Baad	10	15	22
ned	11	15	22
med	12	15	22
Strømmen	1	16	22
mellem	2	16	22
disse	3	16	22
Træer	4	16	22
,	5	16	22
der	6	16	22
staae	7	16	22
som	8	16	22
en	9	16	22
Mur	1	17	22
til	2	17	22
begge	3	17	22
Sider	4	17	22
,	5	17	22
i	6	17	22
den	7	17	22
høitidelige	8	17	22
Dunkelhed	9	17	22
,	10	17	22
som	1	18	22
dannes	2	18	22
af	3	18	22
de	4	18	22
mørkegrønne	5	18	22
Hvælvinger	6	18	22
,	7	18	22
og	8	18	22
man	9	18	22
pludselig	1	19	22
kommer	2	19	22
ud	3	19	22
i	4	19	22
det	5	19	22
fulde	6	19	22
Dagslys	7	19	22
og	8	19	22
seer	9	19	22
det	10	19	22
blaa	1	20	22
,	2	20	22
solbeskinnede	3	20	22
Hav	4	20	22
,	5	20	22
da	6	20	22
gribes	7	20	22
man	8	20	22
af	9	20	22
Undren	10	20	22
og	11	20	22
Glæde	1	21	22
,	2	21	22
og	3	21	22
Hjertet	4	21	22
føler	5	21	22
sig	6	21	22
mildt	7	21	22
bevæget	8	21	22
ved	9	21	22
dette	10	21	22
yndige	1	22	22
og	2	22	22
rige	3	22	22
Naturbillede	4	22	22
.	5	22	22
Paa	1	23	22
dette	2	23	22
Herresæde	3	23	22
boede	4	23	22
nu	5	23	22
Eiler	6	23	22
Grubbe	7	23	22
og	8	23	22
Ida	1	24	22
Krabbe	2	24	22
som	3	24	22
Ægtefolk	4	24	22
.	5	24	22
—	6	24	22
Da	7	24	22
Grubbe	8	24	22
strax	9	24	22
efter	10	24	22
*	1	25	22
)	2	25	22
Findes	3	25	22
ikke	4	25	22
under	5	25	22
dette	6	25	22
Navn	7	25	22
.	8	25	22
Forlovelsen	1	1	23
underrettede	2	1	23
sin	3	1	23
Svigerfader	4	1	23
</text>"""

    # Add uncorrected OCR and difference measures as annotation layers.
    new_vrt = add_ocr_tokens(vrttext, os.path.join(conf['intermediatedir'], '2-uncorrected'))
    # print(new_vrt)
    # Add corrected OCR as and difference measures as annotation layers.
    new_corr_vrt = add_corrected_ocr_tokens(new_vrt, os.path.join(conf['intermediatedir'], '3-corrected'))
    # Add annotations from Text Tonsorium CONLL output.
    new_conll_vrt = add_conll(new_corr_vrt, os.path.join(conf['intermediatedir'], 'tt_output'))
    # print(new_conll_vrt)
    # Add <sentence> segmentation based on CONLL token enumeration per sentence.
    segmented_vrt = add_sentence_elems(new_conll_vrt)
    print(segmented_vrt)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def add_ocr_tokens(novel_vrt: str, ocr_dir: str, freqlist_forms):
    """Align and add tokens from OCR pages to one-novel VRT string (where each token is annotated with page numbers).
    And some difference measures."""
    text_elem, page_tokentuples = get_page_tokentuples(novel_vrt)
    ocr_pages = get_ocr_pages(ocr_dir, text_elem)
    ocr_page_strings = [readfile(f) for f in ocr_pages]
    ocr_page_strings = fix_hyphens(ocr_page_strings)
    if len(page_tokentuples) != len(ocr_pages):
        print('OCR dir:', ocr_dir, text_elem)
        raise Exception('Number of novel pages and number of VRT pages do not match.')
    new_vrt_lines = [f'{text_elem}']  # Put original text element back.
    for ocr_text, vrt_tokentups in zip(ocr_page_strings, page_tokentuples):
        ocr_tokens = tuple(tokenize(ocr_text))
        vrt_tokens = tuple([tup[0] for tup in vrt_tokentups])
        aligned_ocr_toks = align_b_to_a(vrt_tokens, ocr_tokens)
        # Hack to deal with long sequences of tokens due to unequal page samples
        aligned_ocr_toks = [tok if len(tok) < 100 else tok[:30] + '...' for tok in aligned_ocr_toks]
        new_vrt_tups = add_annotation_layer(vrt_tokentups, aligned_ocr_toks)
        new_vrt_tups = add_diff_measures(new_vrt_tups, vrt_tokens, aligned_ocr_toks)
        new_vrt_tups = add_in_freqlist(new_vrt_tups, aligned_ocr_toks, freqlist_forms)
        new_vrt_lines.append('\n'.join(['\t'.join(x) for x in new_vrt_tups]))
    new_vrt_lines.append('</text>')
    return '\n'.join(new_vrt_lines)


def get_page_tokentuples(novel_vrt: str):
    """Get text 'header' and per-page tuples of tokens from token tuples.
       (Where each token is annotated with page number as a P-attribute)."""
    text_elem, tokentuples = get_tokentuples(novel_vrt)
    # Group on page number (tokentup[3]) to get a tokenlist for each line.
    page_tokentuples = [tuple(grp) for _, grp in groupby(tokentuples, key=lambda tokentup: tokentup[4])]
    return text_elem, page_tokentuples


def add_corrected_ocr_tokens(novel_vrt: str, corr_dir: str, freqlist_forms):
    """Align and add tokens from one-novel corrected OCR string to one-novel VRT string.
    And some difference measures."""
    text_elem, vrt_tokentups = get_tokentuples(novel_vrt)
    ocr_pages = get_ocr_pages(corr_dir, text_elem)
    ocr_page_strings = [readfile(f) for f in ocr_pages]
    ocr_page_strings = fix_hyphens(ocr_page_strings)
    ocr_string = '\n'.join(ocr_page_strings)
    ocr_string = re.sub(r'\s*___PAGEBREAK___\s*', '\n', ocr_string)
    new_vrt_lines = [f'{text_elem}']  # Put original text element back.
    ocr_tokens = tuple(tokenize(ocr_string))
    vrt_tokens = tuple([tup[0] for tup in vrt_tokentups])
    aligned_ocr_toks = align_b_to_a(vrt_tokens, ocr_tokens)
    # Hack to deal with long sequences of tokens due to unequal page samples
    aligned_ocr_toks = [tok if len(tok) < 100 else tok[:30] + '...' for tok in aligned_ocr_toks]
    new_vrt_tups = add_annotation_layer(vrt_tokentups, aligned_ocr_toks)
    new_vrt_tups = add_diff_measures(new_vrt_tups, vrt_tokens, aligned_ocr_toks)
    new_vrt_tups = add_in_freqlist(new_vrt_tups, aligned_ocr_toks, freqlist_forms)
    new_vrt_lines.append('\n'.join(['\t'.join(x) for x in new_vrt_tups]))
    new_vrt_lines.append('</text>')
    return '\n'.join(new_vrt_lines)


def get_tokentuples(novel_vrt: str):
    """Get text 'header' and all tuples of tokens from a novel represented as a VRT <text>...</text> string."""
    lines = novel_vrt.splitlines()
    if not lines[0].startswith('<text'):
        raise Exception('VRT does not start with "<text')
    if not lines[-1] == '</text>':
        raise Exception('VRT does not end with </text>')
    text_elem = lines[0]
    tokentuples = [tuple(line.split('\t')) for line in lines[1:-1]]
    return text_elem, tokentuples


def get_ocr_pages(ocr_dir, text_elem):
    """Find the correct novel dir in ocr_dir based on id in text_elem, and return list of page paths."""
    novel_id = re.search(r'id="([^"]+)"', text_elem).group(1)
    novel_dirs = [x for x in os.listdir(ocr_dir) if x.startswith(novel_id)]
    if len(novel_dirs) != 1:
        raise Exception(f'Did not find unique novel dir for novel id "{novel_id}"')
    pages = sorted_listdir(os.path.join(ocr_dir, novel_dirs[0]))
    return [os.path.join(ocr_dir, novel_dirs[0], page) for page in pages]


def add_diff_measures(vrt_tokentups, vrt_tokens, aligned_ocr_toks):
    """Add difference measures pertaining to vrt_tokens and aligned_ocr_toks. CER = char error rate."""
    lev_dists = [lev_dist(v, a) for v, a in zip(vrt_tokens, aligned_ocr_toks)]
    lev_ratios = [round(lev_ratio(v, a), 2) for v, a in zip(vrt_tokens, aligned_ocr_toks)]
    cers = [round(1.0 - ratio, 2) for ratio in lev_ratios]
    types = [get_difftype(v, a) for v, a in zip(vrt_tokens, aligned_ocr_toks)]
    op_strings = [get_op_str(v, a) for v, a in zip(vrt_tokens, aligned_ocr_toks)]
    new_vrt_tups = add_annotation_layer(vrt_tokentups, lev_dists)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, lev_ratios)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, cers)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, types)
    new_vrt_tups = add_annotation_layer(new_vrt_tups, op_strings)
    return new_vrt_tups


def add_in_freqlist(vrt_tokentups, aligned_ocr_toks, freqlist_forms):
    """Add a boolean annotation layer (1 = token exists in freqlist, 0 = not)."""
    in_freq = [str(int(tok.lower() in freqlist_forms)) for tok in aligned_ocr_toks]
    new_vrt_tups = add_annotation_layer(vrt_tokentups, in_freq)
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


def add_conll(novel_vrt: str, conll_dir: str):
    """Align and add tokens from sentence segmented, lemmatized, and PoS-tagged CONLL file
       to one-novel VRT string."""
    text_elem, vrt_tokentuples = get_tokentuples(novel_vrt)
    conll_tokentuples = get_conll_tokentuples(conll_dir, text_elem)
    aligned_tokentuples = align_conll_tuples(vrt_tokentuples, conll_tokentuples)
    new_tokenlines = '\n'.join(['\t'.join(tup) for tup in aligned_tokentuples])
    new_vrt = f'{text_elem}\n{new_tokenlines}\n</text>'
    return new_vrt


def get_conll_tokentuples(conll_dir: str, text_elem: str):
    """Get token tuples with annotations from CONLL file based on id in text_elem."""
    def clean(string): return re.sub(r'\W', '_', string)
    novel_id = re.search(r'id="([^"]+)"', text_elem).group(1)
    clean_novel_id = clean(novel_id)
    novel_files = [x for x in os.listdir(conll_dir) if clean(x).startswith(clean_novel_id)]
    if len(novel_files) != 1:
        raise Exception(f'Did not find unique CONLL file for novel id "{novel_id}"')
    conll_path = os.path.join(conll_dir, novel_files[0])
    with open(conll_path, 'r') as conll_file:
        lines = conll_file.read().splitlines()
    # Remove empty lines
    lines = [line for line in lines if line]
    return [tuple(line.split('\t')) for line in lines]


def add_sentence_elems(novel_vrt: str):
    """Add <sentence> elements to one-novel VRT string (*without* <sentence> or other elements)
       based on CONLL sentence enumeration."""
    lines = novel_vrt.splitlines()
    # lines[1:-1]: Skip <text> element
    # TODO Brittle: line.split('\t')[-4] == '1': Word enumeration per sentence is in the -4 column.
    #  It ougth to be in a fixed position in the front.
    sentence_startgroups = groupby(lines[1:-1], key=lambda line: line.split('\t')[-4] == '1')
    sentence_id = 1
    sentence_strings = []
    for k, grp in sentence_startgroups:
        if k:
            try:
                sentence_lines = chain([next(grp)], (next(sentence_startgroups)[1]))
                sentence = '\n'.join(sentence_lines)
                sentence_strings.append(f'<sentence id="{sentence_id}">\n{sentence}\n</sentence>')
                sentence_id += 1
            except StopIteration:
                print('Ups:', lines[0])
                break
    joined_sents = '\n'.join(sentence_strings)
    new_vrt = f'{lines[0]}\n{joined_sents}\n</text>'
    return new_vrt


def add_ocr_tokens_recursive(text, tess_outdirs: list, freqlist_forms):
    """Return text with added tokens from each tesseract-OCR-model (cf. traineddata_labels)."""
    if not tess_outdirs:
        return text
    else:
        new_text = add_ocr_tokens(text, tess_outdirs[0], freqlist_forms)
        return add_ocr_tokens_recursive(new_text, tess_outdirs[1:], freqlist_forms)


def add_corr_tokens_recursive(text, corr_dirs: list, freqlist_forms):
    """Return text with added tokens from each dir with corrected novel pages."""
    if not corr_dirs:
        return text
    else:
        new_text = add_corrected_ocr_tokens(text, corr_dirs[0], freqlist_forms)
        return add_corr_tokens_recursive(new_text, corr_dirs[1:], freqlist_forms)


def add_gold_in_freq(novel_vrt: str, freqlist_forms):
    """Add a boolean annotation layer wrt. gold tokens (1 = token exists in freqlist, 0 = not)."""
    new_lines = list()
    for line in novel_vrt.splitlines():
        if line.startswith('<text ') or line == '</text>':
            new_lines.append(line)
        else:
            gold_token = line.split('\t')[0]
            in_freq = str(int(gold_token.lower() in freqlist_forms))
            new_line = f'{line}\t{in_freq}'
            new_lines.append(new_line)
    return '\n'.join(new_lines)


if __name__ == '__main__':
    main()
