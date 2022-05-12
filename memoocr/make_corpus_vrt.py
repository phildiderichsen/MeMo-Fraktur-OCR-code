"""
make_corpus_vrt.py
Make VRT file(s) for a whole corpus.

Transform pages in a folder to a VRT file.

- Eliminate hyphenation at EOL.
- Tokenize text.
- Keep word number on line, line and page number as annotation layers.
- When concatenating pages, eliminate hyphenation at end of page.
"""

import configparser
import csv
import os
import re
from datetime import datetime
from myutils import sorted_listdir, tokenize, readfile

from memoocr import ROOT_PATH


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_PATH, 'config', 'config.ini'))
    conf = config['DEFAULT']

    novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')
    corpus_id = 'MEMO_FRAKTUR_GOLD'

    novels_vrt_gen = generate_novels_vrt_from_pages(novels_dir, corpus_id)
    write_novels_vrt(novels_vrt_gen, os.path.join(vrt_dir, corpus_id + '.vrt'))

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def generate_novels_vrt_from_pages(novels_dir, corpus_id):
    """Generator that yields the lines of a VRT file with all novels in a corpus."""
    novel_ids = sorted_listdir(novels_dir)
    novel_dirs = [os.path.join(novels_dir, d) for d in novel_ids]
    yield f'<corpus id="{corpus_id}">' + '\n'
    for novel_id, novel_dir in zip(novel_ids, novel_dirs):
        # Convert novel.
        novel_vrt = pages2vrt(novel_dir)
        yield novel_vrt + '\n'
    yield '</corpus>' + '\n'


def pages2vrt(pagedir):
    """Convert pages to text represented in VRT format."""

    def get_pagenum(page: str):
        """Extract page number from page filename."""
        return re.search(r'page_(\d+)', page).group(1)

    text_id = os.path.basename(pagedir)
    pages = sorted_listdir(pagedir)
    pagepaths = [os.path.join(pagedir, p) for p in pages]
    pagenums = [int(get_pagenum(p)) for p in pages]
    tokenlists = [page2tokens(page, pagenum, text_id) for page, pagenum in zip(pagepaths, pagenums)]
    texttokens = flatten_tokenlists(tokenlists)
    vrt_lines = [f'{d["token"]}\t{d["i"]}\t{d["j"]}\t{d["line"]}\t{d["page"]}\t{d["text_id"]}' for d in texttokens]
    vrt_text = '<text id="{}">\n{}\n</text>'.format(text_id, "\n".join(vrt_lines))
    return vrt_text


def page2tokens(page, pagenum, text_id):
    """Transform a single page to a list of dicts each containing a token and some annotations."""

    def handle_hyphens(text):
        """Eliminate soft hyphens (\xad) and end of line hyphens."""
        new_text = re.sub(r'\xad *', '', text)
        new_text = re.sub(r'(\S+)[\xad⸗—-]\n(\S+) ', r'\1\2\n', new_text)
        return new_text

    def make_line_tokens(line: str, linenum: int, _pagenum: int, _text_id):
        """Tokenize a line and enumerate the tokens by number on line, line number, and page number."""
        tokens = tokenize(line)
        return [{'token': tok, 'i': i + 1, 'line': linenum, 'page': _pagenum, 'text_id': _text_id}
                for i, tok in enumerate(tokens)]

    with open(page, 'r') as infile:
        pagetext = infile.read()
    pagetext = handle_hyphens(pagetext)
    pagelines = [make_line_tokens(line, linenum + 1, pagenum, text_id)
                 for linenum, line in enumerate(pagetext.splitlines())]
    pagetokens = [tokendict for sublist in pagelines for tokendict in sublist]
    # Add overall token enumeration.
    new_pagetokens = [{'token': tok['token'],
                       'i': i + 1,
                       'j': tok['i'],
                       'line': tok['line'],
                       'page': tok['page'],
                       'text_id': tok['text_id']}
                      for i, tok in enumerate(pagetokens)]
    return new_pagetokens


def flatten_tokenlists(tokenlists: list):
    """Flatten lists of page tokens to one big list of text tokens. Eliminate cross-page hyphens."""

    def handle_cross_page_hyphen(page_toks: list, next_page_toks: list):
        """Eliminate hyphenation between page and next_page, if any."""
        if re.match(r'\S+[⸗—-]', page_toks[-1]["token"]) and re.match(r'\S+', next_page_toks[0]["token"]):
            new_prefix = re.sub(r'(\S+)[⸗—-]', r'\1', page_toks[-1]["token"])
            page_toks[-1]["token"] = new_prefix + next_page_toks[0]["token"]
            next_page_toks = next_page_toks[1:]
        return page_toks, next_page_toks

    # For each page: a tuple of the index of the page and the index of the next page (if any)
    indexes = list(range(len(tokenlists)))
    indextuples = list(zip(indexes, indexes[1:]))
    for indextup in indextuples:
        i, j = indextup
        tokenlists[i], tokenlists[j] = handle_cross_page_hyphen(tokenlists[i], tokenlists[j])
    texttokens = [tokendict for sublist in tokenlists for tokendict in sublist]
    return texttokens


def generate_novels_vrt_from_text(novels_dir, corpus_id):
    """Generator that yields the lines of a VRT file with all novels in a corpus."""
    novel_ids = sorted_listdir(novels_dir)
    novel_dirs = [os.path.join(novels_dir, d) for d in novel_ids]
    yield f'<corpus id="{corpus_id}">' + '\n'
    for novel_id, novel_dir in zip(novel_ids, novel_dirs):
        # Convert novel text to VRT <text> element.
        yield text2vrt(novel_dir) + '\n'
    yield '</corpus>' + '\n'


def text2vrt(textdir):
    """Convert a whole novel in a text file to VRT format."""
    text_id = os.path.basename(textdir)
    text = [f for f in sorted_listdir(textdir) if f.endswith('.txt')][0]
    textpath = os.path.join(textdir, text)
    token_generator = text2tokens(textpath, text_id)
    vrt_lines = [f'{d["token"]}\t{d["i"]}\t{d["j"]}\t{d["line"]}\t{d["page"]}\t{d["text_id"]}' for d in token_generator]
    vrt_text = '<text id="{}">\n{}\n</text>'.format(text_id, "\n".join(vrt_lines))
    return vrt_text


def text2tokens(text, text_id):
    """Transform a text to a list of dicts each containing a token and
        some annotations (token, line, and page numbers).
        If the text contains instances of the token "___PAGEBREAK___",
        they will be discarded and the page incremented (and token and line numbering restarted)."""

    with open(text, 'r') as infile:
        pagetext = infile.read()
    # Tokenize lines and enumerate the tokens on each line, and also add line numbers and page numbers."""
    wordnum = 1
    linenum = 1
    pagenum = 1
    for line in pagetext.splitlines():
        tokens = tokenize(line)
        wordnum_on_line = 1
        for token in tokens:
            if '___PAGEBREAK___' in token:
                wordnum_on_line = 1
                linenum = 1
                pagenum += 1
            else:
                tokendict = {'token': token, 'i': wordnum, 'j': wordnum_on_line,
                             'line': linenum, 'page': pagenum, 'text_id': text_id}
                wordnum += 1
                wordnum_on_line += 1
                yield tokendict
        linenum += 1


def write_novels_vrt(vrt_generator, outpath):
    """Write a single VRT for all novels in corpus."""
    with open(outpath, 'w') as f:
        for line in vrt_generator:
            f.write(line)


def add_metadata(annotated_vrt, metadatafile):
    """Copy in metadata from metadatafile."""
    vrt_data = readfile(annotated_vrt)
    with open(metadatafile, newline='') as f:
        metadatarows = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        frakturrows = [row for row in metadatarows
                       if row['typeface (roman or gothic)'] and row['filename']]
    for row in frakturrows:
        if 'gothic' in row['typeface (roman or gothic)']:
            novel_id = row['filename'].replace('.pdf', '')
            metadatastr = f'''<text id="{novel_id}" file_id="{row['file_id']}" firstname="{row['first name']}" surname="{row['surname']}" pseudonym="{row['pseudonym']}" gender="{row['gender (m or f)']}" nationality="{row['nationality (dk or no)']}" title="{row['title']}" subtitle="{row['subtitle']}" volume="{row['volume']}" year="{row['year']}" pages="{row['pages']}" illustrations="{row['illustrations (y or n)']}" typeface="{row['typeface (roman or gothic)']}" publisher="{row['publisher']}" price="{row['price']}" source="{row['source']}" notes="{row['notes']}" readable="{row['readable (y or n)']}">'''
            vrt_data = re.sub(f'<text id="{novel_id}">', metadatastr, vrt_data)
    text_elems_without_metadata = re.findall(r'<text id="[^"\n\r]+">', vrt_data)
    if text_elems_without_metadata:
        print('WARNING: Nogle text-elementer har ikke fået tilføjet metadata:')
        print('\n'.join(text_elems_without_metadata))
    return vrt_data


if __name__ == '__main__':
    main()
