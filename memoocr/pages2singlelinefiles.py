import os
import sys
import myutils as util


def pages2singlelinefiles(indir, outdir):
    """Transform single page novels to single line novels
    with \v and \f representing line and page breaks"""
    for noveldir in util.sorted_listdir(indir):
        full_novel_path = os.path.join(indir, noveldir)
        pages = util.sorted_listdir(full_novel_path)
        if len(pages) != 1:
            sys.exit(f'ERROR: No pages or more than 1 page in novel dir {full_novel_path}')
        pagetext = util.readfile(os.path.join(full_novel_path, pages[0]))
        singleline = pagetext.replace(util.PAGEBREAK, '\f').replace('\n', ' \v')
        with open(os.path.join(outdir, f'{noveldir}.txt'), 'w') as f:
            f.write(singleline)
