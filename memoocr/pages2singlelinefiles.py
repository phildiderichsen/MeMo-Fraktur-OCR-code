import os
import re
import sys
import myutils as util


def pages2singlelinefiles(files_to_process, indir, outdir, metadata):
    """Transform single page novels to single line novels
    with \v and \f representing line and page breaks"""
    for novel in [f.replace('.pdf', '') for f in files_to_process]:
        full_novel_path = os.path.join(indir, novel)
        pages = util.sorted_listdir(full_novel_path)
        if len(pages) != 1:
            sys.exit(f'ERROR: No pages or more than 1 page in novel dir {full_novel_path}')
        pagetext = util.readfile(os.path.join(full_novel_path, pages[0]))
        singleline = pagetext.replace(util.PAGEBREAK, '\f').replace('\n', ' \v')
        outname = metadata[novel]['filename'].replace('.pdf', '')
        with open(os.path.join(outdir, f'{outname}.txt'), 'w') as f:
            f.write(singleline)
