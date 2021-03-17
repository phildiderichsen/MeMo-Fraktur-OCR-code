"""
get_gold.py
Get gold data from the MeMo Fraktur gold standard, and put them in folders page by page per novel.
"""

import configparser
import os
import re


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    mode = 'test'
    conf = config[mode]
    save_evaldata(conf)


def save_evaldata(config, n=None):
    """Save original lines and gold lines to individual novel pages"""
    # TODO Handle hyphens (first, get them back into Guldstandard.txt)
    with open(config['evaldata'], 'r', encoding='utf8') as evaldata:
        data = evaldata.read().splitlines()
    # Remove header if present.
    if 'Dokumentnavn' in data[0]:
        data = data[1:n]
    data = [line.split('\t') for line in data]
    datadicts = [dict(zip(['novel', 'orig', 'corr', 'comment', 'page'], fields)) for fields in data]
    # Remove empty lines
    datadicts = [dd for dd in datadicts if dd['novel']]
    # Remove '[paragraph]', '[section]', '[section + line]', '[page]' lines.
    rgx = r'\[(paragraph|section|page)'
    datadicts = [dd for dd in datadicts if not re.search(rgx, dd['orig'])]
    # Write to files
    write_pages(datadicts, 'corr', config['intermediatedir'])
    write_pages(datadicts, 'orig', config['intermediatedir'])


def write_pages(datadicts, source, interdir):
    """Write novel pages to correct folder."""
    novel_pages = {}
    for dct in datadicts:
        if (dct['novel'], dct['page']) in novel_pages:
            novel_pages[(dct['novel'], dct['page'])].append(dct[source])
        else:
            novel_pages[(dct['novel'], dct['page'])] = [dct[source]]

    for keytuple in novel_pages:
        noveldir = keytuple[0].removesuffix('.pdf')
        page = keytuple[1]
        filename = f'page_{page}.txt'
        noveldir = os.path.join(interdir, source + '_pages', noveldir)
        os.makedirs(noveldir, exist_ok=True)
        with open(os.path.join(noveldir, filename), 'w') as novelpage:
            novelpage.write('\n'.join(novel_pages[keytuple]))


if __name__ == '__main__':
    main()
