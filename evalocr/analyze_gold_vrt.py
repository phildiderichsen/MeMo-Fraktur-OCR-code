"""
analyze_gold_vrt.py
From a VRT file with a gold standard as well as one or more corrected OCR candidates,
create an error dataset and analyze it.
"""

import configparser
import os
import re
import myutils as util
import pandas as pd

from evalocr.analyze_errors import make_stats, print_align_examples
from evalocr import ROOT_PATH


def main():
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_PATH, 'config', 'config.ini'))
    conf = config['DEFAULT']
    corp_label = conf['fraktur_gold_vrt_label']
    vrt_path = os.path.join(conf['annotated_outdir'], corp_label, corp_label + '.annotated.vrt')

    df = transform_vrt(vrt_path)
    make_stats(df, conf, 'blahaaaa.txt')


def transform_vrt(vrt_path):
    vrt_lines = util.readfile(vrt_path).splitlines()
    token_lines = [line.split('\t') for line in vrt_lines if not re.match(r'</?(corpus|text|sentence)', line)]
    df = pd.DataFrame(token_lines)

    print('VRT dataset so far:')
    print(df)
    print(list(df))
    return df


if __name__ == '__main__':
    main()
