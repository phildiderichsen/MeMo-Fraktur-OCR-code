"""
annotate_vrt.py
Annotate VRT file containing several novels with e.g. original OCR tokens and difference measures.
"""
import configparser
import os
from datetime import datetime
from evalocr import ROOT_PATH
from myutils import split_vrt
from memoocr.add_vrt_annotations import add_ocr_tokens, add_corrected_ocr_tokens, add_conll, add_sentence_elems


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_PATH, 'config', 'config.ini'))
    conf = config['DEFAULT']

    corpus_id = 'MEMO_FRAKTUR_GOLD'
    vrt_file = os.path.join(conf['intermediatedir'], 'vrt', corpus_id + '.vrt')
    ocr_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    ocr_kb_dir = os.path.join(conf['intermediatedir'], 'orig_pages')
    ocr_dir2 = os.path.join(conf['intermediatedir'], '3-corrected')
    conll_dir = os.path.join(conf['intermediatedir'], 'tt_output')
    annotated_outdir = os.path.join(conf['annotated_outdir'], corpus_id)
    try:
        os.makedirs(annotated_outdir)
    except FileExistsError:
        pass
    new_vrt = os.path.join(annotated_outdir, os.path.basename(vrt_file).removesuffix('.vrt') + '.annotated.vrt')
    print(new_vrt)
    corpus_id = os.path.basename(vrt_file).removesuffix('.vrt')

    text_generator = split_vrt(vrt_file)
    with open(new_vrt, 'w') as outfile:
        outfile.write(f'<corpus id="{corpus_id}">\n')
        for text in text_generator:
            print(text.splitlines()[0])
            text_w_ocr = add_ocr_tokens(text, ocr_dir)
            text_w_kb_ocr = add_ocr_tokens(text_w_ocr, ocr_kb_dir)
            text_w_corr_ocr = add_corrected_ocr_tokens(text_w_kb_ocr, ocr_dir2)
            text_w_conll = add_conll(text_w_corr_ocr, conll_dir)
            text_w_sents = add_sentence_elems(text_w_conll)
            outfile.write(text_w_sents)
            outfile.write('\n')
        outfile.write('</corpus>')

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
