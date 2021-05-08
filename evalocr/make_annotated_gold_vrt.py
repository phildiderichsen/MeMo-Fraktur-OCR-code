"""
make_annotated_gold_vrt.py
Annotate gold standard VRT file containing several novels with original OCR tokens and difference measures.
"""
import configparser
import os
from datetime import datetime
from evalocr import ROOT_PATH
from myutils import split_vrt
from memoocr.make_corpus_vrt import make_novels_vrt
from memoocr.add_vrt_annotations import add_ocr_tokens, add_corrected_ocr_tokens, add_conll, add_sentence_elems


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_PATH, 'config', 'config.ini'))
    conf = config['DEFAULT']

    corpus_id = conf['fraktur_gold_vrt_label']
    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')
    gold_novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    ocr_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    ocr_kb_dir = os.path.join(conf['intermediatedir'], 'orig_pages')
    ocr_dir2 = os.path.join(conf['intermediatedir'], '3-corrected')
    conll_dir = os.path.join(conf['intermediatedir'], 'tt_output')
    annotated_outdir = os.path.join(conf['annotated_outdir'], corpus_id)
    try:
        os.makedirs(annotated_outdir)
    except FileExistsError:
        pass

    make_annotated_gold_vrt(gold_novels_dir, vrt_dir, annotated_outdir, corpus_id, ocr_dir, ocr_kb_dir, ocr_dir2, conll_dir)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def make_annotated_gold_vrt(gold_novels_dir, vrt_dir, annotated_outdir, corpus_id, ocr_dir, ocr_kb_dir, ocr_dir2, conll_dir):
    """On the basis of the non-annotated vrt_file, create a new VRT file with various OCR outputs as annotations."""
    # TODO Ikke skrive til fil sådan her helt random midt i det hele - no side effects!
    gold_vrt = make_novels_vrt(gold_novels_dir, vrt_dir, corpus_id)
    text_annotation_generator = generate_gold_annotations(gold_vrt, ocr_dir, ocr_kb_dir, ocr_dir2, conll_dir)
    new_vrt = os.path.join(annotated_outdir, os.path.basename(gold_vrt).removesuffix('.vrt') + '.annotated.vrt')
    print('New VRT:', new_vrt)
    with open(new_vrt, 'w') as outfile:
        outfile.write(f'<corpus id="{corpus_id}">\n')
        for text in text_annotation_generator:
            outfile.write(text)
            outfile.write('\n')
        outfile.write('</corpus>')
    return new_vrt


def generate_gold_annotations(vrt_file, ocr_dir, ocr_kb_dir, ocr_dir2, conll_dir):
    """Generator that adds annotations to each text element in original VRT file."""
    text_generator = split_vrt(vrt_file)
    for text in text_generator:
        print(text.splitlines()[0])
        text_w_ocr = add_ocr_tokens(text, ocr_dir)
        text_w_kb_ocr = add_ocr_tokens(text_w_ocr, ocr_kb_dir)
        text_w_corr_ocr = add_corrected_ocr_tokens(text_w_kb_ocr, ocr_dir2)
        text_w_conll = add_conll(text_w_corr_ocr, conll_dir)
        text_w_sents = add_sentence_elems(text_w_conll)
        yield text_w_sents


if __name__ == '__main__':
    main()
