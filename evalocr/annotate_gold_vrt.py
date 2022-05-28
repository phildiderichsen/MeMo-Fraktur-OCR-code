"""
annotate_gold_vrt.py
Annotate gold standard VRT file containing several novels with original OCR tokens and difference measures.
"""
import configparser
import os
from datetime import datetime
from evalocr import ROOT_PATH
from memoocr.add_vrt_annotations import add_ocr_tokens, add_conll, add_sentence_elems, \
    add_gold_in_freq, add_ocr_tokens_recursive, add_corr_tokens_recursive
import myutils as util


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
    basic_gold_vrt_path = os.path.join(vrt_dir, corpus_id + '.vrt')
    annotated_outdir = os.path.join(conf['annotated_outdir'], corpus_id)
    annotated_gold_vrt_path = os.path.join(annotated_outdir, corpus_id + '.annotated.vrt')
    try:
        os.makedirs(annotated_outdir)
    except FileExistsError:
        pass

    annotated_gold_vrt_gen = generate_gold_annotations(basic_gold_vrt_path, ocr_dir, ocr_kb_dir,
                                                       ocr_dir2, conll_dir, corpus_id)
    write_annotated_gold_vrt(annotated_gold_vrt_gen, annotated_gold_vrt_path)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def write_annotated_gold_vrt(text_annotation_generator, annotated_gold_vrt_path):
    """"""
    with open(annotated_gold_vrt_path, 'w') as outfile:
        for chunk in text_annotation_generator:
            outfile.write(chunk + '\n')


def generate_gold_annotations(vrt_file, ocr_kb_dir, conll_dir, corpus_id, tess_outdirs, corr_dirs, conf, metadata):
    """Generator that adds annotations to each text element in original VRT file."""
    yield f'<corpus id="{corpus_id}">'
    text_generator = util.split_vrt(vrt_file)
    for text in text_generator:
        print(text.splitlines()[0])
        text_w_ocr = add_ocr_tokens_recursive(text, tess_outdirs, util.freqlist_forms, metadata)
        text_w_kb_ocr = add_ocr_tokens(text_w_ocr, ocr_kb_dir, util.freqlist_forms, metadata)
        text_w_corr_ocr = add_corr_tokens_recursive(text_w_kb_ocr, corr_dirs, util.freqlist_forms, metadata)
        text_w_conll = add_conll(text_w_corr_ocr, conll_dir, metadata)
        text_w_gold_infreq = add_gold_in_freq(text_w_conll, util.freqlist_forms)
        text_w_sents = add_sentence_elems(text_w_gold_infreq)
        yield text_w_sents
    yield '</corpus>'


if __name__ == '__main__':
    main()
