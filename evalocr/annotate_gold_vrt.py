"""
annotate_gold_vrt.py
Annotate gold standard VRT file containing several novels with original OCR tokens and difference measures.
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


def generate_gold_annotations(vrt_file, ocr_kb_dir, conll_dir, corpus_id, tess_outdirs, corr_dirs):
    """Generator that adds annotations to each text element in original VRT file."""
    yield f'<corpus id="{corpus_id}">'
    text_generator = split_vrt(vrt_file)
    for text in text_generator:
        print(text.splitlines()[0])
        text_w_ocr = add_ocr_tokens_recursive(text, tess_outdirs)
        text_w_kb_ocr = add_ocr_tokens(text_w_ocr, ocr_kb_dir)
        text_w_corr_ocr = add_corr_tokens_recursive(text_w_kb_ocr, corr_dirs)
        text_w_conll = add_conll(text_w_corr_ocr, conll_dir)
        text_w_sents = add_sentence_elems(text_w_conll)
        yield text_w_sents
    yield '</corpus>'


def add_ocr_tokens_recursive(text, tess_outdirs: list):
    """Return text with added tokens from each tesseract-OCR-model (cf. traineddata_labels)."""
    if not tess_outdirs:
        return text
    else:
        new_text = add_ocr_tokens(text, tess_outdirs[0])
        return add_ocr_tokens_recursive(new_text, tess_outdirs[1:])


def add_corr_tokens_recursive(text, corr_dirs: list):
    """Return text with added tokens from each dir with corrected novel pages."""
    if not corr_dirs:
        return text
    else:
        new_text = add_corrected_ocr_tokens(text, corr_dirs[0])
        return add_corr_tokens_recursive(new_text, corr_dirs[1:])


if __name__ == '__main__':
    main()
