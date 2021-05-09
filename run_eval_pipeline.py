"""
run_eval_pipeline.py
Run evaluation pipeline on gold standard data.
"""
import configparser
import os

from datetime import datetime
from evalocr.annotate_gold_vrt import generate_gold_annotations, write_annotated_gold_vrt
from memoocr.make_corpus_vrt import generate_novels_vrt, write_novels_vrt


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join('config', 'config.ini'))
    conf = config['eval']

    corp_label = conf['fraktur_gold_vrt_label']
    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')
    gold_novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    annotated_outdir = os.path.join(conf['annotated_outdir'], corp_label)
    ocr_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    ocr_kb_dir = os.path.join(conf['intermediatedir'], 'orig_pages')
    ocr_dir2 = os.path.join(conf['intermediatedir'], '3-corrected')
    conll_dir = os.path.join(conf['intermediatedir'], 'tt_output')

    basic_gold_vrt_path = os.path.join(vrt_dir, corp_label + '.vrt')
    annotated_gold_vrt_path = os.path.join(annotated_outdir, corp_label + '.annotated.vrt')

    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('make_basic_gold_vrt'):
        gold_vrt_gen = generate_novels_vrt(gold_novels_dir, corp_label)
        write_novels_vrt(gold_vrt_gen, basic_gold_vrt_path)
    if conf.getboolean('annotate_gold_vrt'):
        text_annotation_generator = generate_gold_annotations(basic_gold_vrt_path, ocr_dir, ocr_kb_dir,
                                                              ocr_dir2, conll_dir, corp_label)
        write_annotated_gold_vrt(text_annotation_generator, annotated_gold_vrt_path)
    if conf.getboolean('generate_dataset'):
        pass
    if conf.getboolean('analyze_errors'):
        print('Not implemented: analyze_errors')
    if conf.getboolean('write_word'):
        pass

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
