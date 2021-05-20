"""
run_eval_pipeline.py
Run evaluation pipeline on gold standard data.
"""
import configparser
import os

from datetime import datetime
from memoocr.ocr import do_ocr
from evalocr.annotate_gold_vrt import generate_gold_annotations, write_annotated_gold_vrt
from evalocr.analyze_gold_vrt import analyze_gold_vrt
from memoocr.make_corpus_vrt import generate_novels_vrt, write_novels_vrt
from memoocr.correct_ocr import correct_ocr


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join('config', 'config.ini'))
    conf = config['eval']
    intermediate = conf['intermediatedir']

    corp_label = conf['fraktur_gold_vrt_label']
    vrt_dir = os.path.join(intermediate, 'vrt')
    gold_novels_dir = os.path.join(intermediate, 'corr_pages')
    annotated_outdir = os.path.join(conf['annotated_outdir'], corp_label)
    ocr_kb_dir = os.path.join(intermediate, 'orig_pages')
    corr_ocr_dir = os.path.join(intermediate, '3-corrected')
    conll_dir = os.path.join(intermediate, 'tt_output')
    img_dir = os.path.join(intermediate, '1-imgs')

    # Which OCR traineddata should be used?
    traineddata_labels = ['fraktur', 'dan', 'frk']
    tess_outdirs = [os.path.join(intermediate, f'tess_out_{label}') for label in traineddata_labels]

    basic_gold_vrt_path = os.path.join(vrt_dir, corp_label + '.vrt')
    annotated_gold_vrt_path = os.path.join(annotated_outdir, corp_label + '.annotated.vrt')

    uncorrected_dirs = [os.path.join(intermediate, f'tess_out_{label}') for label in traineddata_labels]
    uncorrected_dirs.append(os.path.join(intermediate, 'orig_pages'))
    corrected_dirs = [f'{d}_corr' for d in uncorrected_dirs]


    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_ocr'):
        # Note! frk.traineddata must be downloaded from tessdata_fast in order to work: https://github.com/tesseract-ocr/tessdata_fast/blob/master/frk.traineddata
        # Same for dan.traineddata: https://github.com/tesseract-ocr/tessdata_fast/blob/master/dan.traineddata
        # fraktur.traineddata can be downloaded from tessdata_best: https://github.com/tesseract-ocr/tessdata_best/blob/master/script/Fraktur.traineddata
        do_ocr(img_dir, conf['intermediatedir'], traineddata_labels)
    if conf.getboolean('correct_ocr'):
        print(uncorrected_dirs)
        correct_ocr(conf, uncorrected_dirs)
    if conf.getboolean('make_basic_gold_vrt'):
        gold_vrt_gen = generate_novels_vrt(gold_novels_dir, corp_label)
        write_novels_vrt(gold_vrt_gen, basic_gold_vrt_path)
    if conf.getboolean('annotate_gold_vrt'):
        text_annotation_generator = generate_gold_annotations(basic_gold_vrt_path, ocr_kb_dir,
                                                              conll_dir, corp_label, tess_outdirs, corrected_dirs)
        write_annotated_gold_vrt(text_annotation_generator, annotated_gold_vrt_path)
    if conf.getboolean('analyze_errors'):
        # TODO Not very transparent error when n_datasets is wrong.
        analyze_gold_vrt(annotated_gold_vrt_path, conf, n_datasets=8)
    if conf.getboolean('write_word'):
        pass

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
