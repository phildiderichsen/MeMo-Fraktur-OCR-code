"""
run_eval_pipeline.py
Run evaluation pipeline on gold standard data.
"""
import configparser
import os

from datetime import datetime
from evalocr.make_annotated_gold_vrt import make_annotated_gold_vrt


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join('config', 'config.ini'))
    conf = config['eval']

    corp_label = conf['fraktur_gold_vrt_label']
    gold_novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    annotated_outdir = os.path.join(conf['annotated_outdir'], corp_label)
    ocr_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    ocr_kb_dir = os.path.join(conf['intermediatedir'], 'orig_pages')
    ocr_dir2 = os.path.join(conf['intermediatedir'], '3-corrected')
    conll_dir = os.path.join(conf['intermediatedir'], 'tt_output')

    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')
    vrt_path = os.path.join(vrt_dir, corp_label + '.annotated.vrt')

    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('make_gold_vrt'):
        gold_vrt = make_annotated_gold_vrt(gold_novels_dir, vrt_dir, annotated_outdir,
                                           corp_label, ocr_dir, ocr_kb_dir, ocr_dir2, conll_dir)
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
