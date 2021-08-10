"""
run_ocr_pipeline.py
Run OCR correction pipeline on full novel data.
"""
import configparser
import os
import shutil
import myutils as util

from datetime import datetime
from memoocr.make_dictionary import make_dic
from memoocr.pdf2img import pdfs2imgs
from memoocr.ocr import do_ocr
from evalocr.annotate_gold_vrt import generate_gold_annotations, write_annotated_gold_vrt
from evalocr.analyze_gold_vrt import analyze_gold_vrt
from memoocr.make_corpus_vrt import generate_novels_vrt, write_novels_vrt
from memoocr.correct_ocr import sym_wordcorrect, correct_easy_fraktur_errors, correct_hard_fraktur_errors


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = util.get_config()

    # Generate various paths and create them if necessary.
    conf = util.Confs(config).corrconf
    *_, param_str = util.get_params(conf)
    pth = util.CorrPaths(conf)

    # Which OCR traineddata should be used?
    # Note! frk.traineddata must be downloaded from tessdata_fast in order to work:
    # https://github.com/tesseract-ocr/tessdata_fast/blob/master/frk.traineddata
    # Same for dan.traineddata: https://github.com/tesseract-ocr/tessdata_fast/blob/master/dan.traineddata
    # fraktur.traineddata can be downloaded from tessdata_best:
    # https://github.com/tesseract-ocr/tessdata_best/blob/master/script/Fraktur.traineddata
    traineddata_labels = ['Fraktur', 'dan', 'frk']
    tess_outdirs = [os.path.join(pth.fulloutputdir, f'tess_out_{label}') for label in traineddata_labels]
    uncorrected_dir = os.path.join(pth.fulloutputdir, conf['base_ocr'])
    corrected_dir = os.path.join(pth.fulloutputdir, param_str)

    # Steps of the pipeline. Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf['metadir'])
    if conf.getboolean('run_pdf2img'):
        pdfs2imgs(pth.frakturpaths, pth.img_dir, int(conf['split_size']))
    if conf.getboolean('run_ocr'):
        print('Running run_ocr ...\n')
        do_ocr(pth.img_dir, pth.fulloutputdir, traineddata_labels)
    if conf.getboolean('correct_easy'):
        print('Running correct_easy ...\n')
        correct_easy_fraktur_errors(uncorrected_dir, corrected_dir)
        uncorrected_dir = corrected_dir
    if conf.getboolean('correct_hard'):
        print('Running correct_hard ...\n')
        correct_hard_fraktur_errors(uncorrected_dir, pth.fulloutputdir, corrected_dir)
        uncorrected_dir = corrected_dir
    if conf.getboolean('sym_wordcorrect'):
        print('Running sym_wordcorrect ...\n')
        sym_wordcorrect(conf, uncorrected_dir, corrected_dir)
    # TODO Will it make any sense to employ SymSpell at the bigram level? Probably not?
    # if conf.getboolean('make_basic_gold_vrt'):
    #     gold_vrt_gen = generate_novels_vrt(corrpaths.gold_novels_dir, corrpaths.corp_label)
    #     write_novels_vrt(gold_vrt_gen, corrpaths.basic_gold_vrt_path)
    # if conf.getboolean('annotate_gold_vrt'):
    #     text_annotation_generator = generate_gold_annotations(corrpaths.basic_gold_vrt_path, corrpaths.ocr_kb_dir,
    #                                                           conf['texton_out_dir'], corrpaths.corp_label, tess_outdirs,
    #                                                           [corrected_dir], conf)  # TODO single dir instead of list of dirs?
    #     write_annotated_gold_vrt(text_annotation_generator, corrpaths.local_annotated_gold_vrt_path)
    #     shutil.copy(corrpaths.local_annotated_gold_vrt_path, corrpaths.annotated_gold_vrt_path)
    # if conf.getboolean('analyze_errors'):
    #     # TODO Not very transparent error when n_datasets is wrong.
    #     analyze_gold_vrt(corrpaths.annotated_gold_vrt_path, conf, corrpaths.analyses_dir, param_str, n_datasets=5)
    # if conf.getboolean('write_korp_configs'):
    #     util.write_frakturgold_mode(conf['frakturgold_mode_template'],
    #                                 conf['gold_vrt_p_attrs'],
    #                                 conf['frakturgold_mode_outpath'])
    #     shutil.copy(conf['frakturgold_mode_outpath'], os.path.join(corrpaths.vrt_dir, 'memo_frakturgold_mode.js'))
    #     util.write_frakturgold_encodescript(conf['frakturgold_encode_template'],
    #                                         corrpaths.annotated_outdir,
    #                                         conf['gold_vrt_p_attrs'],
    #                                         conf['frakturgold_encode_outpath'])
    #     shutil.copy(conf['frakturgold_encode_outpath'], os.path.join(corrpaths.vrt_dir, 'encode_MEMO_fraktur_gold.sh'))
    # if conf.getboolean('write_word'):
    #     pass

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
