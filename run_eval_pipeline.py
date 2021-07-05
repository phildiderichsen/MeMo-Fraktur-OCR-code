"""
run_eval_pipeline.py
Run evaluation pipeline on gold standard data.
"""
import configparser
import os
import shutil
import myutils as util
from myutils import EvalPaths, safe_copytree

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
    conf = util.get_config('eval')
    *_, param_str = util.get_params(conf)

    # Generate various paths and create them if necessary.
    pth = EvalPaths(conf, param_str)

    # Copy original KB pages and Gold pages.
    safe_copytree(conf['orig_page_dir'], pth.ocr_kb_dir)
    safe_copytree(conf['gold_page_dir'], pth.gold_novels_dir)
    # TODO The script for generating pages of corrected gold standard text got lost ..

    # Which OCR traineddata should be used?
    # Note! frk.traineddata must be downloaded from tessdata_fast in order to work:
    # https://github.com/tesseract-ocr/tessdata_fast/blob/master/frk.traineddata
    # Same for dan.traineddata: https://github.com/tesseract-ocr/tessdata_fast/blob/master/dan.traineddata
    # fraktur.traineddata can be downloaded from tessdata_best:
    # https://github.com/tesseract-ocr/tessdata_best/blob/master/script/Fraktur.traineddata
    traineddata_labels = ['fraktur', 'dan', 'frk']
    tess_outdirs = [os.path.join(pth.intermediate, f'tess_out_{label}') for label in traineddata_labels]
    uncorrected_dir = os.path.join(pth.intermediate, conf['base_ocr'])
    corrected_dir = os.path.join(pth.intermediate, param_str)

    # Steps of the pipeline. Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf['metadir'])
    if conf.getboolean('run_pdf2img'):
        pdf_paths = [f for f in os.listdir(conf['inputdir']) if f.endswith('.pdf')]
        pdfs2imgs(pdf_paths, pth.img_dir, int(conf['split_size']))
    if conf.getboolean('run_ocr'):
        do_ocr(pth.img_dir, pth.intermediate, traineddata_labels)
    if conf.getboolean('correct_easy'):
        correct_easy_fraktur_errors(uncorrected_dir, corrected_dir)
        uncorrected_dir = corrected_dir
    if conf.getboolean('correct_hard'):
        correct_hard_fraktur_errors(uncorrected_dir, pth.intermediate, corrected_dir)
        uncorrected_dir = corrected_dir
    if conf.getboolean('sym_wordcorrect'):
        sym_wordcorrect(conf, uncorrected_dir, corrected_dir)
    # TODO Will it make any sense to employ SymSpell at the bigram level? Probably not?
    if conf.getboolean('make_basic_gold_vrt'):
        gold_vrt_gen = generate_novels_vrt(pth.gold_novels_dir, pth.corp_label)
        write_novels_vrt(gold_vrt_gen, pth.basic_gold_vrt_path)
    if conf.getboolean('annotate_gold_vrt'):
        text_annotation_generator = generate_gold_annotations(pth.basic_gold_vrt_path, pth.ocr_kb_dir,
                                                              conf['texton_out_dir'], pth.corp_label, tess_outdirs,
                                                              [corrected_dir], conf)  # TODO single dir instead of list of dirs?
        write_annotated_gold_vrt(text_annotation_generator, pth.local_annotated_gold_vrt_path)
        shutil.copy(pth.local_annotated_gold_vrt_path, pth.annotated_gold_vrt_path)
    if conf.getboolean('analyze_errors'):
        # TODO Not very transparent error when n_datasets is wrong.
        analyze_gold_vrt(pth.annotated_gold_vrt_path, conf, pth.analyses_dir, param_str, n_datasets=5)
    if conf.getboolean('write_korp_configs'):
        util.write_frakturgold_mode(conf['frakturgold_mode_template'],
                                    conf['gold_vrt_p_attrs'],
                                    conf['frakturgold_mode_outpath'])
        shutil.copy(conf['frakturgold_mode_outpath'], os.path.join(pth.vrt_dir, 'memo_frakturgold_mode.js'))
        util.write_frakturgold_encodescript(conf['frakturgold_encode_template'],
                                            pth.annotated_outdir,
                                            conf['gold_vrt_p_attrs'],
                                            conf['frakturgold_encode_outpath'])
        shutil.copy(conf['frakturgold_encode_outpath'], os.path.join(pth.vrt_dir, 'encode_MEMO_fraktur_gold.sh'))
    if conf.getboolean('write_word'):
        pass

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
