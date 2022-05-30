"""
run_ocr_pipeline.py
Run OCR correction pipeline on full novel data.
"""
import os
import pathlib
import shutil
import myutils as util

from datetime import datetime
from memoocr.make_dictionary import make_dic
from memoocr.pdf2img import pdfs2imgs
from memoocr.ocr import do_ocr
from memoocr.annotate_corr_vrt import generate_corr_annotations, write_annotated_corr_vrt
from memoocr.pages2singlelinefiles import pages2singlelinefiles
from memoocr.make_corpus_vrt import generate_novels_vrt_from_text, write_novels_vrt, add_metadata
from memoocr.correct_ocr import sym_wordcorrect, correct_easy_fraktur_errors, correct_hard_fraktur_errors
from memoocr.make_year_vrts import write_year_vrts


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = util.get_config()

    # Generate various paths and create them if necessary.
    conf = util.Confs(config).corrconf
    *_, param_str = util.get_params(conf)
    pth = util.CorrPaths(conf)
    util.clean_datadirs(pth.fulloutputdir)
    if conf.getboolean('precheck'):
        util.precheck_datadirs(pth)
    metadata = util.make_metadata_dict(pth)

    # Discard files that should be discarded or quarantined.
    files_to_process = pth.files_to_process.copy()
    for name in metadata:
        if metadata[name]['discard'] or metadata[name]['quarantine']:
            pdf_name = name + '.pdf'
            if pdf_name in files_to_process:
                files_to_process.remove(pdf_name)

    # Which OCR traineddata should be used?
    # Note! frk.traineddata must be downloaded from tessdata_fast in order to work:
    # https://github.com/tesseract-ocr/tessdata_fast/blob/master/frk.traineddata
    # Same for dan.traineddata: https://github.com/tesseract-ocr/tessdata_fast/blob/master/dan.traineddata
    # fraktur.traineddata can be downloaded from tessdata_best:
    # https://github.com/tesseract-ocr/tessdata_best/blob/master/script/Fraktur.traineddata
    # traineddata_labels = ['Fraktur', 'dan', 'frk']
    traineddata_labels = ['dan']
    tess_outdirs = [os.path.join(pth.fulloutputdir, f'tess_out_{label}') for label in traineddata_labels]
    uncorrected_dir = os.path.join(pth.fulloutputdir, conf['base_ocr'])

    # Hack to remove front and back matter from KB singlefiles
    if pathlib.Path(uncorrected_dir).name == 'orig_pages':
        uncorrected_dir = util.remove_kb_frontmatter(uncorrected_dir, metadata)
    new_base_ocr_dir = uncorrected_dir
    corrected_dir = os.path.join(pth.fulloutputdir, param_str)

    # Steps of the pipeline. Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf['metadir'])
    if conf.getboolean('run_pdf2img'):
        pdf_paths = [os.path.join(conf['pdf_dir'], f) for f in files_to_process]
        pdfs2imgs(pdf_paths, pth.img_dir, metadata, int(conf['split_size']))
    if conf.getboolean('run_ocr'):
        print('Running run_ocr ...\n')
        img_dirs = [os.path.join(pth.img_dir, f.replace('.pdf', '')) for f in files_to_process]
        do_ocr(img_dirs, pth.fulloutputdir, traineddata_labels)
    if conf.getboolean('correct_easy'):
        print('Running correct_easy ...\n')
        correct_easy_fraktur_errors(files_to_process, uncorrected_dir, corrected_dir)
        uncorrected_dir = corrected_dir
    if conf.getboolean('correct_hard'):
        print('Running correct_hard ...\n')
        correct_hard_fraktur_errors(files_to_process, uncorrected_dir, pth.fulloutputdir, corrected_dir)
        uncorrected_dir = corrected_dir
    if conf.getboolean('sym_wordcorrect'):
        print('Running sym_wordcorrect ...\n')
        sym_wordcorrect(files_to_process, conf, uncorrected_dir, corrected_dir)
    if conf.getboolean('make_singleline_novel_textfiles'):
        pages2singlelinefiles(files_to_process, corrected_dir, pth.singleline_dir, metadata)
    if conf.getboolean('make_basic_corr_vrt'):
        print('corrected_dir:', corrected_dir)
        print('param_dir:', os.path.join(pth.fulloutputdir, param_str))
        # TODO Fix code so that 'corrected_dir' does not have to be hardcoded because it depends on previous steps ..
        cordir = corrected_dir
        #cordir = '/Users/phb514/mygit/MeMo-Fraktur-OCR-code/intermediate/2022-05-12/fulloutput/pages_freqs12_correasy_corrhard_symwordcorr'
        gold_vrt_gen = generate_novels_vrt_from_text(files_to_process, cordir, conf['fraktur_vrt_label'], metadata)
        write_novels_vrt(gold_vrt_gen, pth.basic_gold_vrt_path)
    if conf.getboolean('annotate_corr_vrt'):
        text_annotation_generator = generate_corr_annotations(pth.basic_gold_vrt_path, new_base_ocr_dir,
                                                              conf['texton_out_dir'], pth.corp_label, metadata)
        write_annotated_corr_vrt(text_annotation_generator, pth.local_annotated_gold_vrt_path)
        # shutil.copy(pth.local_annotated_gold_vrt_path, pth.annotated_gold_vrt_path)
    if conf.getboolean('add_metadata'):
        # TODO Notice that we are now bypassing the annotation step.
        #  vrt_with_metadata = add_metadata(pth.local_annotated_gold_vrt_path, metadata)
        vrt_with_metadata = add_metadata(pth.basic_gold_vrt_path, metadata)
        with open(pth.local_annotated_gold_vrt_path.replace('.vrt', '.meta.vrt'), 'w') as f:
            f.write(vrt_with_metadata)
    if conf.getboolean('make_yearcorpora'):
        write_year_vrts(pth.local_annotated_gold_vrt_path, conf['yearcorp_outdir'])

        # TODO frakturcorr_mode_template mangler ..?? Brug /opt/corpora/infrastructure/korp/setups/clarin/frontend/app/modes/memo_yearcorpora_mode.js som model ..
        util.write_frakturgold_mode(conf['frakturcorr_mode_template'],
                                    conf['corr_vrt_p_attrs'],
                                    conf['frakturcorr_mode_outpath'])
        util.write_frakturcorr_encodescript(conf['frakturcorr_encode_template'],
                                            conf['corr_vrt_p_attrs'],
                                            conf['frakturcorr_encode_outpath'])

    if conf.getboolean('export_corpora'):
        modefile = os.path.basename(conf['frakturcorr_mode_outpath'])
        modedest = os.path.join(conf['korp_setup_dir'], 'frontend', 'app', 'modes', modefile)
        encodefile = os.path.basename(conf['frakturcorr_encode_outpath'])
        encodedest = os.path.join(conf['korp_setup_dir'], 'corpora', 'encodingscripts', encodefile)
        corporadest = os.path.join(conf['korp_setup_dir'], 'corpora', 'annotated', 'memo_fraktur_corr')
        print(modedest)
        print(encodedest)
        print(conf['yearcorp_outdir'])
        print(corporadest)

        shutil.copy(conf['frakturcorr_mode_outpath'], modedest)
        shutil.copy(conf['frakturcorr_encode_outpath'], encodedest)
        shutil.copytree(conf['yearcorp_outdir'], corporadest, dirs_exist_ok=True)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
