import configparser
import sys
from datetime import datetime
import os
# Image processing
try:
    from PIL import Image
except ImportError:
    import Image
from pdf2image import convert_from_path
# TODO: Are these necessary?
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
# Data flow
import tempfile
import multiprocessing as mp
from myutils.novelnames_startpages import startpage_dict
from myutils.novelnames_endpages import endpage_dict
import myutils as util

Image.MAX_IMAGE_PIXELS = None  # otherwise it thinks it's a bomb


def chunked(lst, img_dir, n):
    """
    Yield successive n-sized chunks from lst in order to prevent memory trouble.
    Include inputdir and img_dir (for output) with each novel in chunk.
    """
    for i in range(0, len(lst), n):
        chunklist = lst[i:i + n]
        yield list(zip(chunklist, [img_dir] * len(chunklist)))


def process(novel_config_tuple):
    """
    Main extraction pipeline

    - It's unclear why but thread_count > 4 causes bottleneck
    - DPI 300 is best quality ratio; higher DPI causes poorer OCR!
    - Format can be set to PNG but it's *much* slower
    """
    filepath, img_dir = novel_config_tuple
    novel = os.path.basename(filepath)
    print(f"Working on {novel}")
    # Create tempfile for images
    with tempfile.TemporaryDirectory() as path:
        print("...converting pdf...")
        print('filepath:', filepath)
        images_from_path = convert_from_path(filepath,
                                             thread_count=4,
                                             dpi=300,
                                             output_folder=path,
                                             fmt='jpeg')

        # Save images from path to individual files
        print(f"...saving images for {novel}...")
        novelname = novel.replace('.pdf', '')
        outfolder = os.path.join(img_dir, novelname)
        # Get the page numbers where the actual novel starts and ends.
        startenddict = util.make_startend_dict()
        startpage = int(startenddict[novelname]['start'])
        # Set page counter
        i = 1
        for image in images_from_path:
            # Skip ahead to actual novel start page. Note: This will obviously only work with the full PDF!
            if i < startpage:
                print(f'Skipping ahead to novel start page (p. {startpage}) ..')
                i += 1
                continue
            # Skip pages after end of novel. TODO: End page data currently cannot be trusted.
            # if i > endpage:
            #     break
            outpath = os.path.join(outfolder, f"page_{i}.jpeg")
            with open(outpath, "w") as out:
                image.save(out)
            image.close()
            i += 1


def pdfs2imgs(pdf_paths, img_dir, split_size):
    """
    Main pipe line to convert PDF to JPEG
    """
    # Check if all PDF names are in the start- and end page dicts.
    if (not all([os.path.basename(pth).replace('.pdf', '') in startpage_dict for pth in pdf_paths])) or \
            (not all([os.path.basename(pth).replace('.pdf', '') in endpage_dict for pth in pdf_paths])):
        print('PDFs not in startpage_dict:')
        print('\n'.join([os.path.basename(pth) for pth in pdf_paths if
                         not os.path.basename(pth).replace('.pdf', '') in startpage_dict]))
        print()
        print('PDFs not in endpage_dict:')
        print('\n'.join([os.path.basename(pth) for pth in pdf_paths if
                         not os.path.basename(pth).replace('.pdf', '') in endpage_dict]))
        print()
        sys.exit('Fix startpage_dict and/or endpage_dict ...')
    try:
        os.makedirs(img_dir)
    except FileExistsError:
        print(f"{img_dir} already exists")
    for novel in pdf_paths:
        outfolder = os.path.join(img_dir, os.path.basename(novel).replace('.pdf', ''))
        try:
            os.mkdir(outfolder)
        except FileExistsError:
            print(f"{outfolder} already exists")

    # Process in chunks equal to the split_size set in options.
    chunks = list(chunked(pdf_paths, img_dir, split_size))
    # Start chunk count at 1
    count = 1
    n_processes = mp.cpu_count() - 2 if mp.cpu_count() > 2 else mp.cpu_count()
    pool = mp.Pool(processes=n_processes)
    # Process in parallel, one chunk at a time to avoid memory leak
    for chunk in chunks:
        print(f"Chunk {count} of {len(chunks)}")
        pool.map(process, chunk)
        count += 1
    pool.close()
    pool.join()


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))

    conf = config['DEFAULT']
    pdfs2imgs(conf['inputdir'], os.path.join(conf['intermediatedir'], '1-imgs'), int(conf['split_size']))

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
