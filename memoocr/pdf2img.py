import configparser
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

Image.MAX_IMAGE_PIXELS = None  # otherwise it thinks it's a bomb


def chunked(lst, inputdir, img_dir, n):
    """
    Yield successive n-sized chunks from lst in order to prevent memory trouble.
    Include inputdir and img_dir (for output) with each novel in chunk.
    """
    for i in range(0, len(lst), n):
        chunklist = lst[i:i + n]
        yield list(zip(chunklist, [inputdir] * len(chunklist), [img_dir] * len(chunklist)))


def process(novel_config_tuple):
    """
    Main extraction pipeline

    - It's unclear why but thread_count > 4 causes bottleneck
    - DPI 300 is best quality ratio; higher DPI causes poorer OCR!
    - Format can be set to PNG but it's *much* slower
    """
    novel, inputdir, img_dir = novel_config_tuple
    print(f"Working on {novel}")
    # get filepath
    filepath = os.path.join(inputdir, novel)
    # Create tempfile for images
    with tempfile.TemporaryDirectory() as path:
        print("...converting pdf...")
        images_from_path = convert_from_path(filepath,
                                             thread_count=4,
                                             dpi=300,
                                             output_folder=path,
                                             fmt='jpeg')

        # Save images from path to individual files
        print(f"...saving images for {novel}...")

        # TODO: These should be defined in ONE place ..
        name = novel.replace('.pdf', '')
        outfolder = os.path.join(img_dir, name)
        # Set page counter
        i = 1
        for image in images_from_path:
            outpath = os.path.join(outfolder, f"page_{i}.jpeg")
            with open(outpath, "w") as out:
                image.save(out)
            image.close()
            i += 1


def pdfs2imgs(inputdir, img_dir, split_size):
    """
    Main pipe line to convert PDF to JPEG
    """
    # Make list of PDFs in inputdir.
    pdf_paths = [f for f in os.listdir(inputdir) if f.endswith('.pdf')]
    try:
        os.makedirs(img_dir)
    except FileExistsError:
        print(f"{img_dir} already exists")
    for novel in pdf_paths:
        outfolder = os.path.join(img_dir, novel.replace('.pdf', ''))
        try:
            os.mkdir(outfolder)
        except FileExistsError:
            print(f"{outfolder} already exists")

    # Process in chunks equal to the split_size set in options.
    chunks = list(chunked(pdf_paths, inputdir, img_dir, split_size))
    print(chunks)
    # Start chunk count at 1
    count = 1
    # Processes equal to split_size; so one CPU per novel
    pool = mp.Pool(processes=split_size)
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
