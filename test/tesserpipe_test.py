import configparser
import os
import tempfile
import unittest
import parameterized

from pdf2image import convert_from_path, convert_from_bytes
from memoocr.align_ocr import align_ocr

"""
- Test whether dependencies are installed (tesseract, poppler).
- Test whether state can be saved.
- Test whether saved state can be recovered.
X Test whether PDF can be split into images.
- Test whether image can be OCR'ed.
- Test whether OCR can be cleaned.
"""


class alignOCRTest(unittest.TestCase):
    """Test alignment of original OCR lines with corrected lines."""

    @parameterized.parameterized.expand([
        ('„Hr. E ta tsra a d Helmer, tlieoloZios',
         '„Hr. Etatsraad Helmer, Candidatus theologiæ'),
        ('langt mere oedel og ridderlig Teenkemaade. ^evnne sti',
         'langt mere ædel og ridderlig Tænkemaade. Senere fik'),
        ('Dronning at syre hende og hendes ^Lyn E a rl t il Flandern,',
         'Dronning at føre hende og hendes Søn Earl til Flandern,'),
        ('Vcegten af de Tanker der gennemfarer d e t... hans',
         'Vægten af de Tanker der gennemfarer det. . . hans'),
        ('sine klare j2jne ind i hans.',
         'sine klare Øjne ind i hans.'),
        ('Ord et „passende" -havde i F r u R a n d u ls M u n d',
         'Ordet „passende" havde i Fru Randuls Mund'),
        ('da Afgiften t i l Herremanden heller ikke var uoverkommelig,',
         'da Afgiften til Herremanden heller ikke var uover[- ]kommelig,')
    ])
    def test_align_len(self, orig, corr):
        alignment = align_ocr(orig, corr)
        origtup = alignment.aligned_orig
        corrtup = alignment.correct
        self.assertTrue(len(origtup) == len(corrtup))


class getConfigsTest(unittest.TestCase):
    """Test the configuration option handling."""

    def test_get_testinput_path(self):
        pathresult = get_config_path('data', 'testinputdir')
        self.assertTrue(os.path.exists(pathresult))


class pdf2ImgTest(unittest.TestCase):
    def test_convert_from_path(self):
        """Make sure convert_from_path returns a list."""
        filepath = '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/testinput/1883_Brosboell_Krybskytten.pdf'
        with tempfile.TemporaryDirectory() as outpath:
            img_list = convert_from_path(filepath,
                                         thread_count=4,
                                         dpi=300,
                                         output_folder=outpath,
                                         fmt='jpeg')
        [img.close() for img in img_list]
        self.assertTrue(isinstance(img_list, list))
