# MeMo-Fraktur-OCR-code

Developing a rule-based procedure for correcting OCR data from 19th century fraktur novels.

There are a few different goals in this project:

- Error analysis of baseline OCR output from KB (The Royal Library), as well as OCR output from other sources (= Tesseract).
- Re-OCR'ing the PDFs from KB using Tesseract (possibly using several different models).
- Correcting OCR using regex replacement, spelling error detection, and possibly custom character level n-gram embedding similarity.
- Error analysis of the corrected OCR vs. the baseline.



## Project structure

```
MeMo-Fraktur-OCR-code
├── README.md
├── analyze_goldstandard.py                   # Script: Run error analysis
├── config.ini                                # Local settings (git-ignored)
├── config.ini.example.txt                    # Example - save as config.ini
├── eval_df.csv                               # Alignment cache (git-ignored)
├── evalocr                                   # Py package: Alignment
│   ├── __init__.py
│   └── align_ocr.py
├── intermediate                              # Interim output (git-ignored)
│   └── 12-02-2021
│       ├── 1-imgs
│       │   └── 1883_Brosboell_Krybskytten
│       │       ├── page_1.jpeg
│       │       ├── ..
│       │       └── page_9.jpeg
│       ├── 2-uncorrected
│       │   └── 1883_Brosboell_Krybskytten
│       │       ├── page_10_uncorrected.txt
│       │       ├── ..
│       │       └── page_9_uncorrected.txt
│       └── 3-corrected
│           └── 1883_Brosboell_Krybskytten
│               └── 1883_Brosboell_Krybskytten_corrected.txt
├── memoocr                                   # Py package: OCR + corrections
│   ├── __init__.py
│   ├── correct_ocr.py
│   ├── make_dictionary.py
│   ├── ocr.py
│   └── pdf2img.py
├── requirements.txt                          # Python requirements
├── run_ocr_pipeline.py                       # Script: Run OCR/corrections
└── test                                      # Unittests
    ├── __init__.py
    ├── analyze_gold_test.py
    └── tesserpipe_test.py
```

## Installation/dependencies

Code developed in python 3.9 in venv (virtual environment).

- Poppler
    - Installed in `/usr/local/Cellar/poppler/21.02.0` using `brew install poppler`.
- Tesseract 4 
    - Installed in `/usr/local/Cellar/tesseract/4.1.1/bin/tesseract` using `brew install tesseract`
- Python modules: See requirements.txt. Install in your virtual environment using `pip install -r requirements.txt`


## OCR analysis

Token-align OCR output with corrected goldstandard, and generate detailed error statistics.

Run:

Create a config.ini with relevant options and local paths (see `config.ini.example.txt`). Then:

```
python3 analyze_goldstandard.py
```



## OCR correction

Work in progress ...

Take PDFs of Fraktur novels through an OCR pipeline from PDF to cleaned OCR text.

- Split PDFs into separate pages.
- Perform OCR with tesseract.
- Clean OCR from tesseract using regex/SymSpell/character n-gram embeddings?

The different steps can be bypassed as per the choices in config.ini.

Run:

Create a config.ini with relevant options and local paths (see `config.ini.example.txt`). Then:

```
python3 run_ocr_pipeline.py
```
