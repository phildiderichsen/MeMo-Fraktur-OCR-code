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

NOTE: In the gold standard (`Guldstandard.txt`), hypens ("[- ]") are removed.

Run:

Create a config.ini with relevant options and local paths (see `config.ini.example.txt`). Then:

```
python3 analyze_goldstandard.py
```



## OCR correction

Work in progress ...

Take PDFs of Fraktur novels through an OCR pipeline from PDF to cleaned OCR text.

- Split PDFs into separate pages and save each page as jpeg.
- Perform OCR with tesseract.
- Clean OCR from tesseract using regex/SymSpell/character n-gram embeddings?
  - The cleaned output is to be tokenized and used as the token layer for all subsequent annotations. The original OCR text is to be aligned to this token layer for later reference.

The different steps can be bypassed as per the choices in config.ini.

Run:

Create a config.ini with relevant options and local paths (see `config.ini.example.txt`). Then:

```
python3 run_ocr_pipeline.py
```


## VRT output

The code should be able to work with .vrt files.

- When correcting OCR, the corrected output should become the reference token layer while the original should become an annotation layer.
- When working with the gold standard, the gold standard should be the reference token layer and the (corrected) OCR should be aligned to it as an annotation layer.
- It should be possible to read in an existing .vrt file and add annotations to it. E.g.: 
  - An existing .vrt file with only reference tokens to which an original OCR layer is aligned and added.
  - An existing .vrt file with reference tokens + original OCR to which another layer of original (or corrected) OCR is added.
- .vrt files should have one novel per `<text>` element, and page numbers as an annotation layer.


## Add annotations from CONLL file

Annotations (including <sentence> segmentation) from a CONLL file per novel can now be added.

See my OneNote and the README for Text Tonsorium output set 502-final (in Downloads) for notes.


## Notes on different frequency lists

I experimented with a number of frequency lists.

- freqs1 = frequency_dict_da_sm.txt (Ross' original unigram list. Common Crawl data?)
- freqs2 = meta/unigrams_brandes_adl.txt (Dorte's unigram freqlist from the Brandes texts and ADL texts in 'Træningskorpus, april 2020')
- freqs3 = unigrams_brandes_adl_ods.txt (freqs2 plus tokens from ODS with freq = 1 if they are not on the freqs2 list)
- freqs4 = unigrams_brandes_adl_da.txt (freqs2, *not lowercased*, plus tokens from freqs1 if they are not on the freqs2 list)
- freqs5 = unigrams_brandes_adl_da_sm.txt (freqs2, *lowercased*, plus tokens from freqs1 if they are not on the freqs2 list)
- freqs6 = unigrams_brandes_adl_da_sm_aa.txt (like freqs5, but with 'å' replaced by 'aa').
- freqs7 = unigrams_brandes_ods_adl_da_sm_aa.txt (like freqs6, but with ODS tokens added (with count = 1) if they are not on freqs6).
- freqs8 = unigrams_brandes_ods6_adl_da_sm_aa.txt (like freqs7, but only ODS tokens at least 6 chars long are added).
- Freqs9 = unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered.txt (ADL words with counts adjusted to match Ross' list; ODS words added if not in ADL; Ross list added, but only words in ODS. Inspired by the original SymSpell docs: https://github.com/wolfgarbe/SymSpell#frequency-dictionary: "The frequency_dictionary_en_82_765.txt was created by intersecting the two lists mentioned below (1. Google Books Ngram data and 2. SCOWL - Spell Checker Oriented Word Lists). By reciprocally filtering only those words which appear in both lists are used. Additional filters were applied and the resulting list truncated to ≈ 80,000 most frequent words.".)
- bifreqs1 = bigrams_dict_da_sm.txt (Ross' original bigram list. Common Crawl Data?)
- bifreqs2 = bigrams_brandes_adl.txt (Dorte's bigram freqlist from the Brandes texts and ADL texts in 'Træningskorpus, april 2020')
- bifreqs3 = bigrams_brandes_adlx10.txt (like bifreqs2, but frequencies multiplied by 10)

Observations:

- Freqs9 - with ODS data, and with Ross' frequency list filtered on ODS - performs best.
- Freqs5 performs a tiny bit better than freqs4, which has a slightly lower match percentage, and slightly more error types. SymSpell docs state that SymSpell expects a lowercased frequency dictionary.
- Using bifreqs1, bifreqs2, or bifreqs3 makes no difference. It also makes no difference to omit the bigram frequencies altogether. So they will be omitted. HOWEVER, this may well be because only word corrections are done at this point. Which of course makes bigrams irrelevant. Correction of longer stretches of text should be explored (e.g. crudely segmented sentences). 
- Freqs2 has the best performance on the most frequent errors (e.g. only half the ø=o errors compared to freqs1). However, overall performance is far worse than freqs1. (Also, there are many more error types).
- Freqs6 (replacing 'å' with 'aa') gives a tiny improvement.
- Freqs8 (limiting ODS tokens to longish words) makes no real difference - if anything, a tiny improvement in the most frequent error types.
- How to best combine freqs2's good performance on frequent errors with freqs1's overall good performance? Freqs7 is the current best option, but why ...

Code used to generate combined frequency dict (the code shown generates freqs9):

```python
from collections import defaultdict

rossfreq = '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/frequency_dict_da_sm.txt'
adlfreq = '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/unigrams_brandes_adl.txt'
odsfile = '/Users/phb514/Downloads/ods.freq1.txt'

with open(rossfreq, 'r') as rf:
    rflines = rf.read().splitlines()
with open(adlfreq, 'r') as adl:
    adllines = adl.read().splitlines()
with open(odsfile, 'r') as ods:
    odslines = ods.read().splitlines()

"""
Hvad skal der ske?
Ross' frekvensliste skal gælde hvis ordet ikke findes i ADL.
Hvis ordet findes i ADL, skal det indsættes i stedet, med frekvensen korrigeret op i samme størrelsesorden som Ross' liste.
"""


odstuples = [line.split() for line in odslines]
odsdict = defaultdict(int)
for token, freq in odstuples:
    if len(token) < 4:
        continue
    token = token.lower().replace('å', 'aa')
    odsdict[token] += int(freq)



adltuples = [line.split() for line in adllines]
adldict = defaultdict(int)
for token, freq in adltuples:
    # adldict[token] += int(freq) * 806
    token = token.lower().replace('å', 'aa')
    adldict[token] += int(freq) * 806

print('len(adldict.items()):', len(adldict.items()))
print("adldict['og']:", adldict['og'])

print('38330072 / 47611:', 38330072 / 47611)

odsplusadl = odsdict.copy()
odsplusadl.update(adldict)


rosstuples = [line.split() for line in rflines]
rossdict = defaultdict(int)
for token, freq in rosstuples:
    token = token.lower().replace('å', 'aa')
    rossdict[token] += int(freq)

print('len(rossdict.items()):', len(rossdict.items()))
print("rossdict['og']:", rossdict['og'])





combineddict = odsplusadl.copy()
combineddict.update(odsplusadl)
print('len(combineddict.items()):', len(combineddict.items()))


# Remove words that are not in ODS + ADL
odsadl_filtered_combineddict = [(k, v) for k, v in combineddict.items() if k in odsplusadl]

freq_token_tuples = [(v,k) for k,v in odsadl_filtered_combineddict]
freq_token_tuples.sort(reverse=True)

freqlist = [f'{k} {v}' for v, k in freq_token_tuples]
print(freqlist[:20])

with open('/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered.txt', 'w') as out:
    out.write('\n'.join(freqlist))
```


## Notes on correcting 