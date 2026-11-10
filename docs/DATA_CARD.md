# Data card: ObliQA, as used in this repo

| | |
|---|---|
| Name | ObliQA (Obligation based Question Answering) |
| Source | [github.com/RegNLP/ObliQADataset](https://github.com/RegNLP/ObliQADataset), pinned to commit `c1b2c3c04aee` (8 Dec 2024) |
| Paper | Tuba Gokhan, Kexin Wang, Iryna Gurevych, Ted Briscoe. *RIRAG: Regulatory Information Retrieval and Answer Generation.* [arXiv:2409.05677](https://arxiv.org/abs/2409.05677), 2024 |
| License | None stated. The source repo has no license file. |
| How this repo uses it | Downloaded by `regrag download` from the source repo. No copy lives in this repo; only aggregate metrics are committed. |
| Domain | Financial regulation in Abu Dhabi Global Market (ADGM), a financial free zone in the UAE |
| Language | English |
| Personal data | None. Public regulatory text and generated questions. |

## License position

With no license file, the default is that no reuse rights are granted. So this repo:

- does not include any ObliQA file, passage or question, and links to the source instead;
- downloads the files to your own machine only after you accept a notice (`regrag download`);
- commits only aggregate metrics computed from the data;
- uses a tiny, made up rulebook for tests and CI.

Cite the paper if you use the data. For anything beyond research, contact the authors (the dataset README lists regnlp2025@gmail.com).

## Composition

Counts in this section come from `uv run regrag stats`, which counts words by splitting text on whitespace. Figures about how the data was built are quoted from the authors.

| Part | Count |
|---|---:|
| Regulatory documents | 40 (about 836,000 words) |
| Passages in the structured documents | 13,732 |
| Empty passages (section breaks), dropped | 720 |
| Passages in the corpus used here | 13,012 |
| Train questions | 22,295 rows, 22,291 unique ids |
| Dev questions | 2,788 |
| Test questions | 2,786 |

Gold passages per question:

| Split | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Dev | 2,115 | 514 | 116 | 30 | 12 | 1 |
| Test | 2,126 | 506 | 105 | 36 | 9 | 4 |

In the test split, 221 questions have gold passages in more than one document and 354 cite a rule by its number. In the train split, 2,668 of the 22,291 questions cite a rule by its number. The median question has 27 words. The median passage has 42 words, the 95th percentile 163, and the longest (a definitions table) 24,312.

## How the authors built it

From the dataset README and paper:

1. ADGM documents (Word files) were structured by hand into passages, each with a rule reference. Tables were tagged and figures turned into text.
2. Questions were generated with `gpt-4-turbo-1106`, both from single passages and from groups of related passages chosen by topic (AML, virtual assets, audit and so on).
3. Each question and passage pair was checked with an NLI model (`nli-deberta-v3-xsmall`) and contradictions were removed.
4. Experts reviewed 991 random questions with 1,847 gold passages: 86.9% of passages were relevant, 9.1% relevant but not a direct answer, 3.5% irrelevant and 0.4% null.

## Processing in this repo

- **Download:** 43 files from the pinned commit (3 question files and 40 documents), or 42 with `--skip-train`. Each file's SHA-256 is checked against `src/regrag/_checksums.py`, and with `--skip-train` a train file that is already there is still checked. Only when every file has passed does `regrag download` write `MANIFEST.json` with the source repo and commit. Every command that loads the data refuses a folder without it, so a partial download is never searched, and each saved result copies the manifest as its dataset record.
- **Corpus:** every non empty passage. None of the 720 empty passages is a gold passage.
- **Gold mapping:** questions point at passages by document number and rule reference. That pair is unique except for 15 references shared by 39 passages; those ties are broken by exact passage text. Every gold reference in all three splits maps to exactly one corpus passage.
- **Question ids:** the train file has 22,295 rows but 22,291 unique question ids; the loader keeps the first row for each id. Dev and test have no repeats.

## Known quirks

- The dataset README lists 2,888 dev questions. The dev file has 2,788.
- One dev question (Doc 10, 4.7.7) and one test question (Doc 1, 2.Guidance.27.) have a broken copy of that gold passage in the question file: the first 13 (dev) or 16 (test) words of the corpus text, one garbled word, then the end of a different passage from Doc 19 (dev) or Doc 12 (test). The other 17 dev and 26 test questions that cite these passages have the exact corpus text, and the references still map to one passage.
- 219 texts appear in more than one passage (650 passages in all). BM25 gives such twins the same score, so only the tie order decides which of them ranks first. That matters where a gold passage has a twin that is not gold for the same question: 13 gold references in 12 test questions, and 13 in 11 dev questions. Ties are broken the way trec_eval breaks them, by passage id in descending order.
- Very long passages are mostly tables. The dense model reads only their first 512 tokens; BM25 reads all of them.
- The hidden shared task test set (446 questions) is not public, so the published shared task leaderboard is not directly comparable with results on the test split.

## Risks for anyone using it

- The questions were generated from the gold passages, so they reuse their wording. Real compliance questions are messier.
- The labels are incomplete: a correct passage that was not used to generate a question counts as wrong.
- The rules are a snapshot from around 2024 and ADGM updates its rulebooks. Do not treat the text as current law.
