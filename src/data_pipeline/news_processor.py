"""FinBERT-MRC event extraction from financial news articles."""
import pandas as pd
import torch
import gc
from transformers import pipeline
from tqdm import tqdm
import os
import sys

# Add src to path if needed for relative imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

class NewsProcessor:
    """
    Processes filtered news headlines through 3 NLP stages:
      1. Sentiment Analysis  (FinBERT-tone)
      2. Event Classification (BART-large-MNLI zero-shot)
      3. MRC Entity Extraction (FinBERT-MRC / FinancialBERT-SQuAD2)

    Models are loaded ONE AT A TIME to stay within GPU memory limits.
    After each stage the model is explicitly deleted and VRAM is freed.
    """

    def __init__(self, data_path, output_path, tickers_path, device=None):
        self.data_path = data_path
        self.output_path = output_path
        self.tickers_path = tickers_path

        # ---- Device selection ------------------------------------------------
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1
        print(f"Using device: {'cuda:0' if self.device == 0 else 'cpu'} for inference")

        # ---- Entity Linker (lightweight, kept in RAM the whole time) ---------
        self.linker = None
        try:
            with open(tickers_path, 'r') as f:
                self.tickers = [t.strip() for t in f.read().strip().split(',')]
            try:
                from src.data_pipeline.entity_linker import EntityLinker
            except ImportError:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from entity_linker import EntityLinker
            self.linker = EntityLinker(self.tickers)
            print(f"Entity Linker initialized with {len(self.tickers)} tickers.")
        except Exception as e:
            print(f"[ERROR] Entity Linker init failed: {e}")

        # ---- Candidate labels for zero-shot event classification -------------
        self.candidate_labels = [
            "Mergers and Acquisitions",
            "Earnings Report",
            "Stock Movement",
            "Legal and Regulation",
            "Product Launch",
            "Macroeconomic",
            "Analyst Rating",
            "Supply Chain"
        ]

        # ---- MRC Queries per Event Type --------------------------------------
        self.mrc_queries = {
            "Mergers and Acquisitions": [
                ("Who is the acquirer?", "entity_a"),
                ("Who is being acquired?", "entity_b")
            ],
            "Earnings Report": [
                ("Which company reported earnings?", "entity_a")
            ],
            "Legal and Regulation": [
                ("Which company is sued or fined?", "entity_a"),
                ("Who is the plaintiff?", "entity_b")
            ],
            "Product Launch": [
                ("Which company launched a product?", "entity_a")
            ],
            "Stock Movement": [
                ("Which stock moved?", "entity_a")
            ],
            "Analyst Rating": [
                ("Which stock was rated?", "entity_a")
            ],
            "Supply Chain": [
                ("Which supplier is directly involved?", "entity_a"),
                ("Who is the key customer?", "entity_b")
            ]
        }

        # ---- Resolve custom MRC model path -----------------------------------
        custom_model_path = os.path.join(BASE_DIR, "models", "finbert_mrc_custom")
        if os.path.exists(custom_model_path) and os.path.exists(
                os.path.join(custom_model_path, "config.json")):
            self.qa_model_name = custom_model_path
            print(f"Will use Custom Fine-Tuned FinBERT-MRC at {custom_model_path}")
        else:
            self.qa_model_name = "ahmedrachid/FinancialBERT-SQuAD2"
            print(f"Custom model not found. Will use Pre-Trained {self.qa_model_name}")

    # ------------------------------------------------------------------
    # Helper: free GPU memory between stages
    # ------------------------------------------------------------------
    @staticmethod
    def _free_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 1 – Sentiment
    # ------------------------------------------------------------------
    def _run_sentiment(self, headlines):
        """Returns list of dicts with keys: label, score"""
        print("\n===== Stage 1/3: Sentiment Analysis (FinBERT-tone) =====")
        from transformers import BertForSequenceClassification, BertTokenizer
        
        # Explicitly load to avoid "Unrecognized model" error due to missing model_type in HF config
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            truncation=True,
            max_length=64,
            framework="pt"
        )

        sentiments = []
        batch_size = 16          # safe for 6 GB
        errors = 0
        for i in tqdm(range(0, len(headlines), batch_size), desc="Sentiment"):
            batch = headlines[i:i + batch_size]
            try:
                results = pipe(batch)
                sentiments.extend(results)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [WARN] Sentiment batch {i//batch_size} error: {e}")
                sentiments.extend([{'label': 'Neutral', 'score': 0.0}] * len(batch))

        # Free model
        del pipe
        self._free_gpu()
        print(f"  Sentiment done. {errors} batch errors out of "
              f"{(len(headlines) + batch_size - 1) // batch_size} batches.")
        return sentiments

    # ------------------------------------------------------------------
    # Stage 2 – Event Classification
    # ------------------------------------------------------------------
    def _run_event_classification(self, headlines):
        """Returns list of event-type strings."""
        print("\n===== Stage 2/3: Event Classification (BART-large-MNLI) =====")
        pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device,
            batch_size=4,          # smaller batches for the largest model
            framework="pt"
        )

        event_labels = []
        batch_size = 8             # outer loop batch
        errors = 0
        for i in tqdm(range(0, len(headlines), batch_size), desc="Events"):
            batch = headlines[i:i + batch_size]
            try:
                results = pipe(batch, candidate_labels=self.candidate_labels)
                # results is a list of dicts when >1 input, single dict when 1
                if isinstance(results, dict):
                    results = [results]
                top_labels = [r['labels'][0] for r in results]
                event_labels.extend(top_labels)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [WARN] Event batch {i//batch_size} error: {e}")
                event_labels.extend(["Stock Movement"] * len(batch))

        del pipe
        self._free_gpu()
        print(f"  Events done. {errors} batch errors out of "
              f"{(len(headlines) + batch_size - 1) // batch_size} batches.")
        return event_labels

    # ------------------------------------------------------------------
    # Stage 3 – MRC Entity Extraction
    # ------------------------------------------------------------------
    def _run_mrc(self, df):
        """Adds entity_a, entity_b, event_confidence columns in-place."""
        print("\n===== Stage 3/3: MRC Entity Extraction =====")
        pipe = pipeline(
            "question-answering",
            model=self.qa_model_name,
            device=self.device,
            framework="pt"
        )

        entity_a_list = []
        entity_b_list = []
        confidence_list = []
        errors = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="MRC"):
            headline = str(row['title'])
            evt = row['event_type']

            ea, eb, score = self._extract_entities(pipe, headline, evt)

            # Fallback: use the 'stock' column ticker
            if not ea and 'stock' in row.index:
                t = str(row['stock']).upper()
                if self.linker and self.linker.link_entity(t):
                    ea = t
                    if score == 0.0:
                        score = 0.5

            entity_a_list.append(ea)
            entity_b_list.append(eb)
            confidence_list.append(score)

        del pipe
        self._free_gpu()

        df['entity_a'] = entity_a_list
        df['entity_b'] = entity_b_list
        df['event_confidence'] = confidence_list
        print(f"  MRC done. {errors} errors.")
        return df

    def _extract_entities(self, pipe, headline, event_type):
        """Uses QA pipeline to extract entities based on event type."""
        if not self.linker:
            return None, None, 0.0

        queries = self.mrc_queries.get(event_type, [])
        if not queries:
            return None, None, 0.0

        found = {}
        total_score = 0

        for question, role in queries:
            try:
                result = pipe(question=question, context=headline)
                if result['score'] > 0.1:
                    ticker = self.linker.link_entity(result['answer'])
                    if ticker:
                        found[role] = ticker
                        total_score += result['score']
            except Exception:
                # Log but continue
                pass

        return found.get("entity_a"), found.get("entity_b"), total_score

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_news(self, force=False):
        """
        Run the full 3-stage pipeline.
        Set force=True to reprocess even if output file already exists.
        """
        # --- Check for existing output ---
        if not force and os.path.exists(self.output_path):
            print(f"Processed news found at {self.output_path}. Checking quality...")
            existing = pd.read_csv(self.output_path)
            if 'entity_a' in existing.columns:
                # Check if it is NOT all-fallback data
                n_unique_events = existing['event_type'].nunique()
                n_unique_sentiments = existing['sentiment_label'].nunique()
                if n_unique_events > 1 and n_unique_sentiments > 1:
                    print("  Existing file has diverse events & sentiments. Skipping re-processing.")
                    print(f"  Event distribution:\n{existing['event_type'].value_counts()}")
                    return existing
                else:
                    print(f"  WARNING: Existing file has only {n_unique_events} event type(s) "
                          f"and {n_unique_sentiments} sentiment label(s).")
                    print("  This looks like fallback data. Re-processing...")
            else:
                print("  Old format detected (missing entity_a). Re-processing...")

        # --- Load the filtered news ---
        print(f"\nLoading filtered news from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        if df.empty:
            print("  Empty DataFrame. Nothing to process.")
            return df

        print(f"  Loaded {len(df)} headlines.")
        headlines = df['title'].astype(str).tolist()

        # --- Stage 1: Sentiment ---
        sentiments = self._run_sentiment(headlines)
        df['sentiment_label'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        df['sentiment_scalar'] = (df['sentiment_label'].map(sentiment_map)
                                  * df['sentiment_score'])

        # --- Stage 2: Event Classification ---
        event_labels = self._run_event_classification(headlines)
        df['event_type'] = event_labels

        # --- Stage 3: MRC Entity Extraction ---
        df = self._run_mrc(df)

        # --- Save ---
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"\n[OK] Saved processed news to {self.output_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Sentiment distribution:\n{df['sentiment_label'].value_counts().to_string()}")
        print(f"  Event distribution:\n{df['event_type'].value_counts().to_string()}")
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process news with FinBERT and MRC")
    parser.add_argument("--portfolio", type=str, default="28stocks",
                        help="Portfolio name (e.g. 28stocks)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing even if output exists")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (slower but avoids GPU OOM)")

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_FILE  = os.path.join(BASE_DIR, "data", "processed", f"filtered_news_{args.portfolio}.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", f"news_with_events_{args.portfolio}.csv")
    TICKERS_FILE = os.path.join(BASE_DIR, "data", f"{args.portfolio}_tickers.txt")

    print(f"=== News Processing for {args.portfolio} ===")
    print(f"Input:   {INPUT_FILE}")
    print(f"Output:  {OUTPUT_FILE}")
    print(f"Tickers: {TICKERS_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"\n[ERROR] Input file not found: {INPUT_FILE}")
        print("Please run news_ingester.py first.")
        exit(1)

    if not os.path.exists(TICKERS_FILE):
        print(f"\n[ERROR] Tickers file not found: {TICKERS_FILE}")
        exit(1)

    device = -1 if args.cpu else None
    processor = NewsProcessor(INPUT_FILE, OUTPUT_FILE, TICKERS_FILE, device=device)
    processor.process_news(force=args.force)
