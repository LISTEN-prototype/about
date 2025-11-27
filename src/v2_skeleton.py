import ollama

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, TimestampType
from pyspark.sql.functions import col

def parse_outputs( content : str ) -> Iterable[Dict[str, Any]]:   
  if isinstance(gen, dict):
      # attempt several common locations
      if 'choices' in gen and len(gen['choices'])>0:
          content = gen['choices'][0].get('message', {}).get('content') or gen['choices'][0].get('text')
      elif 'text' in gen:
          content = gen.get('text')
      else:
          # fallback: use whole response as string
          content = json.dumps(gen)
  else:
      content = str(gen)
  parsed = json.loads(content)



# ---------- Partition-scoped helpers used inside mapPartitions ----------
def partition_ollama_pipeline(rows: Iterable[Row]) -> Iterable[Dict[str, Any]]:
    """
    Process a partition of rows:
     - preprocess text
     - batch embeddings via Ollama
     - batch prompt-classification via Ollama
    Returns dicts with fields: id, text, embed, labels, scores, rationale_json
    """

    prompt = (
              "You are a public-health misinformation classifier. "
              "Return valid JSON with keys: misinfo_probability (0-1), narrative_type (one of ['transmission','vaccine','stigma','conspiracy','other']), "
              "rationale (short human-readable text). "
              "Respond ONLY with JSON.\n\n"
              f"Post: '''{txt}'''"
            )
                
    rows = list(rows)
    
    if not rows:
        return []

    # prepare data lists
    ids = [str(r['id']) for r in rows]
    texts = [r['text'] if r['text'] is not None else "" for r in rows]

    # simple preprocessing (lowercase, trim)
    prepped = [t.strip() for t in texts]

    results = []
  
    # embeddings in batches
    for i in range(0, len(prepped), BATCH_SIZE):
        batch_texts = prepped[i:i+BATCH_SIZE]   
        try:
            embeddings = call_ollama_embed(batch_texts)
        except Exception as e:
            # fallback to empty embeddings
            embeddings = [[0.0]] * len(batch_texts)
          
        try:
          out = ollama.generate( prompt, max_tokens = 256, temperature = 0.0 )
          parsed = parse_outputs( out ) 
          rec = {
                  "id": doc_id,
                  "text": txt,
                  "embedding": emb,
                  "misinfo_probability": parsed.get("misinfo_probability"),
                  "narrative_type": parsed.get("narrative_type"),
                  "rationale": parsed.get("rationale"),
                  "raw_json": parsed,
                  "processed_at": int(time.time()) }
        except Exception as e:
          rec = {
              "id": doc_id,
              "text": txt,
              "embedding": None,
              "misinfo_probability": None,
              "narrative_type": "error",
              "rationale": str(e)[:200],
              "raw_json": {},
              "processed_at": int(time.time())
          }
        results.append(rec)
return iter(results)
  


def partition_metadata_refresh(tweet_id_rows: Iterable[Row], twitter_bearer_token: str) -> Iterable[Dict[str, Any]]:
    """
    Per-partition metadata refresh that respects rate límites by sleeping between requests.
    This uses a partition-local client to avoid creating a session for every row.
    """
    client = SimpleTwitterClient(twitter_bearer_token)
    results = []
    # very simple throttling: allow N calls per minute per partition; adjust as needed
    calls_per_min = int(TWITTER_RATE_LIMIT_PER_MIN / 10) or 1  # naive split if many partitions; better: central coordinator (Redis) in prod
    interval = 60.0 / max(calls_per_min, 1)
    for r in tweet_id_rows:
        tweet_id = str(r['id'])
        try:
            data = client.get_tweet_by_id(tweet_id)
            pm = data.get("public_metrics", {})
            rec = {
                "id": tweet_id,
                "like_count": pm.get("like_count"),
                "retweet_count": pm.get("retweet_count"),
                "reply_count": pm.get("reply_count"),
                "quote_count": pm.get("quote_count"),
                "fetched_at": data.get("fetched_at", int(time.time()))
            }
        except Exception as e:
            rec = {
                "id": tweet_id,
                "like_count": None,
                "retweet_count": None,
                "reply_count": None,
                "quote_count": None,
                "fetched_at": int(time.time()),
                "error": str(e)[:200]
            }
        results.append(rec)
        # throttle
        time.sleep(interval)
    return iter(results)


# ---------- Main Spark job ----------

def main():
    args = parse_args()

    spark = SparkSession.builder.appName("LISTEN_v2_Ollama_Spark").getOrCreate()
    sc = spark.sparkContext

    # Read raw posts: expected schema has at least 'id' and 'text'
    df_raw = spark.read.parquet(args.input).select(col("id").cast(StringType()).alias("id"), col("text").alias("text"))
    
    # repartition to a reasonable number based on cluster size and API throughput (e.g., 200 partitions)
    df_raw = df_raw.repartition(200)

    # MapPartitions: run ollama pipeline per partition for efficiency (keep single HTTP session per partition when possible)
    rdd_rows = df_raw.rdd.mapPartitions(partition_ollama_pipeline)

    # Convert to DataFrame and persist results
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("embedding", StringType(), True),  # storing embedding as JSON string for simplicity; convert if needed
        StructField("misinfo_probability", DoubleType(), True),
        StructField("narrative_type", StringType(), True),
        StructField("rationale", StringType(), True),
        StructField("raw_json", StringType(), True),
        StructField("processed_at", LongType(), True)
    ])

    # Convert RDD of dict -> Rows
    def dict_to_row(d):
        # convert embedding list to JSON string (or binary) for storage
        emb = d.get("embedding")
        if emb is not None and not isinstance(emb, str):
            emb = json.dumps(emb)
        rawj = d.get("raw_json")
        if rawj is not None and not isinstance(rawj, str):
            rawj = json.dumps(rawj)
        return Row(
            id=str(d.get("id")),
            text=d.get("text"),
            embedding=emb,
            misinfo_probability=None if d.get("misinfo_probability") is None else float(d.get("misinfo_probability")),
            narrative_type=d.get("narrative_type"),
            rationale=d.get("rationale"),
            raw_json=rawj,
            processed_at=int(d.get("processed_at") or time.time())
        )

    rows_rdd = rdd_rows.map(dict_to_row)
    df_processed = spark.createDataFrame(rows_rdd, schema=schema)

    # Write results as partitioned Parquet for historical analysis
    output_path = args.output.rstrip("/")
    df_processed.write.mode("overwrite").partitionBy("processed_at").parquet(f"{output_path}/classifications/")

    # Optional: register to Hive or update a table (uncomment if metastore available)
    # df_processed.write.mode("append").saveAsTable("listen_v2.classifications")

    # METADATA REFRESH STEP (optional)
    if args.refresh_metadata and args.tweet_ids:
        # Read tweet id list
        df_ids = spark.read.parquet(args.tweet_ids).select(col("id").cast(StringType()).alias("id"))
        df_ids = df_ids.repartition(200)  # tune partition count

        # Broadcast token or put into cluster-secure store
        twitter_token = "REPLACE_WITH_REAL_TOKEN"  # in production read from Secrets Manager, not hard-coded

        # mapPartitions with Twikit-like client
        rdd_meta = df_ids.rdd.mapPartitions(lambda rows: partition_metadata_refresh(rows, twitter_token))
        # convert to dataframe
        meta_schema = StructType([
            StructField("id", StringType(), True),
            StructField("like_count", LongType(), True),
            StructField("retweet_count", LongType(), True),
            StructField("reply_count", LongType(), True),
            StructField("quote_count", LongType(), True),
            StructField("fetched_at", LongType(), True),
            StructField("error", StringType(), True)
        ])
        df_meta = spark.createDataFrame(rdd_meta.map(lambda d: Row(**d)), schema=meta_schema)
        df_meta.write.mode("overwrite").partitionBy("fetched_at").parquet(f"{output_path}/metadata_refresh/")

    spark.stop()

if __name__ == "__main__":
    main()
