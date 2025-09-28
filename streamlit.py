import os
import sys
import json
import time
import threading
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Imports for PySpark components
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import VectorUDT
from newsapi import NewsApiClient

# =============================================================================
# --- Configuration & Environment Setup ---
# =============================================================================

OUTPUT_DIR = "data/input"
MODEL_PATH = "models/sentiment_model"
DASHBOARD_FILE = "data/output/latest_results.json"

QUERY = "technology OR business OR finance"
FETCH_INTERVAL_SECONDS = 60
API_KEY = os.getenv('NEWS_API_KEY')

# Set up PySpark environment variables
def setup_pyspark_env():
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    # FIX: Commented out PYTHONHASHSEED assignment to resolve deployment IndexErrors
    # os.environ['PYTHONHASHSEED'] = '0'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'
    os.environ['SPARK_SQL_EXECUTION_ARROW_PYSPARK_ENABLED'] = 'false'
    print(f"✅ PySpark environment configured for: {python_path}")
    
setup_pyspark_env()


# =============================================================================
# 1. NEWS PRODUCER (Adapted for Threading)
# =============================================================================

def fetch_and_write_news():
    """
    Fetches news using NewsAPI and writes each headline
    to a separate JSON file in the output directory. (Continuous loop)
    """
    if not API_KEY:
        print("Error: NEWS_API_KEY environment variable not set.")
        return
        
    print("Starting news producer with NewsAPI...")
    try:
        newsapi = NewsApiClient(api_key=API_KEY)
    except Exception:
        print("Error: Invalid API Key. Please check your API_KEY environment variable.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    while True:
        try:
            response = newsapi.get_everything(
                q=QUERY,
                language='en',
                sort_by='relevancy',
                page_size=20 
            )

            if response['status'] == 'ok':
                articles = response.get('articles', [])
                if articles:
                    for item in articles:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        file_path = os.path.join(OUTPUT_DIR, f"news_{timestamp}.json")
                        
                        news_data = {
                            "headline": item['title'],
                            "description": item.get('description', ''),
                            "source": item.get('source', {}).get('name', 'Unknown'),
                            "published_at": item.get('publishedAt', ''),
                            "url": item.get('url', ''),
                            "timestamp": datetime.now().isoformat()
                        }
                        with open(file_path, 'w') as f:
                            json.dump(news_data, f)
                else:
                    print("No news articles found in this fetch.")
            else:
                print(f"Error from NewsAPI: {response.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        time.sleep(FETCH_INTERVAL_SECONDS)


# =============================================================================
# 2. SENTIMENT CLASSIFICATION ML PIPELINE
# =============================================================================

class SentimentClassifier:
    def __init__(self):
        try:
            self.spark = SparkSession.builder \
                .appName("NewsSentimentClassification") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .config("spark.python.worker.reuse", "false")\
                .master("local") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("ERROR")
            print("✅ Spark session created successfully")
        except Exception as e:
            print(f"❌ Error creating Spark session: {e}")
            raise
        self.model = None
        self.pipeline = None

    def create_training_data(self):
        """Create synthetic training data for sentiment classification."""
        positive_headlines = [
            "Company reports record profits this quarter",
            "New breakthrough in renewable energy technology",
            "Stock market reaches new all-time high",
            "Unemployment rate drops to lowest level",
            "Innovation drives economic growth",
            "Successful product launch exceeds expectations",
            "Technology advances improve healthcare outcomes",
            "Strong earnings drive share price up",
            "Investment in green energy creates jobs",
            "Consumer confidence reaches decade high"
        ]
        negative_headlines = [
            "Major data breach affects millions of users",
            "Company announces massive layoffs",
            "Stock prices plummet amid recession fears",
            "Economic downturn impacts global markets",
            "Cybersecurity threat shuts down operations",
            "Inflation reaches concerning levels",
            "Supply chain disruptions cause delays",
            "Environmental disaster affects local business",
            "Trade war escalates between major economies",
            "Corporate scandal leads to investigations"
        ]

        training_data = []
        for headline in positive_headlines:
            training_data.append((headline, "positive", 1))
        
        for headline in negative_headlines:
            training_data.append((headline, "negative", 0))

        schema = StructType([
            StructField("headline", StringType(), True),
            StructField("sentiment_label", StringType(), True),
            StructField("label", IntegerType(), True)
        ])
        return self.spark.createDataFrame(training_data, schema)

    def build_pipeline(self):
        """Build ML pipeline for sentiment classification"""
        tokenizer = Tokenizer(inputCol="headline", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features",
                             vocabSize=1000, minDF=2.0)
        idf = IDF(inputCol="raw_features", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
        self.pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lr])
        return self.pipeline

    def train_model(self):
        """Train the sentiment classification model"""
        print("Creating training data...")
        training_df = self.create_training_data()
        print("Building ML pipeline...")
        pipeline = self.build_pipeline()
        print("Training model...")
        self.model = pipeline.fit(training_df)
        print("Model training completed!")
        return self.model

    def predict_sentiment(self, df):
        """Predict sentiment for given DataFrame"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        predictions = self.model.transform(df)

        # FIX: Use a UDF to properly extract probability values from Vector
        def extract_probability(probability_vector):
            """Extract probability values from ML Vector"""
            if probability_vector is not None:
                # Convert to dense vector and get values
                prob_array = probability_vector.toArray()
                return float(max(prob_array))
            return 0.0

        # Register the UDF
        extract_prob_udf = udf(extract_probability, DoubleType())

        # Add readable sentiment labels and confidence
        predictions = predictions.withColumn(
            "predicted_sentiment",
            when(col("prediction") == 1, "positive").otherwise("negative")
        ).withColumn(
            "confidence",
            extract_prob_udf(col("probability"))
        )
        
        return predictions


# =============================================================================
# 3. SPARK STRUCTURED STREAMING PROCESSOR
# =============================================================================

class NewsStreamProcessor:
    def __init__(self, classifier):
        self.spark = classifier.spark
        self.classifier = classifier
        self.output_dir = "data/output"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_streaming_news(self):
        """Process streaming news data and classify sentiment in real-time"""
        print("Starting streaming news processing...")

        schema = StructType([
            StructField("headline", StringType(), True),
            StructField("description", StringType(), True),
            StructField("source", StringType(), True),
            StructField("published_at", StringType(), True),
            StructField("url", StringType(), True),
            StructField("timestamp", StringType(), True)
        ])

        streaming_df = self.spark \
            .readStream \
            .option("multiline", "true") \
            .schema(schema) \
            .json(OUTPUT_DIR)

        streaming_df = streaming_df.withColumn("processing_time", current_timestamp())

        def process_batch(batch_df, batch_id):
            print(f"Processing batch {batch_id} with {batch_df.count()} records...")
            if batch_df.count() > 0:
                predictions = self.classifier.predict_sentiment(batch_df)
                
                results = predictions.select(
                    "headline", "source", "predicted_sentiment",
                    "confidence", "timestamp", "processing_time" 
                )
                
                # Write to consolidated file for dashboard
                dashboard_file = DASHBOARD_FILE
                results_pandas = results.toPandas()
                results_pandas.to_json(dashboard_file, orient='records', date_format='iso')
                
        query = streaming_df \
            .writeStream \
            .foreachBatch(process_batch) \
            .option("checkpointLocation", "data/checkpoint") \
            .start()

        return query


# =============================================================================
# 4. STREAMLIT DASHBOARD & RUNNER
# =============================================================================

class NewsDashboard:
    def __init__(self):
        self.results_file = DASHBOARD_FILE

    def load_results(self):
        """Load latest sentiment analysis results"""
        try:
            if os.path.exists(self.results_file):
                return pd.read_json(self.results_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return pd.DataFrame()

    def create_dashboard(self, df, stream_running):
        """Create Streamlit dashboard interface"""
        st.set_page_config(
            page_title="Real-Time News Sentiment Dashboard",
            page_icon="📰",
            layout="wide"
        )
        
        st.title("📰 Real-Time News Sentiment Analysis Dashboard")
        st.markdown("Live sentiment classification of news headlines using PySpark ML")

        if not stream_running:
            st.warning("System processes are inactive. Please start the real-time processes.")
        
        if df.empty:
            st.warning("No data available yet. Waiting for streaming results...") 
            return

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Headlines", len(df))
        with col2:
            positive_count = len(df[df['predicted_sentiment'] == 'positive'])
            st.metric("Positive", positive_count)
        with col3:
            negative_count = len(df[df['predicted_sentiment'] == 'negative'])
            st.metric("Negative", negative_count)
        with col4:
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = df['predicted_sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={'positive': 'green', 'negative': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if 'source' in df.columns:
                source_sentiment = df.groupby(['source', 'predicted_sentiment']).size().unstack(fill_value=0)
                fig_bar = px.bar(
                    source_sentiment.reset_index(),
                    x='source',
                    y=['positive', 'negative'],
                    title="Sentiment by News Source",
                    color_discrete_map={'positive': 'green', 'negative': 'red'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Recent headlines table
        st.subheader("Recent Headlines")
        display_df = df[['headline', 'source', 'predicted_sentiment', 'confidence']].copy()
        display_df = display_df.sort_values('confidence', ascending=False).head(10)
        
        def style_sentiment(val):
            color = 'green' if val == 'positive' else 'red'
            return f'color: {color}; font-weight: bold'

        styled_df = display_df.style.applymap(
            style_sentiment, subset=['predicted_sentiment']
        ).format({'confidence': '{:.3f}'})
        
        st.dataframe(styled_df, use_container_width=True)
        st.info("💡 This dashboard updates in real-time. Click refresh to see latest results.")


def initialize_system():
    """Initializes environment, classifier, and trains/loads the ML model."""
    
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if 'classifier' not in st.session_state:
        try:
            classifier = SentimentClassifier()
            st.session_state['classifier'] = classifier
            
            if os.path.exists(MODEL_PATH):
                classifier.model = PipelineModel.load(MODEL_PATH)
                st.success("Loaded pre-trained ML model.")
            else:
                st.info("No model found. Starting training process...")
                classifier.train_model()
                classifier.model.write().overwrite().save(MODEL_PATH)
                st.success(f"Trained and saved new ML model to {MODEL_PATH}.")
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            st.session_state['init_failed'] = True

def start_background_processes():
    """Starts News Producer via Threading and initiates PySpark Streaming."""
    
    if st.session_state.get('streaming_running', False):
        st.info("Background processes are already running.")
        return

    if 'classifier' not in st.session_state:
        st.error("Cannot start streamer: Model Classifier has not been initialized. Please click 'Initialize/Train System'.")
        return

    st.info("Attempting to start continuous processes...")
    
    # 1. Start News Producer in a separate thread (Threading required for non-blocking execution)
    producer_thread = threading.Thread(target=fetch_and_write_news, daemon=True)
    producer_thread.start()
    st.session_state['producer_thread'] = producer_thread
    
    # 2. Start Streaming Processor
    processor = NewsStreamProcessor(st.session_state['classifier'])
    
    try:
        query = processor.process_streaming_news()
        
        st.session_state['spark_query'] = query
        st.session_state['streaming_running'] = True
        
        st.success("✅ PySpark Streaming Processor and News Producer started in background.")
        st.warning(f"Data may take up to {FETCH_INTERVAL_SECONDS} seconds to appear on the dashboard.")
        
    except Exception as e:
         st.error(f"❌ Failed to start Streaming Processor: {e}")
         st.session_state['streaming_running'] = False


def run_unified_app():
    """The main Streamlit interface function."""
    
    dashboard = NewsDashboard()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("1. Initialize/Train System"):
            initialize_system()
        
        if 'classifier' in st.session_state and not st.session_state.get('init_failed', False):
            if st.button("2. Start Real-Time Processes"):
                start_background_processes()
                
        # Status indicators
        st.markdown("---")
        st.subheader("Status")
        if 'classifier' in st.session_state:
            st.success("Model Initialized")
        else:
            st.error("Model Not Initialized")
            
        if st.session_state.get('streaming_running', False):
            st.success("Streaming Active")
        else:
            st.error("Streaming Inactive")

    # --- Dashboard Display ---
    
    df = dashboard.load_results()
    dashboard.create_dashboard(df, st.session_state.get('streaming_running', False))


if __name__ == "__main__":
    if not API_KEY:
        st.error("Please set the NEWS_API_KEY environment variable (see SETUP INSTRUCTIONS).")
    else:
        run_unified_app()