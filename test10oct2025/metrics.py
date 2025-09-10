from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_group"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"]
)

RATE_LIMIT_CHECKED = Counter(
    "rate_limiter_checked_total",
    "Total requests checked by the rate limiter"
)
RATE_LIMIT_ALLOWED = Counter(
    "rate_limiter_allowed_total",
    "Total requests allowed by the rate limiter"
)
RATE_LIMIT_REJECTED = Counter(
    "rate_limiter_rejected_total",
    "Total requests rejected by the rate limiter"
)

REDIS_CONNECTION_ERRORS = Counter(
    "redis_connection_errors_total",
    "Total Redis connection errors"
)
REDIS_CONNECTION_RETRIES = Counter(
    "redis_connection_retries_total",
    "Total Redis connection retries"
)
REDIS_CONNECTION_TIMEOUTS = Counter(
    "redis_connection_timeouts_total",
    "Total Redis connection timeouts"
)

faithfulness_gauge = Gauge("ragas_faithfulness", "Faithfulness score of RAGAS evaluation")
answer_relevance_gauge = Gauge("ragas_answer_relevance", "Answer relevance score of RAGAS evaluation")
context_precision_gauge = Gauge("ragas_context_precision", "Context precision score of RAGAS evaluation")
similarity_gauge = Gauge("ragas_similarity", "Similarity score of RAGAS evaluation (if reference exists)")