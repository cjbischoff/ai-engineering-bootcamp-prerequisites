"""
LangSmith feedback processor (Week 4 / Human Feedback).

Sends thumbs (score) and optional comment feedback from the UI to LangSmith so runs
can be scored and reviewed in the LangSmith/LangGraph UI. trace_id is the run_id
from the RAG/LangGraph invocation; without it we cannot attach feedback to a run.
"""
import logging

from langsmith import Client

logger = logging.getLogger(__name__)
client = Client()


def submit_feedback(
    trace_id: str | None,
    feedback_score: int | None = None,
    feedback_text: str = "",
    feedback_source_type: str = "api",
):
    """Send feedback to LangSmith. No-ops if trace_id is missing (e.g. before first RAG response)."""
    # Optional trace_id: UI may submit feedback before any RAG response (e.g. page load).
    # Skipping here avoids 422 from LangSmith and lets the API still return 200.
    if not trace_id:
        logger.warning(
            "submit_feedback skipped: trace_id is missing (feedback_score=%s, has_text=%s)",
            feedback_score,
            bool(feedback_text and feedback_text.strip()),
        )
        return

    logger.info(
        "Submitting feedback to LangSmith: run_id=%s, score=%s, has_comment=%s",
        trace_id,
        feedback_score,
        bool(feedback_text and feedback_text.strip()),
    )
    # Thumbs: 1 = positive, 0 = negative. Stored under key "thumbs" in LangSmith run feedback.
    if feedback_score is not None:
        client.create_feedback(
            run_id=trace_id,
            key="thumbs",
            score=feedback_score,
            feedback_source_type=feedback_source_type,
        )
        logger.info("LangSmith create_feedback thumbs: run_id=%s, score=%s", trace_id, feedback_score)

    # Comment: optional free-text from "Send Additional Details" (e.g. after thumbs down).
    if feedback_text and len(feedback_text.strip()) > 0:
        client.create_feedback(
            run_id=trace_id,
            key="comment",
            value=feedback_text,
            feedback_source_type=feedback_source_type,
        )
        logger.info("LangSmith create_feedback comment: run_id=%s", trace_id)
