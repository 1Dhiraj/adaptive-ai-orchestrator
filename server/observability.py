"""Prometheus metrics and optional OpenTelemetry tracing."""
from __future__ import annotations

import os
import time
from typing import Any, Callable

try:
    from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest
    HTTP_REQUESTS = Counter("orchestrator_http_requests_total", "HTTP requests",
                            ["method", "route", "status"])
    HTTP_DURATION = Histogram("orchestrator_http_request_seconds", "HTTP request latency",
                              ["method", "route"])
    QUEUED_JOBS = Gauge("orchestrator_jobs_queued", "Durable jobs waiting for a worker")
    FAILED_JOBS = Gauge("orchestrator_jobs_failed", "Failed durable jobs")
except ImportError:  # local minimal install
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    HTTP_REQUESTS = HTTP_DURATION = QUEUED_JOBS = FAILED_JOBS = None


def metrics_payload(state: Any) -> bytes:
    if QUEUED_JOBS is None:
        return b"# prometheus-client is not installed\n"
    jobs = state.list_jobs(limit=1000)
    QUEUED_JOBS.set(sum(j["status"] == "queued" for j in jobs))
    FAILED_JOBS.set(sum(j["status"] == "failed" for j in jobs))
    return generate_latest()


def configure_tracing(app: Any) -> bool:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "adaptive-orchestrator")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        return True
    except ImportError:
        return False

