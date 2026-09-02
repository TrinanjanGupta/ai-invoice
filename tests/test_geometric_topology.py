"""
tests/test_geometric_topology.py

Unit tests for geometric region topology similarity and candidate retrieval fallback behavior.
"""

import pytest
from understanding.template_retriever import TemplateRetriever, CachedTemplateVersion
from preprocessing.document_profile import DocumentProfile


def test_geometric_topology_proximity_and_iou():
    # Matching labels in identical normalized positions -> high score (1.0)
    regions_a = [
        {"label": "header", "bbox_norm": [50, 50, 950, 200], "page": 1},
        {"label": "table", "bbox_norm": [50, 250, 950, 700], "page": 1},
        {"label": "totals", "bbox_norm": [600, 750, 950, 900], "page": 1},
    ]
    regions_b = [
        {"label": "header", "bbox_norm": [50, 50, 950, 200], "page": 1},
        {"label": "table", "bbox_norm": [50, 250, 950, 700], "page": 1},
        {"label": "totals", "bbox_norm": [600, 750, 950, 900], "page": 1},
    ]

    score_identical = TemplateRetriever.compute_region_topology_similarity(regions_a, regions_b)
    assert score_identical >= 0.95

    # Matching labels in distant/different positions -> lower score
    regions_c = [
        {"label": "header", "bbox_norm": [50, 750, 950, 900], "page": 1}, # header at the bottom
        {"label": "table", "bbox_norm": [50, 50, 950, 400], "page": 1},
        {"label": "totals", "bbox_norm": [600, 50, 950, 200], "page": 1},
    ]

    score_shifted = TemplateRetriever.compute_region_topology_similarity(regions_a, regions_c)
    assert score_shifted < score_identical
    assert score_shifted < 0.70


def test_no_arbitrary_candidate_fallback_returns_novel():
    retriever = TemplateRetriever()
    profile = DocumentProfile(
        words=[],
        regions=[],
        page_count=1,
        width=1000,
        height=1400,
        aspect_ratio=1.4,
    )
    # When index has no matching candidates for an empty/novel profile
    res = retriever.retrieve(profile)
    assert res.match_type in ("none", "novel")
    assert res.match_confidence == 0.0
    assert res.matched_version_id == ""
