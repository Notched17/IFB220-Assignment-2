import json
from unittest.mock import MagicMock

from src.api_client import EmbeddingResult
from src.guardrails.topic_relevance import TopicRelevanceChecker
from src.topic import Topic

CLIMBING_TOPIC = Topic(
    topic_id="test_topic",
    display_name="Test",
    persona_description="a test persona",
    in_scope_summary="testing things",
    out_of_scope_note="nothing else",
    refusal_message="refused",
    injection_refusal_message="injection refused",
    similarity_threshold=0.5,
    anchor_phrases=("anchor one", "anchor two"),
)


def _fake_embed_factory(vector_for):
    def _embed(text):
        return EmbeddingResult(vector=vector_for(text), total_tokens=len(text.split()))

    return _embed


def test_centroid_is_mean_of_anchor_embeddings_and_scores_similar_text_highly(tmp_path):
    # Anchors both point in the same direction -> centroid == that direction.
    vectors = {"anchor one": [1.0, 0.0], "anchor two": [1.0, 0.0], "close text": [0.9, 0.1]}
    fake_api = MagicMock()
    fake_api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])

    checker = TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.9)
    result = checker.score("close text")

    assert result.score > 0.9
    assert result.passed is True


def test_orthogonal_text_scores_low_and_fails(tmp_path):
    vectors = {"anchor one": [1.0, 0.0], "anchor two": [1.0, 0.0], "unrelated": [0.0, 1.0]}
    fake_api = MagicMock()
    fake_api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])

    checker = TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5)
    result = checker.score("unrelated")

    assert result.score < 0.5
    assert result.passed is False


def test_centroid_is_cached_to_disk_and_not_recomputed(tmp_path):
    vectors = {"anchor one": [1.0, 0.0], "anchor two": [1.0, 0.0], "query": [1.0, 0.0]}
    fake_api = MagicMock()
    fake_api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])

    checker1 = TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5,
                                     embedding_model="model-a")
    checker1.score("query")
    calls_after_first = fake_api.embed.call_count  # 2 anchors + 1 query = 3

    cache_file = tmp_path / "cache" / f"{CLIMBING_TOPIC.topic_id}__model-a_centroid.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())
    assert cached["centroid"] == [1.0, 0.0]

    # A fresh checker instance should reuse the cached centroid instead of
    # re-embedding the anchor phrases.
    checker2 = TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5,
                                     embedding_model="model-a")
    checker2.score("query")
    calls_after_second = fake_api.embed.call_count

    assert calls_after_second - calls_after_first == 1  # only the new query, no re-embedding anchors


def test_on_embedding_call_hook_receives_token_counts(tmp_path):
    vectors = {"anchor one": [1.0, 0.0], "anchor two": [1.0, 0.0], "query text": [1.0, 0.0]}
    fake_api = MagicMock()
    fake_api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])

    recorded = []
    checker = TopicRelevanceChecker(
        fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5, on_embedding_call=recorded.append
    )
    checker.score("query text")

    # 2 anchor embeds (centroid build) + 1 query embed = 3 recorded calls.
    assert len(recorded) == 3


def test_changing_embedding_model_does_not_reuse_stale_centroid(tmp_path):
    vectors = {"anchor one": [1.0, 0.0], "anchor two": [1.0, 0.0], "query": [1.0, 0.0]}
    fake_api = MagicMock()
    fake_api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])

    TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5,
                          embedding_model="model-a").score("query")
    assert fake_api.embed.call_count == 3  # 2 anchors + query

    TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5,
                          embedding_model="model-b").score("query")
    # A different model must re-embed both anchors, not reuse model-a's centroid.
    assert fake_api.embed.call_count == 3 + 3
    assert (tmp_path / "cache" / "test_topic__model-a_centroid.json").exists()
    assert (tmp_path / "cache" / "test_topic__model-b_centroid.json").exists()


def test_model_name_is_part_of_the_anchor_hash():
    assert CLIMBING_TOPIC.anchor_phrases_hash("model-a") != CLIMBING_TOPIC.anchor_phrases_hash("model-b")


def test_corrupt_cache_file_is_rebuilt_not_fatal(tmp_path):
    vectors = {"anchor one": [1.0, 0.0], "anchor two": [1.0, 0.0], "query": [1.0, 0.0]}
    fake_api = MagicMock()
    fake_api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "test_topic__m_centroid.json").write_text("{not json")

    result = TopicRelevanceChecker(fake_api, CLIMBING_TOPIC, tmp_path, threshold=0.5,
                                   embedding_model="m").score("query")
    assert result.passed is True


# --- Red-team round: closest-anchor score and batch scoring --------------------
from dataclasses import replace as _replace  # noqa: E402

from src.api_client import EmbeddingBatch  # noqa: E402


def _checker(tmp_path, vectors, offset):
    topic = _replace(CLIMBING_TOPIC, anchor_phrases=("a1", "a2"), anchor_score_offset=offset)
    api = MagicMock()
    api.embed.side_effect = _fake_embed_factory(lambda t: vectors[t])
    api.embed_many.side_effect = lambda texts: EmbeddingBatch([vectors[t] for t in texts], 7)
    return TopicRelevanceChecker(api, topic, tmp_path, threshold=0.5), api


def test_closest_anchor_score_lifts_text_that_matches_one_anchor(tmp_path):
    # a1 and a2 point in different directions; "slang" matches a1 exactly but
    # sits 45 degrees from the centroid.
    vectors = {"a1": [1.0, 0.0], "a2": [0.0, 1.0], "slang": [1.0, 0.0]}
    without, _ = _checker(tmp_path / "x", vectors, offset=None)
    with_anchor, _ = _checker(tmp_path / "y", vectors, offset=0.15)
    r0, r1 = without.score("slang"), with_anchor.score("slang")
    assert abs(r0.score - 0.7071) < 1e-3                        # centroid only
    assert r1.anchor_score is not None and abs(r1.anchor_score - 1.0) < 1e-9
    assert abs(r1.score - 0.85) < 1e-9                          # max(0.707, 1.0 - 0.15)


def test_offset_none_keeps_the_original_centroid_behaviour(tmp_path):
    vectors = {"a1": [1.0, 0.0], "a2": [0.0, 1.0], "q": [0.0, 1.0]}
    checker, _ = _checker(tmp_path, vectors, offset=None)
    result = checker.score("q")
    assert result.score == result.centroid_score


def test_score_many_uses_one_embedding_call(tmp_path):
    vectors = {"a1": [1.0, 0.0], "a2": [1.0, 0.0], "s1": [1.0, 0.0], "s2": [0.0, 1.0]}
    calls = []
    checker, api = _checker(tmp_path, vectors, offset=None)
    checker._on_embedding_call = calls.append
    results = checker.score_many(["s1", "s2"], threshold=0.5)
    assert [r.passed for r in results] == [True, False]
    assert api.embed_many.call_count == 1 and calls[0] == 7   # then the 2 one-off anchor builds
    assert checker.score_many([]) == []


def test_anchor_vectors_are_cached_with_the_centroid(tmp_path):
    vectors = {"a1": [1.0, 0.0], "a2": [0.0, 1.0], "q": [1.0, 0.0]}
    checker, api = _checker(tmp_path, vectors, offset=0.1)
    checker.score("q")
    cache = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert cache["anchor_vectors"] == [[1.0, 0.0], [0.0, 1.0]]
    fresh, api2 = _checker(tmp_path, vectors, offset=0.1)
    fresh.score("q")
    assert [c.args[0] for c in api2.embed.call_args_list] == ["q"]   # anchors came from the cache
