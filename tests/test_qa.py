import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from assistant.agents.qa import _Claim, _parse, _render_claims, _verified_evidence, qa_node


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"id": "first", "paper_id": "paper-a", "text": "The method uses attention. It scored 80% on dataset A."},
            {"id": "second", "paper_id": "paper-b", "text": "Hybrid retrieval combines sparse and dense retrieval."},
        ]

    def claim(self, text="The method uses attention.", source=1, quote="The method uses attention."):
        return _Claim.model_validate({"text": text, "evidence": [{"source": source, "quote": quote}]})

    def verify(self, claims, verdicts):
        verifier = Mock()
        verifier.invoke.return_value = SimpleNamespace(content=json.dumps({"verdicts": verdicts}))
        router = Mock()
        router.chat.return_value = verifier
        config = SimpleNamespace(llm=SimpleNamespace(roles={"citation_verifier": object()}))
        with patch("assistant.agents.qa.get_router", return_value=router), patch("assistant.agents.qa.get_config", return_value=config):
            accepted = _verified_evidence(claims, self.chunks)
        return accepted, verifier, router

    def test_verified_claim_and_knowledge_have_distinct_attribution(self):
        claims = [self.claim(), _Claim(text="Attention can model dependencies.")]
        accepted, verifier, router = self.verify(claims, [{"candidate": 1, "supported": True}])
        answer, citations = _render_claims(claims, self.chunks, accepted)
        self.assertEqual(citations, ["first"])
        self.assertEqual(answer, "The method uses attention. [1]\n\nAttention can model dependencies. *(LLM knowledge)*")
        router.chat.assert_called_once_with("citation_verifier")
        payload = json.loads(verifier.invoke.call_args.args[0][1].content)
        self.assertEqual(payload["candidates"][0]["passage"], self.chunks[0]["text"])

    def test_quote_must_occur_in_the_cited_chunk(self):
        claims = [self.claim(quote="Invented evidence"), self.claim(source=2), self.claim(source=99)]
        with patch("assistant.agents.qa.get_router") as router:
            self.assertEqual(_verified_evidence(claims, self.chunks), {})
            router.assert_not_called()

    def test_related_quote_does_not_override_rejected_entailment(self):
        claims = [self.claim(text="The method scored 99% on dataset B.")]
        accepted, _, _ = self.verify(claims, [{"candidate": 1, "supported": False}])
        answer, citations = _render_claims(claims, self.chunks, accepted)
        self.assertEqual(citations, [])
        self.assertIn("LLM knowledge", answer)

    def test_malformed_incomplete_or_duplicate_verdicts_fail_closed(self):
        for verdicts in ([], [{"candidate": 2, "supported": True}], [{"candidate": 1, "supported": "true"}], [{"candidate": 1, "supported": True}] * 2):
            with self.subTest(verdicts=verdicts):
                accepted, _, _ = self.verify([self.claim()], verdicts)
                self.assertEqual(accepted, {})

    def test_source_order_and_injected_markers(self):
        claims = [_Claim(text="Hybrid search [99]."), _Claim(text="Attention."), _Claim(text="Hybrid search again.")]
        answer, citations = _render_claims(claims, self.chunks, {0: [2], 1: [1], 2: [2]})
        self.assertEqual(citations, ["second", "first"])
        self.assertNotIn("[99]", answer)
        self.assertEqual(answer.count("[1]"), 2)
        self.assertEqual(answer.count("[2]"), 1)

    def test_verifier_failure_withholds_citations(self):
        with self.assertLogs("assistant.agents.qa", level="WARNING"), patch("assistant.agents.qa.get_config", side_effect=RuntimeError("Unavailable")):
            self.assertEqual(_verified_evidence([self.claim()], self.chunks), {})

    def test_claim_blocks_cannot_embed_unverified_citation_links(self):
        for text in ("  ", "First claim\nSecond claim", "[source](citation:1)", "[source](citati&#111;n:1)", "[source](citation\\:1)"):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                _Claim(text=text)

    def test_whitespace_in_pdf_quote_is_normalized(self):
        claim = self.claim(quote="The method\nuses attention.")
        accepted, _, _ = self.verify([claim], [{"candidate": 1, "supported": True}])
        self.assertEqual(accepted, {0: [1]})

    def test_verifier_role_falls_back_for_existing_configs(self):
        config = SimpleNamespace(llm=SimpleNamespace(roles={"qa": object()}))
        router = Mock()
        router.chat.return_value.invoke.return_value = SimpleNamespace(content='{"verdicts": [{"candidate": 1, "supported": true}]}')
        with patch("assistant.agents.qa.get_config", return_value=config), patch("assistant.agents.qa.get_router", return_value=router):
            self.assertEqual(_verified_evidence([self.claim()], self.chunks), {0: [1]})
        router.chat.assert_called_once_with("qa")

    def run_qa(self, draft, chunks=None, verdicts=None):
        generator = Mock()
        generator.invoke.return_value = SimpleNamespace(content=json.dumps(draft))
        verifier = Mock()
        verifier.invoke.return_value = SimpleNamespace(content=json.dumps({"verdicts": verdicts or []}))
        router = Mock()
        router.chat.side_effect = lambda role: verifier if role == "citation_verifier" else generator
        config = SimpleNamespace(
            llm=SimpleNamespace(roles={"citation_verifier": object()}),
            rag=SimpleNamespace(history_turns=4, low_confidence_threshold=0.5),
        )
        with patch("assistant.agents.qa.get_router", return_value=router), patch("assistant.agents.qa.get_config", return_value=config), patch("assistant.agents.qa.retrieval_node", return_value={"retrieved_chunks": []}):
            result = qa_node({"question": "Explain the method", "retrieved_chunks": self.chunks if chunks is None else chunks})
        return result, generator, verifier

    def test_qa_wires_verification_and_preserves_response_contract(self):
        draft = {"claims": [self.claim().model_dump(), {"text": "General explanation."}], "confidence": 0.9}
        result, _, verifier = self.run_qa(draft, verdicts=[{"candidate": 1, "supported": True}])
        self.assertEqual(result["citations"], ["first"])
        self.assertIn("The method uses attention. [1]", result["answer"])
        self.assertIn("General explanation. *(LLM knowledge)*", result["answer"])
        self.assertEqual(result["confidence"], 0.9)
        verifier.invoke.assert_called_once()

    def test_no_chunks_can_answer_as_llm_knowledge(self):
        result, generator, verifier = self.run_qa({"claims": [{"text": "General explanation."}], "confidence": 0.9}, chunks=[])
        self.assertEqual(result["answer"], "General explanation. *(LLM knowledge)*")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["confidence"], 0.9)
        generator.invoke.assert_called_once()
        verifier.invoke.assert_not_called()

    def test_rejected_evidence_never_surfaces_a_citation(self):
        result, _, _ = self.run_qa({"claims": [self.claim().model_dump()], "confidence": 0.9}, verdicts=[{"candidate": 1, "supported": False}])
        self.assertEqual(result["citations"], [])
        self.assertIn("LLM knowledge", result["answer"])
        self.assertLess(result["confidence"], 0.5)

    def test_invalid_draft_never_uses_legacy_unverified_citations(self):
        for draft in ({"answer": "Unsupported [1]", "citations": [1], "confidence": 1}, {"claims": [{"text": "Invalid", "evidence": [{"source": True, "quote": "The method uses attention."}]}], "confidence": 1}, {"claims": [], "confidence": 1}):
            with self.subTest(draft=draft):
                result, _, verifier = self.run_qa(draft)
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["confidence"], 0.0)
                verifier.invoke.assert_not_called()

    def test_parse_accepts_fenced_json_but_not_non_object_json(self):
        self.assertEqual(_parse('```json\n{"claims": []}\n```'), {"claims": []})
        for raw in ("[]", "null", "plain prose"):
            self.assertEqual(_parse(raw), {})


if __name__ == "__main__":
    unittest.main()