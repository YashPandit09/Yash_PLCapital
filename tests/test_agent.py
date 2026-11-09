"""
test_agent.py – Unit tests for the PortfolioAgent class.
"""

import pytest
from src.agent import AgentResult, PortfolioAgent


class TestPortfolioAgent:
    """Tests for agent query routing, caching, and execution."""

    @pytest.fixture(autouse=True)
    def patch_offline(self, monkeypatch):
        """Run agent unit tests deterministically using offline fallback."""
        monkeypatch.setattr("src.agent.is_available", lambda: False)

    def setup_method(self):
        self.agent = PortfolioAgent()

    def test_empty_question(self):
        res: AgentResult = self.agent.answer_question("")
        assert res.error != ""
        assert "non-empty" in res.error.lower()

    def test_whitespace_question(self):
        res: AgentResult = self.agent.answer_question("   ")
        assert res.error != ""

    def test_exposure_routing_tech(self):
        q = "What are the sector exposures for the Tech Innovation Fund?"
        res: AgentResult = self.agent.answer_question(q)
        assert res.tool_used == "exposure_calculator"
        assert not res.error
        assert "Technology" in res.answer
        assert res.raw_result.portfolio_name == "Tech Innovation Fund"

    def test_exposure_routing_fuzzy(self):
        q = "Calculate the sector exposure breakdown for international equity"
        res: AgentResult = self.agent.answer_question(q)
        assert res.tool_used == "exposure_calculator"
        assert not res.error
        assert res.raw_result.portfolio_name == "International Equity Fund"

    def test_query_caching(self):
        q = "What are the sector exposures for the Tech Innovation Fund?"
        # First call populates cache
        res1 = self.agent.answer_question(q)
        # Second call should be a cache hit
        res2 = self.agent.answer_question(q)
        assert res1.tool_used == res2.tool_used
        assert res1.answer == res2.answer

    def test_sql_routing_signal(self):
        tool, _ = self.agent._route("How many portfolios do we have in total?")
        assert tool == "sql_query"

    def test_sql_routing_active(self):
        tool, _ = self.agent._route("What are the names of all active portfolios?")
        assert tool == "sql_query"

    def test_out_of_domain_routing(self):
        res = self.agent.answer_question("What's the weather today?")
        assert res.tool_used == "none"
        assert "portfolio data" in res.answer.lower()

    def test_destructive_routing_drop(self):
        res = self.agent.answer_question("DROP TABLE holdings; --")
        assert res.tool_used == "none"
        assert "destructive" in res.answer.lower()

    def test_destructive_routing_delete(self):
        res = self.agent.answer_question("DELETE FROM portfolios WHERE 1=1")
        assert res.tool_used == "none"
        assert "destructive" in res.answer.lower()

    def test_llm_router_out_of_domain(self, monkeypatch):
        monkeypatch.setattr("src.agent.is_available", lambda: True)
        monkeypatch.setattr(
            "src.agent.generate",
            lambda q, system_instruction="": '{"tool": "none", "portfolio_name": null}'
        )
        tool, _ = self.agent._route("What's the weather today?")
        assert tool == "none"

    def test_llm_router_exposure(self, monkeypatch):
        monkeypatch.setattr("src.agent.is_available", lambda: True)
        monkeypatch.setattr(
            "src.agent.generate",
            lambda q, system_instruction="": '{"tool": "exposure_calculator", "portfolio_name": "Conservative Income Fund"}'
        )
        tool, pname = self.agent._route("Give me sector breakdown for Conservative Income Fund")
        assert tool == "exposure_calculator"
        assert pname == "Conservative Income Fund"



