"""
test_agent.py – Unit tests for the PortfolioAgent class.
"""

from src.agent import AgentResult, PortfolioAgent


class TestPortfolioAgent:
    """Tests for agent query routing, caching, and execution."""

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

