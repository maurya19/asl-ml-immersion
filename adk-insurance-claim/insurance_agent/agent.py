"""
Insurance Claim Processing System using Google ADK.

This module implements the main coordinator for processing insurance claims
using multiple specialized agents with both sequential and parallel workflows.
The system follows a microservices-like architecture where each step in the
claims process is handled by dedicated agents that can run sequentially or
in parallel as needed.

Key Components:
- InsuranceClaimProcessor: Main coordinator that orchestrates the workflow
- process_claim: Tool exposing claim processing to the root agent
- ParallelAgent: For running independent validations concurrently
- Multiple specialized agents for different processing steps

Workflow:
1. Data Ingestion: Extract and validate claim data (sequential)
2. Parallel Execution:
   - Policy Validation: Verify claim against policy terms
   - Fraud Check: Analyze claim for potential fraud
3. Payout Estimation: Calculate estimated payout amount (sequential)

Implementation Patterns:
- Sequential Flow: Used for dependent steps that require the output of
  previous steps (e.g., data ingestion before validation)
- Parallel Flow: Used for independent validations that can run concurrently
  (e.g., policy validation and fraud check)

Dependencies:
- google-adk: Google's Agent Development Kit
- pydantic: For data validation and modeling
"""

import logging
import uuid
from typing import Any, AsyncGenerator, Dict

from google.adk.agents import Agent
from google.adk.agents import BaseAgent as AdkBaseAgent
from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import Field

# Relative imports for subagents
from .subagents.data_ingestion import DataIngestionAgent
from .subagents.fraud_check import FraudCheckAgent
from .subagents.payout_estimator import PayoutEstimatorAgent
from .subagents.policy_validation import PolicyValidationAgent

logger = logging.getLogger(__name__)


def _state_event(author: str, **updates: Any) -> Event:
    """Build an event that writes ``updates`` into the ADK session state."""
    return Event(author=author, actions=EventActions(state_delta=updates))


class _WorkflowStep(AdkBaseAgent):
    """Base for ADK workflow steps.

    Each step reads its inputs from the session state and writes its outputs
    back with a state-delta event, so the following steps can see them.
    """

    processor: Any = Field(default=None, exclude=True)


class IngestStep(_WorkflowStep):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        result = await self.processor.data_ingestion_agent.process(
            ctx.session.state["claim_data"]
        )
        if not result.success:
            raise ValueError(f"Data ingestion failed: {result.error}")
        yield _state_event(
            self.name, ingested_claim=result.data.get("ingested_claim", {})
        )


class PolicyValidationStep(_WorkflowStep):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # PolicyValidationAgent expects {"ingested_claim": {...}}
        result = await self.processor.policy_validation_agent.process(
            {"ingested_claim": ctx.session.state["ingested_claim"]}
        )
        yield _state_event(
            self.name,
            policy_validation=result.data.get("policy_validation", {}),
        )


class FraudCheckStep(_WorkflowStep):
    """Runs the fraud check and records its analysis in session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        claim_data = state["claim_data"]
        result = await self.processor.fraud_check_agent.process(
            {
                "ingested_claim": state["ingested_claim"],
                "claim_data": claim_data,
                "vehicle": claim_data.get("vehicle", {}),
            }
        )
        yield _state_event(
            self.name, fraud_analysis=result.data.get("fraud_analysis", {})
        )


class PayoutStep(_WorkflowStep):
    """Calculates the payout and records the outcome in session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        payout_input = {
            key: state.get(key)
            for key in (
                "claim_data",
                "ingested_claim",
                "policy_validation",
                "fraud_analysis",
            )
        }
        result = await self.processor.payout_estimator_agent.process(
            payout_input
        )
        if result.success:
            yield _state_event(
                self.name,
                payout_estimation=result.data.get("payout_estimation", {}),
            )
        elif "claim_denied" in result.data or result.data.get(
            "needs_manual_review"
        ):
            # A denial or manual-review flag is an outcome, not a failure.
            yield _state_event(
                self.name,
                payout_estimation={"outcome": result.error, **result.data},
            )
        else:
            raise ValueError(f"Payout calculation failed: {result.error}")


ROOT_INSTRUCTION = (
    "You are an automated insurance claim processor. "
    "Your role is to process claims through the workflow.\n\n"
    "RULES:\n"
    "1. When you receive a claim, IMMEDIATELY call process_claim "
    "with the provided data\n"
    "2. NEVER ask for confirmation - process all claims automatically\n"
    "3. The system will handle validation and fraud detection\n"
    "4. ALWAYS respond with the final claim processing result\n\n"
    "Example usage:\n"
    "User: {claim data}\n"
    "You: [Call process_claim with the data]\n"
    "System: [Processing result]\n"
    "You: Here's the processing result: [summary]"
)


class InsuranceClaimProcessor:
    """
    Main coordinator for the insurance claim processing workflow.

    Workflow: ingest -> (policy validation || fraud check) -> payout estimation.
    The root LLM agent exposes a single tool, ``process_claim``, which runs the
    ADK workflow (``run_workflow``) and turns the final session state into a
    claim decision (``_process_claim_impl``).
    """

    def __init__(self):
        # Specialized agents, one per processing step
        self.data_ingestion_agent = DataIngestionAgent()
        self.policy_validation_agent = PolicyValidationAgent()
        self.fraud_check_agent = FraudCheckAgent()
        self.payout_estimator_agent = PayoutEstimatorAgent()

        # Sequential/parallel ADK workflow that runs the steps
        self.workflow = self._create_workflow()

        async def process_claim(claim_data: dict) -> dict:
            """
            Process an insurance claim through the complete workflow.

            Args:
                claim_data: Claim information, with keys such as:
                    - policy_number: The policy number (e.g. "POL-123456")
                    - claim_type: "auto", "home" or "health"
                    - claim_amount: The amount being claimed
                    - incident_date: Date of the incident (YYYY-MM-DD)
                    - incident_details: Description of the incident
                    - supporting_documents: List of document references
                    - vehicle: Vehicle details (auto claims)

            Returns:
                A dictionary containing:
                - status: "approved", "rejected", "requires_review" or "error"
                - steps: Results from each processing step
                - approved_amount / currency: Only when status is "approved"
                - error: Reason, when the claim is not approved
            """
            return await self._process_claim_impl(claim_data)

        # Root agent that hands every claim to the workflow via the tool
        self.root_agent = Agent(
            name="insurance_claim_processor",
            model="gemini-3.5-flash",
            description="Coordinates the insurance claim processing workflow",
            instruction=ROOT_INSTRUCTION,
            tools=[FunctionTool(process_claim)],
        )

    def _create_parallel_agents(self) -> ParallelAgent:
        """Build a ParallelAgent for policy validation and fraud check."""
        return ParallelAgent(
            name="validation_and_fraud_check",
            description="Runs policy validation and fraud check in parallel",
            sub_agents=[
                PolicyValidationStep(
                    name="policy_validation",
                    description="Verifies the claim against policy terms",
                    processor=self,
                ),
                FraudCheckStep(
                    name="fraud_check",
                    description="Analyzes the claim for potential fraud",
                    processor=self,
                ),
            ],
        )

    def _create_workflow(self) -> SequentialAgent:
        """Build the SequentialAgent: ingest, parallel checks, payout."""
        return SequentialAgent(
            name="claim_processing_workflow",
            description="Processes claims through validation steps",
            sub_agents=[
                IngestStep(
                    name="ingest_claim_data",
                    description="Ingests and validates claim data",
                    processor=self,
                ),
                self._create_parallel_agents(),
                PayoutStep(
                    name="calculate_payout",
                    description="Calculates the final payout amount",
                    processor=self,
                ),
            ],
        )

    async def run_workflow(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow for a claim; return the final session state."""
        app_name, user_id = "insurance_claims", "claims_processor"
        claim_data = dict(claim_data)
        if "claim_id" not in claim_data:
            claim_data["claim_id"] = f"CLAIM-{str(uuid.uuid4())[:8].upper()}"
        runner = InMemoryRunner(agent=self.workflow, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name, user_id=user_id, state={"claim_data": claim_data}
        )
        trigger = types.Content(
            role="user",
            parts=[types.Part(text="Process the claim in session state.")],
        )
        async for _ in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=trigger
        ):
            pass
        final = await runner.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )
        return dict(final.state)

    async def _process_claim_impl(
        self, claim_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the workflow for a claim and translate its state into a decision.

        Returns a dict with ``status`` ("approved", "rejected",
        "requires_review" or "error"), ``claim_id``, ``steps`` and, depending
        on the outcome, ``approved_amount``/``currency`` (approved only) or
        ``error``.
        """
        claim_data = dict(claim_data or {})
        claim_data.setdefault(
            "claim_id", f"CLAIM-{str(uuid.uuid4())[:8].upper()}"
        )
        result: Dict[str, Any] = {
            "status": "processing",
            "claim_id": claim_data["claim_id"],
            "steps": {},
        }

        # claim_amount must be a positive number; reject anything else early.
        try:
            claim_data["claim_amount"] = float(claim_data.get("claim_amount"))
            if claim_data["claim_amount"] <= 0:
                raise ValueError
        except (TypeError, ValueError):
            bad_amount = claim_data.get("claim_amount")
            result["status"] = "rejected"
            result["error"] = (
                f"Invalid claim_amount: {bad_amount!r} "
                "(must be a positive number)"
            )
            return result

        try:
            state = await self.run_workflow(claim_data)
        except ValueError as ve:
            # Raised by workflow steps for unusable input, e.g. missing fields.
            logger.warning("Claim %s rejected: %s", result["claim_id"], ve)
            result["status"] = "rejected"
            result["error"] = str(ve)
            return result
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception(
                "Unexpected error processing claim %s", result["claim_id"]
            )
            result["status"] = "error"
            result["error"] = f"Unexpected error: {e}"
            return result

        fraud = state.get("fraud_analysis") or {}
        payout = state.get("payout_estimation") or {}
        result["steps"] = {
            "data_ingestion": state.get("ingested_claim", {}),
            "policy_validation": state.get("policy_validation", {}),
            "fraud_check": fraud,
            "payout_estimation": payout,
        }

        risk_score = fraud.get("risk_score", 0)
        needs_review = fraud.get("needs_review", False)
        indicators = fraud.get("fraud_indicators") or []
        indicator_text = ", ".join(indicators)
        logger.info(
            "[FRAUD ANALYSIS] Risk Score: %s, Needs Review: %s",
            risk_score,
            needs_review,
        )

        if "claim_denied" in payout:
            outcome = payout.get("outcome")
            denied = payout["claim_denied"]
            result["status"] = "rejected"
            result["error"] = f"{outcome}: {denied}"
        elif payout.get("needs_manual_review"):
            result["status"] = "requires_review"
            result["error"] = payout.get("outcome") or "Requires manual review"
        elif indicators and risk_score >= 0.7:
            result["status"] = "rejected"
            result["error"] = (
                f"Claim rejected due to high fraud risk (score: {risk_score}). "
                f"Indicators: {indicator_text}"
            )
        elif indicators and (needs_review or risk_score >= 0.4):
            result["status"] = "requires_review"
            result["error"] = (
                "Claim requires manual review due to potential "
                f"fraud indicators (score: {risk_score}). "
                f"Indicators: {indicator_text}"
            )
        elif payout.get("approved_amount", 0) > 0:
            result["status"] = "approved"
            result["approved_amount"] = payout["approved_amount"]
            result["currency"] = payout.get("currency", "USD")
            if indicators:
                result["warning"] = (
                    "Claim approved but with fraud indicators "
                    f"(score: {risk_score}). "
                    f"Indicators: {indicator_text}"
                )
        else:
            result["status"] = "requires_review"
            result["error"] = (
                "Payout estimation failed or resulted in zero amount"
            )

        return result


# Create a global instance for easy access
insurance_processor = InsuranceClaimProcessor()
root_agent = insurance_processor.root_agent
