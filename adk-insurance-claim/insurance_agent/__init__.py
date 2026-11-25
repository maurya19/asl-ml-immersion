"""
Insurance Claim Processing System using Google ADK.

This package implements a complete insurance claim processing system using
Google's Agent Development Kit (ADK). The system follows a microservices-like
architecture where each step in the claims process is handled by a dedicated
agent.

Key Components:
- InsuranceClaimProcessor: Main coordinator that orchestrates the workflow
- Multiple specialized agents for different processing steps
- FastAPI application for HTTP interface

Basic Usage:
    ```python
    from insurance_agent import InsuranceClaimProcessor

    # Create an instance of the processor
    processor = InsuranceClaimProcessor()

    # Process a claim
    result = await processor._process_claim_impl({
        "policy_number": "POL-123456",
        "claim_type": "auto",
        "incident_date": "2026-10-01",
        "claim_amount": 5000,
        "incident_details": "Car accident on Main St",
        "supporting_documents": ["doc1.pdf", "photo1.jpg"]
    })
    print(result)
    ```

For web interface, use the ADK CLI:
    ```bash
    adk web
    ```
    Then access the web interface at http://localhost:8000

Note:
    This package requires the Google ADK to be properly configured with
    appropriate API keys and permissions.
"""

from fastapi import FastAPI

from .agent import InsuranceClaimProcessor, insurance_processor, root_agent

# Create a FastAPI application instance
# This can be used to add additional API endpoints if needed
app = FastAPI(
    title="Insurance Claim Processing API",
    description="API for processing insurance claims using Google ADK",
    version="1.0.0",
)

# For backward compatibility and easier imports
__all__ = [
    "root_agent",
    "InsuranceClaimProcessor",
    "app",
    "insurance_processor",
]
