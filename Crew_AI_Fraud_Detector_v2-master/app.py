import os
import streamlit as st
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from typing import List
import json

# Configuration
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# Pydantic Models
class FraudFlag(BaseModel):
    transaction_id: str = Field(..., description="Transaction identifier")
    risk_level: str = Field(..., description="HIGH, MEDIUM, or LOW")
    risk_score: float = Field(..., description="Risk score 0-100")
    reasons: List[str] = Field(..., description="List of risk factors")
    recommendation: str = Field(..., description="Action to take")

class FraudReport(BaseModel):
    total_transactions: int
    flagged_count: int
    high_risk_count: int
    flagged_transactions: List[FraudFlag]
    summary: str = Field(..., description="Executive summary")

# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=1500,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# Create Agents
@st.cache_resource
def create_agents():
    llm = get_llm()
    
    data_analyst = Agent(
        role="Transaction Data Analyst",
        goal="Analyze transaction patterns and identify anomalies",
        backstory="Expert in financial data analysis with 10+ years experience in fraud detection",
        llm=llm,
        verbose=True
    )
    
    fraud_investigator = Agent(
        role="Fraud Investigation Specialist", 
        goal="Investigate suspicious transactions and assess fraud risk",
        backstory="Senior fraud investigator with expertise in pattern recognition and risk assessment",
        llm=llm,
        verbose=True
    )
    
    compliance_officer = Agent(
        role="Compliance and Risk Officer",
        goal="Make final fraud determinations and compliance recommendations",
        backstory="Compliance expert ensuring all fraud flags meet regulatory standards",
        llm=llm,
        verbose=True
    )
    
    return data_analyst, fraud_investigator, compliance_officer

# UI
st.title("🔍 AI-Powered Fraud Detection System")
st.markdown("Detect fraudulent transactions using multi-agent AI analysis")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Data")
    
    # Sample data section
    with st.expander("📋 Sample Data (Click to Copy)", expanded=False):
        sample_data_text = """TXN001,John Doe,2500,2024-01-15,credit_card,USA
TXN002,Jane Smith,75000,2024-01-15,wire_transfer,Nigeria
TXN003,Bob Wilson,150,2024-01-15,debit_card,Canada
TXN004,Alice Brown,35000,2024-01-15,crypto,Unknown
TXN005,Mike Chen,500,2024-01-15,credit_card,Singapore
TXN006,Sarah Jones,95000,2024-01-15,wire_transfer,Russia
TXN007,Tom Davis,25,2024-01-15,paypal,USA
TXN008,Lisa Wang,45000,2024-01-15,bank_transfer,China"""
        
        st.code(sample_data_text, language=None)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 Use Sample Data", key="use_sample"):
                st.session_state.transaction_input = sample_data_text
                st.success("Sample data loaded! ✅")
        with col_b:
            st.write("👆 Copy the text above or click 'Use Sample Data'")
    
    # Get initial value
    initial_value = st.session_state.get('transaction_input', '')
    
    transaction_data = st.text_area(
        "Enter transaction data (one per line):",
        value=initial_value,
        placeholder="TXN001,John Doe,5000,2024-01-15,credit_card,USA\nTXN002,Jane Smith,50000,2024-01-15,wire_transfer,Nigeria",
        height=150,
        help="Format: ID,Customer,Amount,Date,Type,Location",
        key="main_input"
    )

with col2:
    st.subheader("Analysis Parameters")
    risk_threshold = st.slider("Risk Threshold", 0, 100, 70)
    analysis_type = st.selectbox("Analysis Type", ["Standard", "Enhanced", "Quick"])
    
    # Customer profile
    customer_context = st.text_area(
        "Customer Context (optional):",
        placeholder="Previous fraud history, account age, typical transaction patterns...",
        height=80
    )

# Analysis Button
if st.button("🚀 Analyze Transactions", type="primary"):
    if not transaction_data.strip():
        st.error("Please enter transaction data to analyze")
    else:
        with st.spinner("Analyzing transactions for fraud patterns..."):
            try:
                # Parse transaction data
                transactions = []
                for line in transaction_data.strip().split('\n'):
                    if line.strip():
                        transactions.append(line.strip())
                
                # Create agents
                data_analyst, fraud_investigator, compliance_officer = create_agents()
                
                # Create tasks
                analysis_task = Task(
                    description=f"""
                    Analyze the following {len(transactions)} transactions for suspicious patterns:
                    
                    Transaction Data:
                    {chr(10).join(transactions)}
                    
                    Customer Context: {customer_context or 'None provided'}
                    
                    Look for:
                    - Unusual amounts or frequencies
                    - High-risk locations or payment methods
                    - Velocity patterns
                    - Amount anomalies
                    
                    Provide a structured analysis of each transaction's risk factors.
                    """,
                    agent=data_analyst,
                    expected_output="Detailed analysis of transaction patterns and anomalies"
                )
                
                investigation_task = Task(
                    description=f"""
                    Based on the data analysis, investigate each transaction for fraud indicators.
                    
                    Risk Threshold: {risk_threshold}%
                    Analysis Type: {analysis_type}
                    
                    For each transaction, determine:
                    - Risk level (HIGH/MEDIUM/LOW)
                    - Specific fraud indicators
                    - Risk score (0-100)
                    - Supporting evidence
                    
                    Focus on transactions that exceed the {risk_threshold}% risk threshold.
                    """,
                    agent=fraud_investigator,
                    expected_output="Investigation report with risk assessment for each transaction"
                )
                
                final_report_task = Task(
                    description=f"""
                    Create a final fraud detection report based on the analysis and investigation.
                    
                    Generate a comprehensive report including:
                    - Total transactions analyzed: {len(transactions)}
                    - Number of flagged transactions
                    - High-risk transaction count
                    - Detailed findings for each flagged transaction
                    - Executive summary with key findings
                    - Compliance recommendations
                    
                    Only flag transactions that truly warrant investigation based on the evidence.
                    """,
                    agent=compliance_officer,
                    expected_output="Final fraud detection report with flagged transactions",
                    output_pydantic=FraudReport
                )
                
                # Create and run crew
                crew = Crew(
                    agents=[data_analyst, fraud_investigator, compliance_officer],
                    tasks=[analysis_task, investigation_task, final_report_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                result = crew.kickoff()
                
                # Display Results
                if hasattr(final_report_task.output, 'pydantic'):
                    report = final_report_task.output.pydantic
                    
                    # Summary Cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", report.total_transactions)
                    with col2:
                        st.metric("Flagged", report.flagged_count)
                    with col3:
                        st.metric("High Risk", report.high_risk_count)
                    with col4:
                        risk_rate = (report.flagged_count / report.total_transactions * 100) if report.total_transactions > 0 else 0
                        st.metric("Risk Rate", f"{risk_rate:.1f}%")
                    
                    # Executive Summary
                    st.subheader("📊 Executive Summary")
                    st.info(report.summary)
                    
                    # Flagged Transactions
                    if report.flagged_transactions:
                        st.subheader("🚨 Flagged Transactions")
                        
                        for flag in report.flagged_transactions:
                            risk_color = {
                                "HIGH": "🔴",
                                "MEDIUM": "🟡", 
                                "LOW": "🟢"
                            }.get(flag.risk_level, "⚪")
                            
                            with st.expander(f"{risk_color} {flag.transaction_id} - {flag.risk_level} Risk ({flag.risk_score:.0f}/100)"):
                                st.write("**Risk Factors:**")
                                for reason in flag.reasons:
                                    st.write(f"• {reason}")
                                
                                st.write("**Recommendation:**")
                                st.write(flag.recommendation)
                    else:
                        st.success("✅ No fraudulent transactions detected!")
                
                else:
                    st.error("Error processing the analysis results")
                    
            except Exception as e:
                error_msg = str(e)
                if "api" in error_msg.lower() or "key" in error_msg.lower():
                    st.error("API Error: Please check if GROQ_API_KEY is properly set in environment variables")
                elif "token" in error_msg.lower() or "limit" in error_msg.lower():
                    st.error("API Limit Reached: Please try again in a few minutes")
                else:
                    st.error(f"Analysis failed: {error_msg}")
                st.info("💡 Tip: Try reducing the number of transactions or simplify the input data")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI-powered fraud detection system uses:
    - **CrewAI**: Multi-agent orchestration
    - **Groq**: Fast LLM inference
    - **Streamlit**: Interactive UI
    
    The system analyzes transaction patterns using specialized AI agents for comprehensive fraud detection.
    """)
    
    st.header("🔧 Setup")
    
    # Check API key status
    api_key_status = "✅ Connected" if os.getenv("GROQ_API_KEY") else "❌ Not Set"
    st.write(f"**API Status:** {api_key_status}")
    
    st.write("""
    **Input Format:**
    Each line should contain:
    ID,Customer,Amount,Date,Type,Location
    """)
    
    if st.button("💡 Quick Start Guide"):
        st.info("""
        **How to test the app:**
        1. Click on '📋 Sample Data' above to expand
        2. Either copy the sample data or click 'Use Sample Data' 
        3. Adjust risk threshold (try 60 for more flags)
        4. Click 'Analyze Transactions'
        
        **Sample includes suspicious patterns like:**
        - Large wire transfers from high-risk countries
        - Crypto transactions from unknown locations
        - Unusual amount patterns
        """)

