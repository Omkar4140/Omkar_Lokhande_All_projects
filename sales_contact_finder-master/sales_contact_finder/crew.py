from crewai import Agent, Crew, Process, Task, LLM
import os

class SalesContactFinderCrew:
    """Sales Contact Finder crew"""

    def __init__(self):
        self.agents_list = []
        self.tasks_list = []
        
        # Initialize OpenRouter LLM
        self.llm = LLM(
            model="openrouter/mistralai/mistral-7b-instruct:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            max_tokens=600,
            temperature=0.3
        )

    def company_researcher(self) -> Agent:
        agent = Agent(
            role="Company Research Specialist",
            goal="Gather comprehensive information about the target company",
            backstory="""You are an expert at researching companies, with a keen eye for details that matter to sales professionals.
            Your task is to gather key information about the target company that will be relevant for identifying potential buyers of the product.""",
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def org_structure_analyst(self) -> Agent:
        agent = Agent(
            role="Organizational Structure Analyst",
            goal="Analyze the company's structure to identify key decision-making roles",
            backstory="""You understand corporate hierarchies and help map decision-makers.
            Analyze the target company's structure and highlight who might influence purchases of the product.""",
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def contact_finder(self) -> Agent:
        agent = Agent(
            role="Key Contact Identifier",
            goal="Find specific individuals in key roles at the target company",
            backstory="""You are skilled at identifying and locating individuals within organizations.
            Find names, titles, departments, and contact information for decision-makers at the target company.""",
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def sales_strategist(self) -> Agent:
        agent = Agent(
            role="Sales Approach Strategist",
            goal="Develop a strategy for approaching the identified contacts",
            backstory="""You're a sales expert. Craft a tailored outreach strategy for the target company that appeals to decision-makers and highlights the value of the product.""",
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )
        return agent

    def research_company_task(self, company_researcher_agent) -> Task:
        return Task(
            description="""Conduct thorough research on {target_company}. Focus on industry, size, recent news, and challenges where {our_product} could help.
            Look for relevant initiatives (digital transformation, procurement revamp, etc.).
            
            Use your knowledge to provide insights about the company's business model, market position, and potential pain points that {our_product} could address.""",
            expected_output="""## Company Overview
- Brief background on the target company
- Key business focus and industry
- Current challenges or initiatives
- Market position and recent developments
- Notes on potential fit for the product""",
            agent=company_researcher_agent,
        )

    def analyze_org_structure_task(self, org_analyst_agent) -> Task:
        return Task(
            description="""Analyze {target_company}'s organizational structure based on typical corporate hierarchies for companies in their industry and size.
            Identify relevant departments and roles (e.g., CTO, Head of Procurement, Operations, IT Director, etc.) that would be involved in purchasing decisions for {our_product}.""",
            expected_output="""## Departmental Overview
- List departments most relevant to buying decisions for this type of product
- Key decision-maker roles and their typical responsibilities
- How each department might be involved in the purchase process
- Reporting structure and influence patterns""",
            agent=org_analyst_agent,
        )

    def find_key_contacts_task(self, contact_finder_agent) -> Task:
        return Task(
            description="""Based on the company research and organizational analysis, identify the types of individuals at {target_company} who would have influence over purchasing {our_product}.
            Focus on likely job titles, departments, and roles rather than specific named individuals (since we don't have access to real-time contact databases).
            Provide guidance on where to find these contacts.""",
            expected_output="""## Target Contact Profiles
| Role Type | Likely Title | Department | Why Important | Where to Find |
|-----------|--------------|------------|---------------|---------------|
| Primary Decision Maker | e.g., CTO, VP of Operations | Technology/Operations | Budget authority | LinkedIn, company website |
| Technical Evaluator | e.g., IT Director, Senior Engineer | IT/Engineering | Technical assessment | Professional networks |
| Budget Influencer | e.g., Procurement Manager | Procurement/Finance | Cost evaluation | Industry events |

- Aim for 3–5 contact types most relevant to the identified roles
- Include strategies for finding and reaching these contacts""",
            agent=contact_finder_agent,
        )

    def develop_approach_strategy_task(self, sales_strategist_agent) -> Task:
        return Task(
            description="""Using all gathered data about {target_company} and {our_product}, develop a comprehensive contact and outreach strategy.
            Focus on messaging that resonates with each type of decision-maker and highlights the specific value {our_product} brings to {target_company}'s situation.""",
            expected_output="""## Outreach Strategy

### Messaging Framework
- Core value proposition tailored to {target_company}
- Key pain points {our_product} addresses
- Specific benefits for this company

### Persona-Based Approach
- Technical decision makers: Focus on features, integration, scalability
- Business decision makers: Focus on ROI, efficiency gains, competitive advantage
- Procurement/Finance: Focus on cost savings, implementation timeline, support

### Recommended Outreach Sequence
1. Initial contact method and messaging
2. Follow-up strategy
3. Key information to gather in first conversations
4. Common objections and responses

### Value Proposition Summary
- Why {our_product} is ideal for {target_company} specifically
- Unique differentiators relevant to their industry/situation
- Quantifiable benefits where possible

## Executive Summary
A concise overview combining all findings into actionable next steps for the sales team.""",
            agent=sales_strategist_agent,
            output_file="buyer_contact.md",
        )

    def crew(self) -> Crew:
        """Creates the SalesContactFinder crew"""
        
        # Create agents
        company_researcher_agent = self.company_researcher()
        org_analyst_agent = self.org_structure_analyst()
        contact_finder_agent = self.contact_finder()
        sales_strategist_agent = self.sales_strategist()
        
        # Create tasks with dependencies
        research_task = self.research_company_task(company_researcher_agent)
        org_task = self.analyze_org_structure_task(org_analyst_agent)
        contact_task = self.find_key_contacts_task(contact_finder_agent)
        strategy_task = self.develop_approach_strategy_task(sales_strategist_agent)
        
        # Set up task dependencies
        org_task.context = [research_task]
        contact_task.context = [research_task, org_task]
        strategy_task.context = [research_task, org_task, contact_task]
        
        return Crew(
            agents=[company_researcher_agent, org_analyst_agent, contact_finder_agent, sales_strategist_agent],
            tasks=[research_task, org_task, contact_task, strategy_task],
            process=Process.sequential,
            verbose=True,
        )
