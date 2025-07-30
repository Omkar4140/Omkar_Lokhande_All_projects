import os
print("Working dir:", os.getcwd())
print("Files:", os.listdir("config"))

import streamlit as st
from sales_contact_finder.crew import SalesContactFinderCrew
import os

# Set up Streamlit page
st.set_page_config(page_title="Sales Contact Finder", page_icon="🔍")
st.title("🔍 Sales Contact Finder")

# Input fields
target_company = st.text_input("Target Company Name", placeholder="e.g. Google")
our_product = st.text_input("Your Product/Service", placeholder="e.g. AI-powered CRM software")

if st.button("Find Sales Contacts"):
    if not target_company or not our_product:
        st.warning("Please fill in all fields")
    else:
        with st.spinner("Finding the best sales contacts..."):
            try:
                # Run the crew
                inputs = {
                    "target_company": target_company,
                    "our_product": our_product,
                }
                result = SalesContactFinderCrew().crew().kickoff(inputs=inputs)
                
                # Display results
                st.success("Found the following contacts and strategy:")
                st.markdown(result)
                
                # Option to download results
                st.download_button(
                    label="Download Contact Strategy",
                    data=result,
                    file_name="sales_contacts.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
