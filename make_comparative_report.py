import streamlit as st
from fpdf import FPDF
import matplotlib.pyplot as plt

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Regression Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def make_report():
    st.title("Generate Report")

    regression_results = st.session_state.get('regression_results', None)
    if regression_results is None:
        st.error("No regression results found.")
        return

    st.write("Generating a PDF report with the following regression results:")
    for key, value in regression_results.items():
        st.write(f"### {key}")
        for metric, result in value.items():
            if not isinstance(result, plt.Figure):
                st.write(f"- **{metric}**: {result}")

    if st.button("Download Report"):
        pdf = PDF()
        pdf.add_page()

        for key, value in regression_results.items():
            pdf.chapter_title(key)
            for metric, result in value.items():
                if isinstance(result, plt.Figure):
                    result.savefig(f'{metric}.png')
                    pdf.image(f'{metric}.png', x=10, w=100)
                else:
                    pdf.chapter_body(f"{metric}: {result}")

        pdf.output("regression_report.pdf")
        with open("regression_report.pdf", "rb") as f:
            st.download_button(label="Download PDF", data=f, file_name="regression_report.pdf")

if __name__ == "__main__":
    make_report()
