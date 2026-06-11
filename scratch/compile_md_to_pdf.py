import os
import markdown
from playwright.sync_api import sync_playwright

def compile_pdf():
    md_path = "reports/TP3_Relatorio_Apresentacao.md"
    html_path = "reports/TP3_Relatorio_Apresentacao.html"
    pdf_path = "reports/TP3_Relatorio_Apresentacao.pdf"
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # Convert markdown to html using extra and tables extensions
    html_content = markdown.markdown(md_text, extensions=['extra', 'tables'])
    
    # Wrap in html template with styling
    template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TP3 - Relatório de Progresso</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            color: #2c3e50;
            line-height: 1.6;
            margin: 0;
            padding: 10px;
            background-color: #ffffff;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 24px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-top: 30px;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 18px;
            color: #2980b9;
            margin-top: 25px;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 14px;
            color: #34495e;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        p {{
            font-size: 13px;
            text-align: justify;
        }}
        ul, ol {{
            font-size: 13px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 8px 10px;
            font-size: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        img {{
            max-width: 90%;
            max-height: 250px;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            display: block;
            margin: 10px auto;
            page-break-inside: avoid;
        }}
        .row {{
            display: flex;
            justify-content: space-between;
            margin: 20px -10px;
            page-break-inside: avoid;
        }}
        .col {{
            flex: 1;
            margin: 0 10px;
            text-align: center;
        }}
        .col img {{
            max-width: 100%;
            max-height: 200px;
        }}
        .img-caption {{
            font-size: 10px;
            text-align: center;
            font-style: italic;
            margin-top: 6px;
            color: #7f8c8d;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #ecf0f1;
            margin: 30px 0;
        }}
        .prompt-box {{
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
            font-family: monospace;
            font-size: 10px;
            page-break-inside: avoid;
        }}
        .prompt-section {{
            padding: 10px 12px;
            border-bottom: 1px solid #cbd5e1;
            white-space: pre-wrap;
            text-align: left;
        }}
        .prompt-section:last-child {{
            border-bottom: none;
        }}
        .role {{
            background-color: #f0f7ff;
            border-left: 5px solid #2563eb;
            color: #1e3a8a;
        }}
        .general-rule {{
            background-color: #fffbeb;
            border-left: 5px solid #d97706;
            color: #78350f;
        }}
        .categories {{
            background-color: #fef2f2;
            border-left: 5px solid #dc2626;
            color: #7f1d1d;
        }}
        .decision-guidelines {{
            background-color: #faf5ff;
            border-left: 5px solid #9333ea;
            color: #581c87;
        }}
        .few-shot {{
            background-color: #f0fdf4;
            border-left: 5px solid #16a34a;
            color: #14532d;
        }}
        .output-format {{
            background-color: #f8fafc;
            border-left: 5px solid #475569;
            color: #0f172a;
        }}
        .math-equation {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            color: #2c3e50;
            margin: 25px 0;
            font-weight: 500;
            page-break-inside: avoid;
        }}
        .math-fraction {{
            display: inline-flex;
            flex-direction: column;
            vertical-align: middle;
            text-align: center;
            margin: 0 4px;
        }}
        .math-numerator {{
            border-bottom: 1.5px solid #2c3e50;
            padding: 0 6px 2px 6px;
            font-size: 11px;
        }}
        .math-denominator {{
            padding: 2px 6px 0 6px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
"""
    # Write temporary HTML file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(template)
        
    print(f"Temporary HTML written to {html_path}")
    
    # Use playwright to convert HTML to PDF
    print("Launching Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page()
        
        # Open HTML file
        abs_html_path = os.path.abspath(html_path)
        page.goto(f"file://{abs_html_path}")
        
        # Wait a small bit for images/fonts to render
        page.wait_for_timeout(2000)
        
        # Print to PDF
        page.pdf(
            path=pdf_path,
            format="A4",
            margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
            print_background=True
        )
        
        browser.close()
        
    print(f"PDF successfully created: {pdf_path}")

if __name__ == "__main__":
    compile_pdf()
