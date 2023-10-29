from atlassian import Confluence
from pathlib import Path
import os
import markdownify
import getpass


confluence_output_folder = 'doc'
cred_path = Path(".cred/config")


class ConflunenceParser:
    def __init__(self) -> None:
        if not cred_path.exists():
            self.write_cred()
        username, password = self.load_cred()
        self.session = Confluence(
                url="https://confluence.imgdev.bioclinica.com/",
                username=username,
                password=password
            )

    @staticmethod
    def write_cred():
        # write username / password that can log in to confluence
        print("USERNAME:")
        username = input()
        password = getpass.getpass("PASSWORD:")
        cred_path.parent.mkdir(exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(f"username={username}\n")
            f.write(f"password={password}\n")
        print(f"Write credential to {cred_path}")

    @staticmethod
    def load_cred():
        username = "NO USER"
        password = "NO PASSWORD"
        with open(cred_path, "r") as f:
            for line in f.read().splitlines(): 
                if "username=" in line:
                    username = line.split('=')[1]
                if "password=" in line:
                    password = line.split('=')[1]
        return username, password

    @staticmethod
    def save_html_to_markdown(html_string, output_path, heading_style="ATX"):
        convert = markdownify.markdownify(html_string, heading_style=heading_style)
        # make soure output folder exists
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            f.write(convert)
        print(f"Save markdown file: {output_path}")

    @staticmethod
    def save_pdf(content, output_path):
        with open(output_path, "wb") as pdf_file:
            pdf_file.write(content)
            pdf_file.close()
            print(f"Save PDF file: {output_path}")
    
    def save_page_md(self, space, title, md_path=""):
        page_id = self.session.get_page_id(space, title)
        page = self.session.get_page_by_id(page_id, expand='body.storage', status=None, version=None)
        body_html = page['body']['storage']['value']
        output_path = md_path if md_path else os.path.join(confluence_output_folder, title + ".md")
        self.save_html_to_markdown(body_html, output_path)

    def save_page_word(self, space, title, pdf_path=""):
        content = self.session.get_page_as_word(self.session.get_page_id(space, title))
        output_path = pdf_path if pdf_path else os.path.join(confluence_output_folder, title + ".pdf")
        with open(output_path, "wb") as pdf_file:
            pdf_file.write(content)
            pdf_file.close()
            print("Completed")
    
    def save_page_pdf(self, space, title, pdf_path=""):
        content = self.session.export_page(self.session.get_page_id(space, title))
        output_path = pdf_path if pdf_path else os.path.join(confluence_output_folder, title + ".pdf")
        self.save_pdf(content, output_path)

    def save_pages_by_label_pdf(self, label, folder_path = confluence_output_folder):
        pages = self.session.get_all_pages_by_label(label=label, start=0, limit=10)
        for page in pages:
            response = self.session.get_page_as_pdf(page["id"])
            output_path = os.path.join(folder_path ,page["title"] + ".pdf")
            self.save_pdf(response, output_path)
                        