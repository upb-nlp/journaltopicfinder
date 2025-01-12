import os
import re
import json
import traceback
from collections import Counter
from io import BytesIO, StringIO
from shutil import move
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Dict, List, Tuple, Union

# import camelot
import numpy as np
# import xmltodict
# from camelot.core import Table
from console_progressbar import ProgressBar
from pdfminer.high_level import extract_pages, extract_text, extract_text_to_fp
from pdfminer.layout import (LTFigure, LTImage, LTLine, LTPage,
                             LTTextBoxHorizontal, LTTextLine, LTTextLineHorizontal)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfparser import PDFParser

from parsers.springer_conf import SpringerConfParser
from parsers.ijcsclParser import IjcsclParser
from parsers.lak_sigcse import LakSigcseParser
from parsers.elsevier import ElsevierParser
from parsers.frontiers import FrontiersParser
from parsers.section import Paragraph, Section

from parsers.parser import Parser
from utils.utils import Heading

from PyPDF3 import PdfFileReader, PdfFileWriter
from PyPDF3.generic import Destination

YEAR_REGEX = [
    (re.compile(r"\[\]"), ""),
    (re.compile(r"--"), "00"),
    (re.compile(r"\d+-(\d+)"), "\1"),
    (re.compile(r"\D+"), ""),
]

heading_pattern = re.compile(r"^([0-9]+(\.)?)+\s*")

class LineWrapper:

    def __init__(self, line: LTTextLineHorizontal):
        self.inner = line

    def __lt__(self, other):
        if abs(self.inner.y0 - other.inner.y0) < 2:  # same line
            return self.inner.x0 < other.inner.x0
        return self.inner.y0 > other.inner.y0

def detect_page_limits(lines: List[LTTextLine]) -> Tuple[int, int]:
    c = Counter(round(line.x0 / 10) for line in lines)
    a, _ = c.most_common(1)[0]
    c = Counter(round(line.x1 / 10) for line in lines)
    b, _ = c.most_common(1)[0]
    return a * 10, b * 10


def is_centered(line: LTTextLine, left: int, right: int) -> bool:
    return abs(line.x0 - left) > 50 and abs(line.x1 - right) > 50


def enumerate_outlines(outlines: List[Union[Destination, List]], parent_index=None) -> List[Heading]:
    headings = []
    heading_index = 0
    for outline in outlines:
        if isinstance(outline, Destination):
            heading_index += 1
            headings.append(Heading(outline['/Title'], parent_index, heading_index, []))
        elif isinstance(outline, List):
            headings[-1].subheadings = enumerate_outlines(outline, heading_index)
    return headings


def get_pdf_outlines(pdf_name):
    with open(pdf_name, "rb") as f:
        pdf = PdfFileReader(f, strict=False)
        if len(pdf.outlines) == 0:
            return []
        outlines = enumerate_outlines(pdf.outlines[1:][0])
        return outlines


def extract_content(pdf_file: str, parser: Parser):
    # tables = extract_tables(pdf_file)
    pages = list(extract_pages(pdf_file))
    heading_list = get_pdf_outlines(pdf_file)
    last_line = None
    index = 0
    found_start = True
    result: List[Section] = []
    result_without_sections: List[str] = []
    for i, page in enumerate(pages):
        if not parser.check_page(page):
            continue
        for col in range(parser.cols):
            same_page = False
            lines = list(sorted(
                (line
                for obj in page
                if isinstance(obj, LTTextBoxHorizontal)
                for line in obj
                if parser.is_horizontal(line) and parser.check_limits(line, i, col)),
                key=LineWrapper))

            if not found_start:
                for j, line in enumerate(lines):
                    if parser.check_heading(line) and parser.heading_level(line) == 0:  # section = None
                        result.append(Section(line))
                        found_start = True
                        break
                lines = lines[(j + 1):]
            else: 
                lines = parser.exclude_tables_and_footnotes(lines, page, col)
                lines = parser.exclude_shapes(lines, page, col)
            # left, right = detect_page_limits(lines)
            # if is_centered(lines[0], left, right) and len(lines[0].get_text().strip()) < 5:
            #     lines = lines[1:]
            # if is_centered(lines[-1], left, right) and len(lines[-1].get_text().strip()) < 5:
            #     lines = lines[:-1]
            # footnotes = find_footnotes(lines)
            # if footnotes:
            #     print(f"page: {i}")
            for line in lines:
                line_text = line.get_text().lstrip()
                if not line_text:
                    continue
                result_without_sections.append(line_text)
                # if parser.check_heading(line):  # section = last_section()
                #     print("01")
                #     level = parser.heading_level(line)
                #     print("02")
                #     if not result[-1].add_heading(line, level):
                #         result.append(Section(line))
                #     continue
                # print("1")
                # print(result) # gol
                # if result[-1].last_paragraph() is None:
                #     result[-1].append_paragraph(Paragraph(line_text))
                #     last_line = line
                #     same_page = True
                #     continue
                # same line
                # print("2")
                # if same_page and line.y1 > last_line.y0 and line.x0 > last_line.x1 \
                #         and result[-1].last_paragraph().endswith("\n"):
                #     result[-1].last_paragraph().pop()
                # print("3")
                # if abs(last_line.x1 - right) < 50 and (same_page or last_line.y0 < 100):
                #     # separated word
                #     if not result[-1].last_paragraph().remove_hyphen():
                #         if result[-1].last_paragraph().same_paragraph(line_text):
                #             result[-1].last_paragraph().pop()
                # print("4") #nu ajunge aici
                # if result[-1].last_paragraph().endswith("\n"):
                #     index += 1
                #     result[-1].append_paragraph(Paragraph(line_text))
                # else:
                #     result[-1].last_paragraph().append(line_text)
    #             last_line = line
    #             same_page = True
    # for section in result:
    #     section.remove_empty()
    # result = [section for section in result if not section.empty()]
    return result_without_sections


def structure_section(section: Section, result: List[Dict], parent=None):
    index = len(result)
    obj = {
        "heading": re.sub(heading_pattern, "", section.heading),
        "index": index,
        "parent": parent,
        "text": "\n".join(str(par) for par in section.paragraphs)
    }
    result.append(obj)
    for subsection in section.subsections:
        structure_section(subsection, result, parent=index)


# def update_sections(filename: str, out_file: str, sections: List[Section]):
#     entry = get_entry_by_filename(filename, es)
#     if not entry:
#         print(f"Not found: {filename}")
#         raise Exception("Article not found")
#     result = []
#     for section in sections:
#         structure_section(section, result)
#     # print([(section["heading"], section["index"]) for section in result])
#     entry["_source"]["sections"] = result
#     es.update(index="articles", id=entry["_id"], doc=entry["_source"])


if __name__ == "__main__":
    parser = SpringerConfParser()
    errors = []
    output_file = "../ixdea_parsed.json"
    for edition in range(61):
        folder = f"../ixdea/{edition}"
        if not os.path.exists(folder):
            continue
        # empty_folder = f"../ixdea/{edition}-empty"
        errors_folder = f"../ixdea/{edition}-errors"
        files = [filename for filename in os.listdir(folder) if not filename.startswith(".")] 
        pb = ProgressBar(len(files))
        for filename in files:
            # if filename != "fpsyg.2011.00262.pdf":
            #     continue
            new_article = json.loads(json.dumps({"filename": filename, "text":[]}))
            # with open(os.path.join(folder, filename), "rb") as f:
            try:
                text = extract_content(os.path.join(folder, filename), parser)
                new_article["text"] = text
                with open(output_file, 'r') as openfile:
                    all_articles = json.load(openfile)
                all_articles['corpus'].append(new_article)
                with open(output_file, "w") as outfile:
                    outfile.write(json.dumps(all_articles))
                # if not sections:
                #     os.makedirs(empty_folder, exist_ok=True)
                #     os.rename(os.path.join(folder, filename), os.path.join(empty_folder, filename))
                # else:
                #     update_sections(filename, output_file, sections)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                errors.append(filename)
                continue
            pb.next()
    print(len(errors))
    print(errors)
