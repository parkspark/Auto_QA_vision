import zipfile
import xml.etree.ElementTree as ET
import sys
import re

pptx_path = r'c:\Users\park\Desktop\MINI_DATA_PROJECT\reports\비전 기반 게임 QA 자동화 발표.pptx'
try:
    with open('pptx_content.txt', 'w', encoding='utf-8') as f:
        with zipfile.ZipFile(pptx_path) as z:
            slides = [f_name for f_name in z.namelist() if f_name.startswith('ppt/slides/slide') and f_name.endswith('.xml')]
            
            def get_slide_num(name):
                match = re.search(r'slide(\d+)\.xml', name)
                return int(match.group(1)) if match else 0
                
            slides.sort(key=get_slide_num)
            
            for slide in slides:
                f.write(f"--- {slide} ---\n")
                xml_content = z.read(slide)
                root = ET.fromstring(xml_content)
                ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
                texts = root.findall('.//a:t', ns)
                slide_text = [t.text for t in texts if t.text]
                f.write(" ".join(slide_text) + "\n\n")
except Exception as e:
    print("Error:", e)
