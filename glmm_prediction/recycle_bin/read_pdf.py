import fitz
doc = fitz.open('../data/raw_data/xie_liu_jaeger21/NIH-preprint-for-Xie, Liu, Jaeger 2021.pdf')
print(doc.load_page(0).get_text()[:2000])
