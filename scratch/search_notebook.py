import nbformat

with open("notebooks/TP2_Youtube_EDA.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        source = cell.source
        if "timestamp_jogo_segundos" in source or "Intervalo" in source or "mask" in source:
            print(f"Cell {idx}:")
            print(source[:200])
            print("-" * 40)
