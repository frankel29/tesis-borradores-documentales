# batch.py
import argparse
from pathlib import Path
from extractor import DocumentProcessor

# Carpetas por defecto — se crean automáticamente si no existen
CARPETA_ENTRADA = Path("documentos")
CARPETA_SALIDA  = Path("resultados")

def main():
    parser = argparse.ArgumentParser(description="Procesa múltiples PDFs en lote")
    parser.add_argument("--entrada", default=CARPETA_ENTRADA, help="Carpeta con los PDFs (default: documentos)")
    parser.add_argument("--output",  default=CARPETA_SALIDA,  help="Carpeta de salida  (default: resultados)")
    args = parser.parse_args()

    carpeta = Path(args.entrada)
    salida  = Path(args.output)

    # Crear carpetas si no existen
    carpeta.mkdir(exist_ok=True)
    salida.mkdir(exist_ok=True)

    pdfs = list(carpeta.glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en: {carpeta.resolve()}")
        print("Coloca tus PDFs en esa carpeta y vuelve a ejecutar.")
        return

    print(f"PDFs encontrados: {len(pdfs)}")

    processor = DocumentProcessor()
    exitosos, fallidos = 0, []

    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        try:
            result   = processor.process(pdf)
            out_path = salida / f"{pdf.stem}.json"
            processor.save_output(result, out_path)
            print(f"  ✓ {result.summary['total_entities']} entidades — {result.execution_time_seconds}s")
            exitosos += 1
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            fallidos.append(pdf.name)

    print(f"\n{'='*50}")
    print(f"  Procesados : {exitosos}/{len(pdfs)}")
    if fallidos:
        print(f"  Fallidos   : {', '.join(fallidos)}")
    print(f"  JSONs en   : {salida.resolve()}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()