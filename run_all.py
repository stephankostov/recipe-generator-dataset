from subprocess import run
from pathlib import Path
import argparse

exclusions = [3, 10, 99]

notebooks = list((Path.cwd()/'notebooks').glob(r"*[0-9]*.ipynb"))
notebooks.sort()

def main():

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--start_from', type=int, default=0)
    p.add_argument('-e','--exclude', nargs='+', required=False)
    args = p.parse_args()

    if args.exclude: exclusions.extend([int(x) for x in args.exclude])

    for notebook in notebooks:
        notebook_number = int(notebook.name.split('-')[0])
        if notebook_number < args.start_from or notebook_number in exclusions: continue
        print("================================================", f"RUNNING NOTEBOOK {notebook.name}", "================================================", "\n", sep="\n")
        result = run([f'jupyter nbconvert --execute --to notebook --inplace notebooks/{notebook.name}'], shell=True)
        if result.returncode != 0: 
            print(f"Notebook {notebook_number} returned with an error, breaking")
            break

if __name__ == '__main__':
    main()
