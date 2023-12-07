from pathlib import Path
from urllib.request import urlopen
from random import randint, choices


def main():
    alph: list[str] = sorted(urlopen('https://www.mit.edu/~ecprice/wordlist.10000')
                             .read()
                             .decode('utf-8')
                             .split('\n'),
                             key=len)[396:]

    inputs_dir = Path('inputs')
    inputs_dir.mkdir(exist_ok=True)

    for i in range(int(input('> Count of inputs: '))):
        open(inputs_dir / f'input{i + 1}.txt', 'w').write(
            '\n'.join(
                choices(
                    alph,
                    weights=[2] * 2201 + [4] * 5513 + [1] * 1891,
                    k=randint(7, 12)
                )
            ) + '\n'
        )


if __name__ == '__main__':
    main()
