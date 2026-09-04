import pandas as pd

def sample_csv(input_path, output_path, n=None, frac=None):
    df = pd.read_csv(input_path)

    if n is None and frac is None:
        frac = 0.1  # default to 10% sample

    sample = df.sample(n=n, frac=frac, random_state=42)
    sample.to_csv(output_path, index=False)
    print(f"Sampled {len(sample)} rows from {len(df)} total ({len(sample)/len(df)*100:.1f}%)")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    input_path = input("Enter input CSV path: ")
    output_path = input("Enter output CSV path: ")
    mode = input("Sample by [N]umber or [P]ercentage?: ").upper()

    if mode == 'N':
        n = int(input("Enter number of rows: "))
        sample_csv(input_path, output_path, n=n)
    elif mode == 'P':
        frac = float(input("Enter percentage (e.g. 10 for 10%): ")) / 100
        sample_csv(input_path, output_path, frac=frac)
    else:
        print("Please enter N or P.")